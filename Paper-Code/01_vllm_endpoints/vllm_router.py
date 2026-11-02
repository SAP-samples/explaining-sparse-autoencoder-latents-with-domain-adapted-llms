#!/usr/bin/env python3
"""
vllm_router.py

Client-side router / load-balancer for multiple vLLM instances serving the SAME model.

vLLM has no built-in cross-server load balancing — each `vllm serve` process is
independent and knows nothing about the others. This module distributes requests
across a pool of instance URLs so that each instance runs at most `per_instance`
concurrent requests (match this to the instance's --max-num-seqs).

Two policies:
  - "fill": a shared async slot pool across all instances. A request grabs the next
    free slot on whichever instance has capacity. Saturates the whole pool
    automatically without polling. (default, recommended for bulk feature jobs)
  - "roundrobin": request i -> instance[i % n]. Simple, no capacity awareness.

Usage (as a library):

    from vllm_router import VLLMRouter

    router = VLLMRouter(
        endpoints=[
            "http://gpu-dy-g524xlarge-15:18001/v1",
            "http://gpu-dy-g524xlarge-XX:18001/v1",
        ],
        model="qwen3-8b",
        per_instance=32,
    )
    results = router.complete_many(
        prompts=[...],          # list of prompt strings
        max_tokens=40,
        temperature=0.7,
    )

    # logprobs for specific next tokens (Pred Acc / Detection):
    lp = router.completion(prompt, max_tokens=1, logprobs=20)

CLI self-test:
    python vllm_router.py --endpoints URL1 URL2 --model qwen3-8b --n 100
"""
import argparse
import asyncio
import concurrent.futures
import json
import time
import urllib.request
import urllib.error


class VLLMRouter:
    def __init__(self, endpoints, model, per_instance=32, policy="fill", timeout=36000):
        if not endpoints:
            raise ValueError("endpoints must be a non-empty list")
        self.endpoints = [e.rstrip("/") for e in endpoints]
        self.model = model
        self.per_instance = per_instance
        self.policy = policy
        self.timeout = timeout

    # ---- low-level blocking HTTP call (runs in a thread) ----
    def _post(self, base_url, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url}{path}", data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    # ---- single synchronous completion (convenience) ----
    def completion(self, prompt, endpoint=None, **kwargs):
        base = endpoint or self.endpoints[0]
        payload = {"model": self.model, "prompt": prompt, **kwargs}
        return self._post(base, "/completions", payload)

    def chat(self, messages, endpoint=None, **kwargs):
        base = endpoint or self.endpoints[0]
        payload = {"model": self.model, "messages": messages, **kwargs}
        return self._post(base, "/chat/completions", payload)

    # ---- bulk: distribute many prompts across the instance pool ----
    def complete_many(self, prompts, **kwargs):
        """Return a list of /completions responses aligned with `prompts` (None on error)."""
        return asyncio.run(self._dispatch_many("/completions", "prompt", prompts, kwargs))

    def chat_many(self, messages_list, **kwargs):
        """Return a list of /chat/completions responses aligned with `messages_list`."""
        return asyncio.run(self._dispatch_many("/chat/completions", "messages", messages_list, kwargs))

    async def _dispatch_many(self, path, body_key, items, kwargs):
        n = len(self.endpoints)
        # one semaphore per instance, each sized to that instance's capacity
        sems = [asyncio.Semaphore(self.per_instance) for _ in self.endpoints]
        loop = asyncio.get_event_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=n * self.per_instance + 8)
        )
        results = [None] * len(items)

        async def worker(idx, item):
            if self.policy == "roundrobin":
                inst = idx % n
                async with sems[inst]:
                    results[idx] = await self._call(loop, inst, path, body_key, item, kwargs)
            else:  # "fill": try instances in order, take first with a free slot
                while True:
                    for inst in range(n):
                        if sems[inst].locked() and sems[inst]._value == 0:
                            continue
                        # try to acquire without blocking the whole loop
                        try:
                            await asyncio.wait_for(sems[inst].acquire(), timeout=0.01)
                        except asyncio.TimeoutError:
                            continue
                        try:
                            results[idx] = await self._call(loop, inst, path, body_key, item, kwargs)
                        finally:
                            sems[inst].release()
                        return
                    await asyncio.sleep(0.005)  # all busy, back off briefly

        await asyncio.gather(*(worker(i, p) for i, p in enumerate(items)))
        return results

    async def _call(self, loop, inst, path, body_key, item, kwargs, max_retries=3, backoff=2.0):
        base = self.endpoints[inst]
        payload = {"model": self.model, body_key: item, **kwargs}
        def do():
            last_err = None
            for attempt in range(max_retries):
                try:
                    return self._post(base, path, payload)
                except urllib.error.HTTPError as e:
                    body = e.read().decode()[:500]
                    last_err = f"HTTP {e.code}: {body}"
                    if attempt < max_retries - 1:
                        time.sleep(backoff * (attempt + 1))
                except Exception as e:
                    last_err = e
                    if attempt < max_retries - 1:
                        time.sleep(backoff * (attempt + 1))
            return {"error": str(last_err), "endpoint": base}
        return await loop.run_in_executor(None, do)


def _selftest(endpoints, model, n, per_instance, policy):
    router = VLLMRouter(endpoints, model, per_instance=per_instance, policy=policy)
    prompts = [f"Describe SAP feature {i} in one sentence:" for i in range(n)]
    t0 = time.time()
    res = router.complete_many(prompts, max_tokens=40, temperature=0.7)
    dt = time.time() - t0
    errs = [r for r in res if r is None or "error" in (r or {})]
    toks = sum(r["usage"]["completion_tokens"] for r in res
               if r and "usage" in r)
    print(f"policy={policy} | endpoints={len(endpoints)} | {n} reqs "
          f"| {dt:.1f}s | {(n-len(errs))/dt:.1f} req/s | {toks/dt:.0f} tok/s "
          f"| errors={len(errs)}")
    if errs:
        first = next(e for e in res if e is None or "error" in (e or {}))
        print(f"    first error: {first}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="+", required=True,
                    help="Full /v1 URLs, e.g. http://node:18001/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--per-instance", type=int, default=32)
    ap.add_argument("--policy", choices=["fill", "roundrobin"], default="fill")
    args = ap.parse_args()
    _selftest(args.endpoints, args.model, args.n, args.per_instance, args.policy)
