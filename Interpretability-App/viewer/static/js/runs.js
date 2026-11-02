let selectedRunId = null;
let pollTimer = null;

const STATE_ICON = {
  done: '✓', running: '▶', queued: '⏳', failed: '✗',
  pending: '○', skipped: '–', cancelled: '✗', unknown: '?', submit_failed: '✗'
};

async function loadRuns() {
  const list = document.getElementById('run-list');
  try {
    const res = await fetch('/api/runs');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const runs = await res.json();
    if (!runs.length) { list.innerHTML = '<div style="padding:20px;color:var(--muted);font-size:13px">No runs yet. Click + New Run.</div>'; return; }
    list.innerHTML = '';
    runs.forEach(run => {
      const div = document.createElement('div');
      div.className = 'run-item' + (run.run_id === selectedRunId ? ' selected' : '');
      div.dataset.runId = run.run_id;
      div.setAttribute('role', 'button');
      div.setAttribute('tabindex', '0');
      div.onclick = () => selectRun(run.run_id);
      div.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectRun(run.run_id); } };
      const ma = run.config.model_a.split('/').slice(-1)[0];
      const mb = run.config.model_b.split('/').slice(-1)[0];
      const runName = run.config.name || (ma + ' → ' + mb);
      div.innerHTML = `
        <div class="ri-id">${run.run_id}</div>
        <div class="ri-models">${runName}</div>
        <div class="ri-meta">
          <span style="color:var(--muted);font-size:11px">${ma} → ${mb}</span>
        </div>
        <div class="ri-meta">
          <span>layerA ${run.config.layer_a ?? run.config.layer}</span>
          <span>layerB ${run.config.layer_b ?? run.config.layer}</span>
          <span>${run.config.n_features.toLocaleString()} features</span>
          <span class="state-badge state-${run.overall}">${run.overall}</span>
        </div>`;
      list.appendChild(div);
    });
  } catch(e) {
    list.innerHTML = '<div style="padding:20px;color:#ef4444;font-size:13px">Failed to load runs: ' + e.message + '</div>';
  }
}

async function selectRun(runId) {
  selectedRunId = runId;
  document.querySelectorAll('.run-item').forEach(el => el.classList.toggle('selected', el.dataset.runId === runId));
  document.getElementById('run-placeholder').style.display = 'none';
  document.getElementById('run-detail').style.display = '';
  document.getElementById('run-detail').innerHTML = '<div style="color:var(--muted);padding:40px">Loading…</div>';
  await refreshRunDetail(runId);
  // Poll if not done
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`/api/runs/${runId}`);
      if (!res.ok) return;
      const run = await res.json();
      if (['done','failed'].includes(run.overall)) { clearInterval(pollTimer); }
      // Guard against race: only update UI if this run is still selected
      if (selectedRunId === runId) { renderRunDetail(run); loadRuns(); }
    } catch(e) {}
  }, 8000);
}

async function refreshRunDetail(runId) {
  try {
    const res = await fetch(`/api/runs/${runId}`);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    renderRunDetail(data);
    // Log is shown by default; mark it open and start fetching (only on first load)
    if (!_logOpen[data.run_id]) {
      _logOpen[data.run_id] = true;
      fetchLog(data.run_id);
    }
  } catch(e) {
    if (selectedRunId === runId) {
      document.getElementById('run-detail').innerHTML = '<div style="padding:40px;color:#ef4444">Failed to load run: ' + e.message + '</div>';
    }
  }
}

function renderRunDetail(run) {
  const cfg = run.config, st = run.status, art = run.artifacts;
  const steps = ['00_generate_dataset','01_collect_acts_A','01_collect_acts_B','02_train_crosscoder','03_collect_cc_acts','04_llmaaj','04_fve_dead_neurons'];
  const stepsHtml = steps.map(s => {
    const info = st[s] || {state:'pending',job_id:null};
    const cls = `s-${info.state}`;
    const icon = STATE_ICON[info.state] || '?';
    const job = info.job_id ? `<span class="step-job">[job ${info.job_id}]</span>` : '';
    const FAIL_STATES = ['failed','cancelled','submit_failed'];
    const errBtn = FAIL_STATES.includes(info.state)
      ? `<button class="step-err-btn" onclick="showStepLog('${run.run_id}','${s}')">Show error log</button>`
      : '';
    return `<div class="step-row">
      <span class="step-icon ${cls}">${icon}</span>
      <span class="step-name">${s}</span>
      ${job}
      <span class="step-state ${cls}">${info.state}</span>
      ${errBtn}
    </div>`;
  }).join('');

  const artHtml = Object.entries(art).map(([k,v]) => {
    const path = v ? v.path : '';
    const exists = v && v.exists;
    const short = path ? path.split('/').slice(-1)[0] : '—';
    const existMark = exists
      ? `<span class="art-exists">✓</span>`
      : `<span class="art-missing">✗</span>`;
    const copyBtn = path
      ? `<button class="art-copy" onclick="copyArtPath(this,'${path.replace(/'/g,"\\'")}')">⧉</button>`
      : '';
    return `<div class="artifact-row"><span class="art-key">${k}</span>${existMark}<span class="art-val" title="${path}">${short}</span>${copyBtn}</div>`;
  }).join('');

  // FVE metrics: async load after render
  let fveHtml = '';
  if (art.fve_json && art.fve_json.exists) {
    fveHtml = `<div class="card" id="fve-card-${run.run_id}"><h3>FVE + Dead Neurons</h3><div style="color:var(--muted);font-size:12px">Loading metrics…</div></div>`;
  } else if (art.fve_json && art.fve_json.path) {
    fveHtml = `<div class="card"><h3>FVE + Dead Neurons</h3><div style="color:var(--muted);font-size:12px">Job not completed yet.</div></div>`;
  }

  const isDone = run.overall === 'done';
  document.getElementById('run-detail').innerHTML = `
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;flex-wrap:wrap">
      <div>
        <div style="font-size:18px;font-weight:700">${cfg.name || cfg.model_a.split('/').slice(-1)[0] + ' → ' + cfg.model_b.split('/').slice(-1)[0]}</div>
        <div style="font-size:12px;color:var(--muted);margin-top:2px">${cfg.model_a.split('/').slice(-1)[0]} → ${cfg.model_b.split('/').slice(-1)[0]}</div>
        <div style="font-size:11px;color:var(--muted);font-family:monospace;margin-top:2px">${run.run_id}</div>
      </div>
      <span class="state-badge state-${run.overall}" style="font-size:13px;padding:4px 12px">${run.overall}</span>
      ${isDone ? `<button class="load-btn" onclick="loadRunInViewer('${run.run_id}')">Load in Feature Browser</button>` : ''}
      ${isDone ? `<a href="/api/runs/${run.run_id}/download_csv" download style="margin-left:8px;background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:5px 14px;font-size:12px;color:var(--text);text-decoration:none;vertical-align:middle">↓ CSV</a>` : ''}
    </div>

    <div class="card">
      <h3>Configuration</h3>
      <div class="metrics-grid">
        <div class="metric"><div class="m-label">Model A</div><div class="m-value" style="font-size:13px">${cfg.model_a.split('/').slice(-1)[0]}</div></div>
        <div class="metric"><div class="m-label">Model B</div><div class="m-value" style="font-size:13px">${cfg.model_b.split('/').slice(-1)[0]}</div></div>
        <div class="metric"><div class="m-label">Layer A</div><div class="m-value">${cfg.layer_a ?? cfg.layer ?? '?'}</div></div>
        <div class="metric"><div class="m-label">Layer B</div><div class="m-value">${cfg.layer_b ?? cfg.layer ?? '?'}</div></div>
        <div class="metric"><div class="m-label">n_features</div><div class="m-value">${cfg.n_features.toLocaleString()}</div></div>
        <div class="metric"><div class="m-label">Exp. factor</div><div class="m-value">${cfg.expansion_factor ?? 16}×</div></div>
        <div class="metric"><div class="m-label">TopK (k)</div><div class="m-value">${cfg.topk}</div></div>
        <div class="metric"><div class="m-label">Total tokens</div><div class="m-value" style="font-size:14px">${cfg.total_tokens >= 1e6 ? (cfg.total_tokens/1e6).toFixed(0)+'M' : cfg.total_tokens >= 1e3 ? (cfg.total_tokens/1e3).toFixed(0)+'K' : cfg.total_tokens}</div></div>
      </div>
    </div>

    <div class="card">
      <h3>Pipeline Steps</h3>
      ${stepsHtml}
    </div>

    ${fveHtml}

    <div class="card">
      <h3>Artifacts</h3>
      ${artHtml}
    </div>

    <div class="card" id="log-card-${run.run_id}">
      <h3 style="display:flex;align-items:center;gap:10px">
        Pipeline Log
        <span style="font-size:11px;color:var(--muted);font-weight:400" id="log-meta-${run.run_id}"></span>
        <button onclick="toggleLog('${run.run_id}')" style="margin-left:auto;background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:2px 10px;font-size:12px;cursor:pointer" id="log-toggle-${run.run_id}">Hide</button>
      </h3>
      <div id="log-body-${run.run_id}" style="display:block">
        <pre id="log-pre-${run.run_id}" style="background:var(--bg1);border:1px solid var(--border);border-radius:4px;padding:10px;font-size:11px;line-height:1.5;max-height:400px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;color:var(--text)">Loading...</pre>
      </div>
    </div>`;

  // Async load FVE metrics if the card was rendered
  if (art.fve_json && art.fve_json.exists) {
    fetch(`/api/runs/${run.run_id}/fve`)
      .then(r => r.json())
      .then(d => {
        const card = document.getElementById(`fve-card-${run.run_id}`);
        if (!card) return;
        if (!d.available) {
          card.querySelector('div').textContent = 'Metrics not available.';
          return;
        }
        const pct = v => (v * 100).toFixed(1) + '%';
        card.innerHTML = `<h3>FVE + Dead Neurons</h3>
          <div class="metrics-grid" style="margin-top:10px">
            <div class="metric"><div class="m-label">FVE Model A</div><div class="m-value" style="color:var(--blue)">${pct(d.fve_a)}</div></div>
            <div class="metric"><div class="m-label">FVE Model B</div><div class="m-value" style="color:var(--purple)">${pct(d.fve_b)}</div></div>
            <div class="metric"><div class="m-label">FVE Mean</div><div class="m-value">${pct(d.fve_mean)}</div></div>
            <div class="metric"><div class="m-label">Dead Neurons</div><div class="m-value" style="color:var(--red)">${pct(d.dead_frac)}</div><div style="font-size:11px;color:var(--muted);margin-top:2px">${d.n_dead.toLocaleString()} / ${d.n_features?.toLocaleString?.() ?? '?'}</div></div>
          </div>`;
      }).catch(() => {});
  }
}

let _copyToastTimer = null;
function copyArtPath(btn, path) {
  navigator.clipboard.writeText(path).then(() => {
    const toast = document.getElementById('copy-toast');
    toast.classList.add('show');
    clearTimeout(_copyToastTimer);
    _copyToastTimer = setTimeout(() => toast.classList.remove('show'), 2000);
  }).catch(() => { alert('Could not copy: ' + path); });
}

async function loadRunInViewer(runId) {
  const btn = event.target;
  btn.disabled = true; btn.textContent = 'Loading…';
  try {
    const res = await fetch(`/api/runs/${runId}/load_csv`, {method:'POST'});
    if (!res.ok) { const t = await res.text(); alert('Error ' + res.status + ': ' + t); btn.disabled=false; btn.textContent='Load in Feature Browser'; return; }
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); btn.disabled=false; btn.textContent='Load in Feature Browser'; return; }
    window.location.href = '/';
  } catch(e) {
    alert('Network error: ' + e.message); btn.disabled=false; btn.textContent='Load in Feature Browser';
  }
}

// New Run Modal
function openNewRunModal() { document.getElementById('new-run-modal').classList.add('open'); }
function closeNewRunModal() { document.getElementById('new-run-modal').classList.remove('open'); document.getElementById('form-error').style.display='none'; }
function toggleAdvanced() {
  const el = document.getElementById('advanced-fields');
  el.classList.toggle('open');
  document.querySelector('.advanced-toggle').textContent = el.classList.contains('open') ? '▾ Advanced options' : '▸ Advanced options';
}

// Pipeline Log
const _logOpen = {};
function toggleLog(runId) {
  const body = document.getElementById('log-body-' + runId);
  const btn  = document.getElementById('log-toggle-' + runId);
  if (_logOpen[runId]) {
    _logOpen[runId] = false;
    body.style.display = 'none';
    btn.textContent = 'Show';
  } else {
    _logOpen[runId] = true;
    body.style.display = 'block';
    btn.textContent = 'Hide';
    fetchLog(runId);
  }
}

async function fetchLog(runId) {
  if (!_logOpen[runId]) return;
  try {
    const res = await fetch(`/api/runs/${runId}/log?lines=500`);
    const data = await res.json();
    const pre  = document.getElementById('log-pre-' + runId);
    const meta = document.getElementById('log-meta-' + runId);
    if (!pre) return;
    if (!data.exists) {
      pre.textContent = '(no pipeline.log yet — run has not started)';
      return;
    }
    pre.textContent = data.lines.join('\n');
    pre.scrollTop = pre.scrollHeight;
    meta.textContent = `${data.total_lines} lines total`;
  } catch(e) {}
  // Poll every 5s if log panel is open
  setTimeout(() => fetchLog(runId), 5000);
}

// Auto-detect d_model when Model A field loses focus
let modelATimer = null, modelBTimer = null;
let numLayersA = null, numLayersB = null;

document.addEventListener('DOMContentLoaded', () => {
  const inpA = document.getElementById('f-model-a');
  const inpB = document.getElementById('f-model-b');
  if (inpA) {
    inpA.addEventListener('blur', () => { if (inpA.value.trim()) fetchModelInfo(inpA.value.trim()); });
    inpA.addEventListener('input', () => {
      clearTimeout(modelATimer);
      modelATimer = setTimeout(() => { if (inpA.value.trim().length > 4) fetchModelInfo(inpA.value.trim()); }, 1200);
    });
  }
  if (inpB) {
    inpB.addEventListener('blur', () => { if (inpB.value.trim()) fetchModelBInfo(inpB.value.trim()); });
    inpB.addEventListener('input', () => {
      clearTimeout(modelBTimer);
      modelBTimer = setTimeout(() => { if (inpB.value.trim().length > 4) fetchModelBInfo(inpB.value.trim()); }, 1200);
    });
  }
});

async function fetchModelInfo(modelId) {
  const box = document.getElementById('model-info-box');
  box.style.display = 'none';
  try {
    const res = await fetch('/api/suggest_hyperparams?model_a=' + encodeURIComponent(modelId));
    const data = await res.json();
    if (data.d_model) {
      numLayersA = data.num_layers;
      document.getElementById('info-dmodel').textContent = 'd_model=' + data.d_model;
      document.getElementById('info-numlayers').textContent = data.num_layers ? data.num_layers + ' layers (0–' + data.last_layer + ')' : '?';
      document.getElementById('info-nfeatures').textContent = data.n_features.toLocaleString();
      document.getElementById('info-auxk').textContent = data.aux_k;
      box.style.display = 'block';
      // Store detected d_model for updateNFeatures preview
      window._detectedDModel = data.d_model;
      updateNFeatures();
      // Update layer-a hint
      if (data.num_layers) {
        document.getElementById('hint-layer-a').textContent =
          '0-indexed. Valid range: 0–' + data.last_layer + ' (last)';
        // Default to last layer
        const la = document.getElementById('f-layer-a');
        if (la.value == '25') la.value = data.last_layer;
      }
      validateLayers();
    }
  } catch {}
}

async function fetchModelBInfo(modelId) {
  const box = document.getElementById('model-b-info-box');
  box.style.display = 'none';
  try {
    const res = await fetch('/api/model_layer_count?model=' + encodeURIComponent(modelId));
    const data = await res.json();
    numLayersB = data.num_layers;
    if (data.num_layers) {
      box.style.display = 'block';
      document.getElementById('info-b-numlayers').textContent =
        data.num_layers + ' layers (0–' + data.last_layer + ')';
      document.getElementById('hint-layer-b').textContent =
        '0-indexed. Valid range: 0–' + data.last_layer + ' (last)';
      validateLayers();
    }
  } catch {}
}

function updateNFeatures() {
  const factor = parseInt(document.getElementById('f-expansion-factor').value) || 16;
  const dModel = window._detectedDModel;
  const preview = document.getElementById('nfeatures-preview');
  if (dModel) {
    const nf = factor * dModel;
    document.getElementById('nf-val').textContent = nf.toLocaleString();
    preview.style.display = 'block';
  } else {
    preview.style.display = 'none';
  }
}

function validateLayers() {
  const errEl = document.getElementById('layer-error');
  const laVal = document.getElementById('f-layer-a').value;
  const lbVal = document.getElementById('f-layer-b').value;
  const la = laVal !== '' ? parseInt(laVal) : null;
  const lb = lbVal !== '' ? parseInt(lbVal) : null;
  const errors = [];

  if (la !== null && numLayersA !== null && la >= numLayersA)
    errors.push(`Layer A (${la}) exceeds Model A's last layer (${numLayersA - 1})`);
  if (la !== null && la < 0)
    errors.push('Layer A must be ≥ 0');
  if (lb !== null && numLayersB !== null && lb >= numLayersB)
    errors.push(`Layer B (${lb}) exceeds Model B's last layer (${numLayersB - 1})`);
  if (lb !== null && lb < 0)
    errors.push('Layer B must be ≥ 0');

  if (errors.length) {
    errEl.textContent = errors.join(' · ');
    errEl.style.display = 'block';
  } else {
    errEl.style.display = 'none';
  }
  return errors.length === 0;
}

async function submitNewRun() {
  const errEl = document.getElementById('form-error');
  errEl.style.display = 'none';

  if (!validateLayers()) {
    errEl.textContent = 'Please fix the layer index errors above.';
    errEl.style.display = 'block'; return;
  }

  const existing = document.getElementById('f-existing-dataset').value.trim();
  const lbRaw = document.getElementById('f-layer-b').value;
  const body = {
    name:          document.getElementById('f-name').value.trim(),
    model_a:       document.getElementById('f-model-a').value.trim(),
    model_b:       document.getElementById('f-model-b').value.trim(),
    dataset_a:     document.getElementById('f-dataset-a').value.trim(),
    dataset_b:     document.getElementById('f-dataset-b').value.trim(),
    layer_a:          parseInt(document.getElementById('f-layer-a').value),
    layer_b:          lbRaw !== '' ? parseInt(lbRaw) : null,
    expansion_factor: parseInt(document.getElementById('f-expansion-factor').value) || 32,
    topk:             parseInt(document.getElementById('f-topk').value),
    judge_top_n:      document.getElementById('f-judge-top-n').value ? parseInt(document.getElementById('f-judge-top-n').value) : 999999,
    total_tokens:     parseInt(document.getElementById('f-total-tokens').value),
    warmup:           parseInt(document.getElementById('f-warmup').value) || 1000,
    norm_init_scale:  parseFloat(document.getElementById('f-norm-init-scale').value) || 1.0,
    skip_dataset_gen: !!existing,
    existing_dataset: existing || null,
  };
  if (!body.model_a || !body.model_b || !body.dataset_a || !body.dataset_b) {
    errEl.textContent = 'Please fill in all required fields.'; errEl.style.display='block'; return;
  }
  const btn = document.querySelector('.btn-submit');
  btn.disabled = true; btn.textContent = 'Submitting…';
  try {
    const res = await fetch('/api/runs/create', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    const data = await res.json();
    if (data.error) { errEl.textContent = data.error; errEl.style.display='block'; btn.disabled=false; btn.textContent='Submit Pipeline Run'; return; }
    closeNewRunModal();
    await loadRuns();
    selectRun(data.run_id);
  } catch(e) {
    errEl.textContent = 'Network error: ' + e; errEl.style.display='block';
  }
  btn.disabled=false; btn.textContent='Submit Pipeline Run';
}

// Close modal on backdrop click
document.getElementById('new-run-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('new-run-modal')) closeNewRunModal();
});

loadRuns();
// Auto-refresh run list every 30s
setInterval(loadRuns, 30000);

async function showStepLog(runId, step) {
  const modal = document.getElementById('step-log-modal');
  const pre = document.getElementById('step-log-pre');
  const title = document.getElementById('step-log-title');
  const fileEl = document.getElementById('step-log-file');
  title.textContent = `Error log: ${step}`;
  fileEl.textContent = '';
  pre.textContent = 'Loading...';
  modal.classList.add('open');
  try {
    const res = await fetch(`/api/runs/${runId}/step_log?step=${encodeURIComponent(step)}&lines=2000`);
    const d = await res.json();
    if (!d.exists) {
      pre.textContent = '(No log file found — job may not have started or log file was deleted)';
    } else {
      fileEl.textContent = d.file || '';
      pre.textContent = d.lines.join('\n') || '(empty log)';
      pre.scrollTop = pre.scrollHeight;
    }
  } catch(e) {
    pre.textContent = 'Failed to load log: ' + e.message;
  }
}
function closeStepLog() {
  document.getElementById('step-log-modal').classList.remove('open');
}
document.getElementById('step-log-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('step-log-modal')) closeStepLog();
});
