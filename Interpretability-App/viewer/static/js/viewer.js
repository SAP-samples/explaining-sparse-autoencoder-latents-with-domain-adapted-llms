const state = { lang:'', norm_flag:'', q:'', sort:'delta_norm', order:'desc', page:0, total:0, selectedFidx:null, n_active_min:0 };
let searchTimer = null;

function setSort() {
  state.sort = document.getElementById('sort-select').value;
  state.page = 0; reloadList();
}
function toggleOrder() {
  state.order = state.order === 'desc' ? 'asc' : 'desc';
  document.getElementById('order-btn').textContent = state.order === 'desc' ? '↓' : '↑';
  state.page = 0; reloadList();
}
function setNActive(val) {
  const v = Math.max(0, parseInt(val) || 0);
  state.n_active_min = v; state.page = 0;
  document.getElementById('n-active-slider').value = v;
  document.getElementById('n-active-input').value = v;
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => reloadList(), 250);
}
function scoreColor(s) {
  s = parseFloat(s);
  if (s >= 9) return '#4ade80'; if (s >= 6) return '#fbbf24';
  if (s >= 3) return '#f97316'; return '#ef4444';
}
function setFilter(key, val) {
  state[key] = val; state.page = 0;
  const groupMap = { norm_flag: 'norm' };
  document.querySelectorAll(`[data-group="${groupMap[key] || key}"]`).forEach(c => {
    c.classList.toggle('active', c.dataset.val === val);
  });
  // Auto-set sort direction for norm_flag so strongest features always come first:
  // A (delta_norm ≈ 0) → asc,  B (delta_norm ≈ 1) → desc,  clear → desc
  if (key === 'norm_flag' && state.sort === 'delta_norm') {
    state.order = (val === 'A') ? 'asc' : 'desc';
    const btn = document.getElementById('order-btn');
    if (btn) btn.textContent = state.order === 'asc' ? '↑' : '↓';
  }
  reloadList();
}
function debounceSearch() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.q = document.getElementById('search').value; state.page = 0; reloadList(); }, 250);
}
async function reloadList() {
  state.page = 0;
  document.getElementById('feature-list').innerHTML = '';
  await fetchPage();
}
async function loadMore() { state.page++; await fetchPage(); }
let _fetchingPage = false;
async function fetchPage() {
  if (_fetchingPage) return;
  _fetchingPage = true;
  const p = new URLSearchParams({ lang: state.lang, norm_flag: state.norm_flag, q: state.q, page: state.page, limit: 50, sort: state.sort, order: state.order, n_active_min: state.n_active_min });
  if (state.page === 0) {
    document.getElementById('feature-list').innerHTML = '<div style="padding:20px;color:var(--muted);font-size:12px">Loading…</div>';
  }
  try {
    const res = await fetch('/api/index?' + p);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    state.total = data.total;
    renderList(data.rows, state.page > 0);
    const hasMore = (state.page + 1) * 50 < data.total;
    const lm = document.getElementById('load-more');
    if (lm) lm.style.display = hasMore ? '' : 'none';
  } catch(e) {
    document.getElementById('feature-list').innerHTML = '<div style="padding:20px;color:#ef4444;font-size:12px">Failed: ' + e.message + '</div>';
  } finally {
    _fetchingPage = false;
  }
}
function renderList(rows, append) {
  const list = document.getElementById('feature-list');
  if (!append) list.innerHTML = '';
  rows.forEach(r => {
    const dn = parseFloat(r.delta_norm);
    const dnCol = dn < 0.5 ? 'var(--base)' : 'var(--swallow)';
    const div = document.createElement('div');
    div.className = 'feat-item' + (r.feature_idx == state.selectedFidx ? ' selected' : '');
    div.dataset.fidx = r.feature_idx;
    div.setAttribute('role', 'button');
    div.setAttribute('tabindex', '0');
    div.onclick = () => selectFeature(parseInt(r.feature_idx));
    div.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectFeature(parseInt(r.feature_idx)); } };
    div.innerHTML = `
      <div class="fi-top">
        <span class="fi-label" title="${r.judge_label}">${r.judge_label || '(no label)'}</span>
        <span class="fi-idx">#${r.feature_idx}</span>
      </div>
      <div class="fi-bottom">
        <span class="badge ${r.norm_flag}">Δnorm: ${_flagLabel(r.norm_flag)}</span>
        <span style="font-size:11px;color:${dnCol};font-weight:600">${isNaN(dn)?'?':dn.toFixed(3)}</span>
        <span style="font-size:11px;color:var(--muted);margin-left:auto">${parseInt(r.n_active).toLocaleString()} act</span>
      </div>`;
    list.appendChild(div);
  });
}
async function selectFeature(fidx) {
  state.selectedFidx = fidx;
  document.querySelectorAll('.feat-item').forEach(el => el.classList.toggle('selected', parseInt(el.dataset.fidx) === fidx));
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('detail').style.display = '';
  document.getElementById('detail').innerHTML = '<div style="color:var(--muted);padding:40px">Loading…</div>';
  try {
    const res = await fetch(`/api/feature/${fidx}`);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    renderDetail(await res.json());
  } catch(e) {
    document.getElementById('detail').innerHTML = '<div style="padding:40px;color:#ef4444">Failed to load feature: ' + e.message + '</div>';
  }
}
// Maps domain_label CSV values to short display strings
function _domainLabel(val) {
  const v = (val || '').toUpperCase();
  if (v === 'ENG') return 'EN';
  if (v === 'JPN') return 'JA';
  if (v === 'BOTH') return 'BOTH';
  if (v === 'A') return 'EN';
  if (v === 'B') return 'JA';
  return val || '?';
}
// Maps raw CSV flag values (old: JPN/BASE/SWALLOW, new: A/B) to user-visible model labels.
function _flagLabel(val) {
  if (!val) return val;
  const v = val.toUpperCase();
  if (v === 'A' || v === 'ENG' || v === 'BASE')    return _labelA;
  if (v === 'B' || v === 'JPN' || v === 'SWALLOW') return _labelB;
  return val; // BOTH or anything else: show as-is
}

function renderDetail(f) {
  const dn = parseFloat(f.delta_norm).toFixed(4);
  const dnCol = parseFloat(f.delta_norm) < 0.5 ? 'var(--base)' : 'var(--swallow)';
  const qKeys = ['0_10','10_20','20_30','30_40','40_50','50_60','60_70','70_80','80_90','90_100'];
  const qLabels = ['0–10%','10–20%','20–30%','30–40%','40–50%','50–60%','60–70%','70–80%','80–90%','90–100%'];
  let qHtml = '';
  qKeys.forEach((k, i) => {
    const raw = f[`quantile_${k}`] || '';
    const toksHtml = raw.split(' | ').filter(t=>t.length>0).map(t => {
      const clean = t.replace(/^'|'$/g, '');
      return `<span class="tok ${/[　-鿿豈-﫿]/.test(clean)?'jpn':'eng'}" title="${clean}">${clean}</span>`;
    }).join('');
    qHtml += `<div class="q-cell"><div class="q-label">${qLabels[i]}</div><div class="q-tokens">${toksHtml||'<span style="color:var(--muted);font-size:11px">—</span>'}</div></div>`;
  });
  const isBookmarked = _bookmarks.has(f.feature_idx);
  document.getElementById('detail').innerHTML = `
    <div class="feat-header">
      <div class="feat-title">
        <div class="feat-badges">
          <span class="badge big-badge ${f.norm_flag}">Δ Norm: ${_flagLabel(f.norm_flag)}</span>
          <button class="bm-btn ${isBookmarked?'bm-on':''}" onclick="toggleBookmark(${f.feature_idx})" title="${isBookmarked?'Remove bookmark':'Bookmark this feature'}">${isBookmarked?'★':'☆'}</button>
        </div>
        <h1>${f.judge_label||'(no label)'}</h1>
        <div class="feat-idx">Feature #${f.feature_idx}</div>
      </div>
    </div>
    <div class="card"><h3>Judge Explanation</h3><p class="explanation">${f.judge_explanation||'—'}</p></div>
    <div class="card"><h3>Decoder Norms &amp; Classification</h3><div class="metrics-grid">
      <div class="metric"><div class="m-label">norm_${_labelA}</div><div class="m-value" style="color:var(--base)">${parseFloat(f.norm_a).toFixed(4)}</div></div>
      <div class="metric"><div class="m-label">norm_${_labelB}</div><div class="m-value" style="color:var(--swallow)">${parseFloat(f.norm_b).toFixed(4)}</div></div>
      <div class="metric"><div class="m-label">Δnorm</div><div class="m-value" style="color:${dnCol}">${dn}</div></div>
      <div class="metric"><div class="m-label">n_active</div><div class="m-value">${parseInt(f.n_active).toLocaleString()}</div></div>
      <div class="metric"><div class="m-label">mean_act</div><div class="m-value">${parseFloat(f.mean_activation).toFixed(3)}</div></div>
      <div class="metric"><div class="m-label">max_act</div><div class="m-value">${parseFloat(f.max_activation).toFixed(3)}</div></div>
      ${f.judge_score != null ? `
      <div class="metric"><div class="m-label">judge_score</div><div class="m-value">${parseInt(f.judge_score)}/12</div></div>
      <div class="metric"><div class="m-label">lang_purity</div><div class="m-value">${parseInt(f.judge_domain_purity)}/4</div></div>
      <div class="metric"><div class="m-label">content_richness</div><div class="m-value">${parseInt(f.judge_content_richness)}/4</div></div>
      <div class="metric"><div class="m-label">specificity</div><div class="m-value">${parseInt(f.judge_specificity)}/4</div></div>
      ` : ''}
    </div></div>
    <div class="card"><h3>Activating Tokens by Quantile</h3><div class="quantile-grid">${qHtml}</div></div>`;
}

let _labelA = 'Model A', _labelB = 'Model B';

// ── Bookmarks (stored server-side in settings.json, localStorage as cache) ───
const _BM_KEY = 'cc_viewer_bookmarks';
let _bookmarks = new Map(); // fidx(int) → label
let _bmSaveTimer = null;
function _loadBookmarks() {
  try { const s = localStorage.getItem(_BM_KEY); if (s) _bookmarks = new Map(JSON.parse(s).map(([k,v])=>[parseInt(k),v])); } catch {}
}
function _loadBookmarksFromServer(d) {
  if (d && d.bookmarks && typeof d.bookmarks === 'object') {
    const entries = Object.entries(d.bookmarks).map(([k,v]) => [parseInt(k), v]);
    if (entries.length > 0 || _bookmarks.size === 0) {
      _bookmarks = new Map(entries);
      try { localStorage.setItem(_BM_KEY, JSON.stringify([..._bookmarks])); } catch {}
    }
  }
}
function _saveBookmarks() {
  try { localStorage.setItem(_BM_KEY, JSON.stringify([..._bookmarks])); } catch {}
  clearTimeout(_bmSaveTimer);
  _bmSaveTimer = setTimeout(() => {
    const obj = Object.fromEntries([..._bookmarks.entries()].map(([k,v])=>[k,v]));
    fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({bookmarks: obj})}).catch(()=>{});
  }, 300);
}
function toggleBookmark(fidx) {
  if (_bookmarks.has(fidx)) {
    _bookmarks.delete(fidx);
  } else {
    const label = document.querySelector(`[data-fidx="${fidx}"] .fi-label`)?.textContent || '#'+fidx;
    _bookmarks.set(fidx, label);
  }
  _saveBookmarks();
  // Update button in-place — no re-fetch needed
  const btn = document.querySelector(`.bm-btn[onclick*="toggleBookmark(${fidx})"]`);
  if (btn) {
    const on = _bookmarks.has(fidx);
    btn.textContent = on ? '★' : '☆';
    btn.classList.toggle('bm-on', on);
    btn.title = on ? 'Remove bookmark' : 'Bookmark this feature';
  }
}
function renderBookmarkList() {
  const el = document.getElementById('bm-list');
  if (!el) return;
  if (_bookmarks.size === 0) {
    el.innerHTML = '<div style="padding:6px 10px;color:var(--muted)">No bookmarks yet.</div>';
    return;
  }
  el.innerHTML = [..._bookmarks.entries()].map(([fidx, label]) =>
    `<div class="bm-item" onclick="closeSettings();selectFeature(${fidx})">
      <span class="bm-item-idx">#${fidx}</span>
      <span class="bm-item-label">${label}</span>
      <button class="bm-item-del" onclick="event.stopPropagation();removeBm(${fidx})" title="Remove">✕</button>
    </div>`
  ).join('');
}
function removeBm(fidx) { _bookmarks.delete(fidx); _saveBookmarks(); renderBookmarkList(); }
_loadBookmarks();

function _escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function showTokenContext(fidx, tokenText) {
  const modal = document.getElementById('ctx-modal');
  const list  = document.getElementById('ctx-list');
  const title = document.getElementById('ctx-title');
  title.textContent = `Contexts for token "${tokenText}" — Feature #${fidx}`;
  list.innerHTML = '<div style="color:var(--muted);font-size:12px">Loading…</div>';
  modal.style.display = 'flex';

  try {
    const res = await fetch(`/api/feature/${fidx}/contexts?n=20`);
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    if (data.error) { list.innerHTML = `<div style="color:#ef4444">${data.error}</div>`; return; }

    const positions = data.positions;
    if (!positions || positions.length === 0) {
      list.innerHTML = '<div style="color:var(--muted);font-size:12px">No activating contexts found.</div>';
      return;
    }

    // Filter to only contexts where the clicked token actually fired
    const filtered = positions.filter(p => p.token === tokenText);
    const display = filtered.length > 0 ? filtered : positions; // fallback to all if no exact match

    if (filtered.length === 0) {
      list.innerHTML = '<div style="color:var(--muted);font-size:12px">No exact match for this token — showing top contexts for the feature.</div>';
    }

    list.innerHTML = (filtered.length > 0 ? filtered : positions).map((p, i) => {
      const ctx = p.context || '';
      const tok = p.token || tokenText;
      const act = parseFloat(p.activation).toFixed(2);
      const domain = p.domain === 1 ? `<span style="color:var(--swallow);font-size:11px">${_labelB}</span>` : `<span style="color:var(--base);font-size:11px">${_labelA}</span>`;

      // Highlight the activating token using exact char offset when available,
      // otherwise fall back to first-occurrence regex (no global flag)
      let highlighted;
      if (p.char_start_in_ctx >= 0 && p.token_len > 0) {
        const before = _escHtml(ctx.slice(0, p.char_start_in_ctx));
        const target = _escHtml(ctx.slice(p.char_start_in_ctx, p.char_start_in_ctx + p.token_len));
        const after  = _escHtml(ctx.slice(p.char_start_in_ctx + p.token_len));
        highlighted = before + `<mark style="background:#fbbf2440;color:#fbbf24;border-radius:2px;padding:0 2px">${target}</mark>` + after;
      } else {
        const escaped = tok.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        highlighted = _escHtml(ctx).replace(new RegExp(escaped),
          `<mark style="background:#fbbf2440;color:#fbbf24;border-radius:2px;padding:0 2px">${_escHtml(tok)}</mark>`);
      }

      const total = filtered.length > 0 ? filtered.length : positions.length;
      return `<div style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:10px 12px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
          <span style="font-size:11px;color:var(--muted)">Context ${i+1} / ${total} &nbsp;·&nbsp; chunk ${p.chunk_idx}, pos ${p.tok_pos} &nbsp;·&nbsp; ${domain}</span>
          <span style="font-size:12px;font-weight:600;color:#fbbf24">act: ${act}</span>
        </div>
        <div style="font-size:13px;line-height:1.6;font-family:monospace;white-space:pre-wrap;word-break:break-all">${highlighted}</div>
      </div>`;
    }).join('');

  } catch(e) {
    list.innerHTML = `<div style="color:#ef4444">Failed: ${e.message}</div>`;
  }
}

function applyLabels(la, lb) {
  _labelA = la; _labelB = lb;
  ['lang','norm','freq'].forEach(g => {
    const a = document.getElementById('chip-'+g+'-a');
    const b = document.getElementById('chip-'+g+'-b');
    if (a) a.textContent = la;
    if (b) b.textContent = lb;
  });
}

async function populateRunSelector(activePath) {
  const sel = document.getElementById('run-selector');
  if (!sel) return;
  try {
    const res = await fetch('/api/available_csvs');
    const opts = await res.json();
    if (opts.length === 0) { sel.style.display = 'none'; return; }
    sel.style.display = '';
    sel.innerHTML = opts.map(o =>
      `<option value="${o.path}" ${o.active?'selected':''}>${o.label}</option>`
    ).join('');
    sel.disabled = opts.length <= 1;
    sel.title = opts.length <= 1 ? 'No other runs available yet' : 'Switch active dataset';
  } catch(e) {}
}
async function switchRun(csvPath) {
  if (!csvPath) return;
  // Find run_id for this csv path (if any)
  try {
    const res = await fetch('/api/available_csvs');
    const opts = await res.json();
    const opt = opts.find(o => o.path === csvPath);
    if (!opt) return;
    if (opt.run_id) {
      // Load via run's load_csv endpoint
      const r = await fetch(`/api/runs/${opt.run_id}/load_csv`, {method:'POST'});
      if (!r.ok) { alert('Failed to switch run'); return; }
    } else {
      // Default CSV — reload with default
      const r = await fetch('/api/runs/load_default_csv', {method:'POST'});
      if (!r.ok) { alert('Failed to switch to default CSV'); return; }
    }
    // Reload list
    state.page = 0; reloadList(); loadStats();
  } catch(e) { alert('Switch failed: ' + e.message); }
}

async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const langCounts = data.lang || {};
    const langHtml = Object.entries(langCounts).filter(([k]) => k).map(([k,v]) => `<span>${k} <b>${v}</b></span>`).join('');
    document.getElementById('stats-bar').innerHTML =
      `<span><b>${data.total.toLocaleString()}</b> features</span>` +
      langHtml +
      `<span>avg LLMaaJ score <b>${data.score_mean}</b></span>`;
    if (data.csv_path) {
      document.getElementById('download-csv-btn').style.display = '';
      populateRunSelector(data.csv_path);
    }
    if (data.label_a || data.label_b) {
      applyLabels(data.label_a || 'Model A', data.label_b || 'Model B');
    }
  } catch(e) {
    document.getElementById('stats-bar').innerHTML = '<span style="color:#ef4444">Failed to load stats: ' + e.message + '</span>';
  }
}

// Resizable sidebar
(function() {
  const handle = document.getElementById('resize-handle');
  const left   = document.getElementById('left');
  let dragging = false, startX = 0, startW = 0;
  handle.addEventListener('mousedown', e => {
    dragging = true; startX = e.clientX; startW = left.offsetWidth;
    handle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    const w = Math.max(180, Math.min(600, startW + e.clientX - startX));
    left.style.width = w + 'px';
  });
  window.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
})();

loadStats().then(() => {
  const bm = new URLSearchParams(window.location.search).get('bm');
  if (bm !== null) {
    const fidx = parseInt(bm);
    if (!isNaN(fidx)) reloadList().then(() => selectFeature(fidx));
  }
});
fetch('/api/settings').then(r=>r.json()).then(d=>{ applyLabels(d.label_a||'Model A',d.label_b||'Model B'); _loadBookmarksFromServer(d); }).catch(()=>{});
reloadList();

// Infinite scroll: observe #load-more (sibling of feature-list, inside scrollable #left)
const _scrollSentinel = document.getElementById('load-more');
const _scrollObserver = new IntersectionObserver(entries => {
  if (entries[0].isIntersecting && _scrollSentinel.style.display !== 'none') {
    state.page++;
    fetchPage();
  }
}, { root: document.getElementById('left'), rootMargin: '0px 0px 80px 0px', threshold: 0 });
_scrollObserver.observe(_scrollSentinel);
