const _BM_KEY = 'cc_viewer_bookmarks';
let _bookmarks = new Map();
let _bmSaveTimer2 = null;
function _loadBookmarks() {
  try { const s = localStorage.getItem(_BM_KEY); if (s) _bookmarks = new Map(JSON.parse(s).map(([k,v])=>[parseInt(k),v])); } catch {}
}
function _saveBookmarks() {
  try { localStorage.setItem(_BM_KEY, JSON.stringify([..._bookmarks])); } catch {}
  clearTimeout(_bmSaveTimer2);
  _bmSaveTimer2 = setTimeout(() => {
    const obj = Object.fromEntries([..._bookmarks.entries()].map(([k,v])=>[k,v]));
    fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({bookmarks: obj})}).catch(()=>{});
  }, 300);
}
function removeBm(fidx) { _bookmarks.delete(fidx); _saveBookmarks(); renderBookmarkList(); }
function renderBookmarkList() {
  const el = document.getElementById('bm-list');
  if (_bookmarks.size === 0) {
    el.innerHTML = '<div style="padding:8px 12px;color:var(--muted)">No bookmarks yet.</div>';
    return;
  }
  el.innerHTML = [..._bookmarks.entries()].map(([fidx, label]) =>
    `<div class="bm-item" onclick="window.location='/?bm=${fidx}'">
      <span class="bm-item-idx">#${fidx}</span>
      <span class="bm-item-label">${label}</span>
      <button class="bm-item-del" onclick="event.stopPropagation();removeBm(${fidx})" title="Remove">✕</button>
    </div>`
  ).join('');
}

let _labelA = 'Model A', _labelB = 'Model B';
async function loadSettings() {
  try {
    const res = await fetch('/api/settings');
    const d = await res.json();
    _labelA = d.label_a || 'Model A';
    _labelB = d.label_b || 'Model B';
    document.getElementById('s-label-a').value = _labelA;
    document.getElementById('s-label-b').value = _labelB;
    document.getElementById('lo-label').textContent = _labelA + ' <';
    document.getElementById('hi-label').textContent = _labelB + ' >';
    if (d.norm_thresh_lo != null) { threshLo = parseFloat(d.norm_thresh_lo); document.getElementById('lo-input').value = threshLo.toFixed(2); }
    if (d.norm_thresh_hi != null) { threshHi = parseFloat(d.norm_thresh_hi); document.getElementById('hi-input').value = threshHi.toFixed(2); }
    if (d.bookmarks && typeof d.bookmarks === 'object') {
      const entries = Object.entries(d.bookmarks).map(([k,v]) => [parseInt(k), v]);
      _bookmarks = new Map(entries);
      try { localStorage.setItem(_BM_KEY, JSON.stringify([..._bookmarks])); } catch {}
      renderBookmarkList();
    }
  } catch(e) {}
}
async function saveLabels() {
  const la = document.getElementById('s-label-a').value.trim() || 'Model A';
  const lb = document.getElementById('s-label-b').value.trim() || 'Model B';
  try {
    const res = await fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({label_a:la, label_b:lb})});
    if (!res.ok) throw new Error('HTTP ' + res.status);
    _labelA = la; _labelB = lb;
    document.getElementById('lo-label').textContent = la + ' <';
    document.getElementById('hi-label').textContent = lb + ' >';
    if (dnormData) renderThreshCounts(dnormData);
    showToast('Settings saved!');
  } catch(e) { alert('Save failed: ' + e.message); }
}
function showToast(msg) {
  const t = document.getElementById('save-toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2200);
}

// Threshold histogram (same logic as viewer)
let dnormData = null, threshLo = 0.1, threshHi = 0.9, dragging = null, threshTimer = null;
function onThreshInput() {
  threshLo = parseFloat(document.getElementById('lo-input').value)||0.1;
  threshHi = parseFloat(document.getElementById('hi-input').value)||0.9;
  if (threshLo >= threshHi) return;
  clearTimeout(threshTimer); threshTimer = setTimeout(() => { fetchHistogram(); saveThresholds(); }, 300);
}
async function saveThresholds() {
  try {
    await fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({norm_thresh_lo: threshLo, norm_thresh_hi: threshHi})});
  } catch(e) {}
}
async function fetchHistogram() {
  try {
    const res = await fetch(`/api/delta_norm_hist?lo=${threshLo}&hi=${threshHi}`);
    if (!res.ok) return;
    dnormData = await res.json();
    threshLo = dnormData.lo; threshHi = dnormData.hi;
    drawHistogram(); renderThreshCounts(dnormData);
  } catch(e) {}
}
function drawHistogram() {
  const canvas = document.getElementById('dnorm-canvas');
  if (!canvas || !dnormData) return;
  const ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
  const pad = {l:4,r:4,t:6,b:18}, iW=W-pad.l-pad.r, iH=H-pad.t-pad.b;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#23273a'; ctx.fillRect(0,0,W,H);
  const bins=dnormData.bins, maxN=Math.max(...bins.map(b=>b.n),1);
  bins.forEach(b => {
    const x=pad.l+b.lo*iW, w=(b.hi-b.lo)*iW-1, mid=(b.lo+b.hi)/2;
    const bh=iH*b.n/maxN, y=pad.t+iH-bh;
    ctx.fillStyle=mid<threshLo?'#60a5fa':mid>threshHi?'#f472b6':'#a78bfa';
    ctx.beginPath(); ctx.roundRect(x,y,Math.max(w,1),bh,2); ctx.fill();
  });
  ctx.strokeStyle='#fbbf24'; ctx.lineWidth=1.5; ctx.setLineDash([3,2]);
  [pad.l+threshLo*iW, pad.l+threshHi*iW].forEach(x => {
    ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+iH); ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.fillStyle='#8892b0'; ctx.font='9px monospace'; ctx.textAlign='center';
  [0,0.25,0.5,0.75,1.0].forEach(v => ctx.fillText(v.toFixed(2), pad.l+v*iW, H-2));
  ctx.fillStyle='#fbbf24'; ctx.font='bold 9px monospace';
  ctx.textAlign='left'; ctx.fillText(threshLo.toFixed(2), pad.l+threshLo*iW+2, pad.t+8);
  ctx.textAlign='right'; ctx.fillText(threshHi.toFixed(2), pad.l+threshHi*iW-2, pad.t+8);
}
function renderThreshCounts(d) {
  document.getElementById('thresh-counts').innerHTML =
    `<span class="tc-base">${_labelA}: ${d.n_base.toLocaleString()}</span>` +
    `<span class="tc-both">BOTH: ${d.n_both.toLocaleString()}</span>` +
    `<span class="tc-swallow">${_labelB}: ${d.n_swallow.toLocaleString()}</span>`;
}
function setupCanvasDrag() {
  const canvas = document.getElementById('dnorm-canvas');
  if (!canvas) return;
  function xToVal(ev) {
    const rect=canvas.getBoundingClientRect(), px=(ev.clientX-rect.left)/rect.width;
    return Math.max(0,Math.min(1,(px-4/canvas.width)/(1-8/canvas.width)));
  }
  canvas.addEventListener('mousedown', ev => {
    if (!dnormData) return;
    const v=xToVal(ev); dragging=Math.abs(v-threshLo)<Math.abs(v-threshHi)?'lo':'hi';
  });
  window.addEventListener('mousemove', ev => {
    if (!dragging||!dnormData) return;
    const v=Math.round(xToVal(ev)*100)/100;
    if (dragging==='lo'&&v<threshHi-0.05) { threshLo=v; document.getElementById('lo-input').value=v.toFixed(2); }
    else if (dragging==='hi'&&v>threshLo+0.05) { threshHi=v; document.getElementById('hi-input').value=v.toFixed(2); }
    let nb=0,ns=0,nboth=0;
    dnormData.bins.forEach(b=>{const mid=(b.lo+b.hi)/2; if(mid<threshLo)nb+=b.n; else if(mid>threshHi)ns+=b.n; else nboth+=b.n;});
    Object.assign(dnormData,{n_base:nb,n_swallow:ns,n_both:nboth,lo:threshLo,hi:threshHi});
    drawHistogram(); renderThreshCounts(dnormData);
  });
  window.addEventListener('mouseup', () => { if (dragging) saveThresholds(); dragging=null; });
}

_loadBookmarks();
renderBookmarkList();
loadSettings().then(() => fetchHistogram());
setupCanvasDrag();
