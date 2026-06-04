/* ==========================================================================
   app.js — SteamPredict v2 (SPA): constructor de juego + 5 pestañas
   ========================================================================== */

let CFG = null;
let MODE = 'presentacion';
let TAB = 'simulador';
let simModel = 'mlp';
const features = {};
let lastPredict = null, lastSimilar = null, lastRec = null, statsData = null;
let realGame = null;        // juego real cargado (modo validación)
let timer = null;

// Controles numéricos expuestos (los num_* y dev_success_prior se derivan en backend)
const NUM_UI = {
  price: { label: 'Precio (USD)', min: 0, max: 60, step: 1, fmt: v => '$' + (+v).toFixed(0) },
  supported_languages: { label: 'Idiomas soportados', min: 1, max: 30, step: 1 },
  total_achievements: { label: 'Logros', min: 0, max: 100, step: 1 },
  total_dlcs: { label: 'DLCs', min: 0, max: 20, step: 1 },
  min_ram_gb: { label: 'RAM mínima (GB)', min: 0, max: 16, step: 1 },
  short_desc_len: { label: 'Long. descripción', min: 0, max: 600, step: 10 },
};
const CAT_UI = {
  dev_experience: 'Experiencia del estudio',
  pub_experience: 'Experiencia del publisher',
  controller_support: 'Soporte de control',
  release_quarter: 'Trimestre de lanzamiento',
};
const EXP_GAMES = { Novato: 1, Establecido: 5, AAA: 15 };
const CLS = ['Flop', 'Rentable', 'Hit'];
const CLS_COLOR = { Flop: '#c75450', Rentable: '#66c0f4', Hit: '#a4d007' };

const $ = s => document.querySelector(s);
const overlay = (on) => { $('#overlay').style.display = on ? 'flex' : 'none'; };
const pct = x => (x * 100).toFixed(1) + '%';
const fmtOwners = n => n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1000 ? Math.round(n / 1000) + 'k' : '' + n;
const icons = () => window.lucide && lucide.createIcons();

// ── Inicio ─────────────────────────────────────────────────────────────
async function init() {
  CFG = await API.get('/api/config');
  $('#nav-meta').innerHTML =
    `Modelos: <b>${CFG.models.join(' · ').toUpperCase()}</b><br>` +
    `Umbral Hit: ${CFG.thresholds.hit_min / 1e6}M owners<br>` +
    `Flop: &lt;${CFG.thresholds.flop_max / 1000}k owners`;
  initFeatures();
  buildForm();
  bindUI();
  setMode('presentacion');
  setTab('simulador');
}

function initFeatures() {
  for (const c in CFG.numeric) features[c] = CFG.numeric[c].median;
  ['genre', 'cat', 'tag', 'platform'].forEach(g => CFG.groups[g].forEach(it => features[it.col] = 0));
  for (const c in CFG.categorical) features[c] = CFG.categorical[c].default;
  features['platform_windows'] = 1;   // default razonable
  features['cat_single_player'] = 1;
}

// ── Constructor del formulario ─────────────────────────────────────────
function buildForm() {
  const f = $('#form');
  f.innerHTML = '';

  // Numéricos
  const numWrap = document.createElement('div');
  numWrap.className = 'space-y-3';
  for (const col in NUM_UI) {
    if (!(col in CFG.numeric)) continue;
    numWrap.appendChild(numField(col, NUM_UI[col]));
  }
  f.appendChild(numWrap);

  // Categóricos
  const catWrap = document.createElement('div');
  catWrap.className = 'grid grid-cols-2 gap-2 pt-1';
  for (const col in CAT_UI) {
    if (!(col in CFG.categorical)) continue;
    catWrap.appendChild(catField(col, CAT_UI[col]));
  }
  f.appendChild(catWrap);

  // Grupos de chips
  f.appendChild(chipGroup('Géneros', CFG.groups.genre, true));
  f.appendChild(chipGroup('Etiquetas (tags)', CFG.groups.tag, MODE === 'presentacion'));
  f.appendChild(chipGroup('Características', CFG.groups.cat, false));
  f.appendChild(chipGroup('Plataformas', CFG.groups.platform, true));
  icons();
}

function numField(col, ui) {
  const d = document.createElement('div');
  const val = features[col];
  d.innerHTML = `
    <div class="field-label"><span>${ui.label}</span><span class="field-val" id="lab-${col}">${ui.fmt ? ui.fmt(val) : val}</span></div>
    <input type="range" min="${ui.min}" max="${ui.max}" step="${ui.step}" value="${Math.min(ui.max, Math.max(ui.min, val))}" class="w-full" id="in-${col}">`;
  d.querySelector('input').addEventListener('input', e => {
    features[col] = +e.target.value;
    $('#lab-' + col).textContent = ui.fmt ? ui.fmt(features[col]) : features[col];
    onChange();
  });
  return d;
}

function catField(col, label) {
  const d = document.createElement('div');
  const opts = CFG.categorical[col].categories
    .filter(c => c !== 'Desconocido')
    .map(c => `<option value="${c}" ${features[col] === c ? 'selected' : ''}>${c}</option>`).join('');
  d.innerHTML = `<label class="field-label"><span>${label}</span></label>
    <select id="in-${col}" class="w-full bg-steam-bg border border-steam-border rounded px-2 py-1.5 text-sm">${opts}</select>`;
  d.querySelector('select').addEventListener('change', e => {
    features[col] = e.target.value;
    if (col === 'dev_experience') features['dev_game_count'] = EXP_GAMES[e.target.value] || 1;
    if (col === 'pub_experience') features['pub_game_count'] = EXP_GAMES[e.target.value] || 1;
    onChange();
  });
  return d;
}

function chipGroup(title, items, openByDefault) {
  const det = document.createElement('details');
  det.open = openByDefault;
  det.className = 'border-t border-steam-border pt-2';
  const popular = MODE === 'quick' ? items.filter(i => i.prevalencia >= 0.12) : items;
  det.innerHTML = `<summary class="section-title cursor-pointer mb-2 select-none">${title} <span class="text-steam-muted">(${popular.length})</span></summary>`;
  const wrap = document.createElement('div');
  wrap.className = 'flex flex-wrap gap-1.5';
  popular.sort((a, b) => b.prevalencia - a.prevalencia).forEach(it => {
    const c = document.createElement('span');
    c.className = 'chip' + (features[it.col] ? ' on' : '');
    c.textContent = it.label;
    c.title = 'Prevalencia: ' + pct(it.prevalencia);
    c.addEventListener('click', () => {
      features[it.col] = features[it.col] ? 0 : 1;
      c.classList.toggle('on', !!features[it.col]);
      onChange();
    });
    wrap.appendChild(c);
  });
  det.appendChild(wrap);
  return det;
}

// ── Interacción ────────────────────────────────────────────────────────
function bindUI() {
  document.querySelectorAll('.nav-btn').forEach(b =>
    b.addEventListener('click', () => setTab(b.dataset.tab)));
  document.querySelectorAll('.mode-btn').forEach(b =>
    b.addEventListener('click', () => setMode(b.dataset.mode)));
  $('#reset-btn').addEventListener('click', () => { realGame = null; initFeatures(); buildForm(); runActive(); });
  bindSearch();
}

function setMode(m) {
  MODE = m;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === m));
  buildForm();
}

function setTab(name) {
  TAB = name;
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  const panelMode = name === 'panel';
  $('#input-panel').style.display = panelMode ? 'none' : '';
  $('#layout').style.gridTemplateColumns = panelMode ? '1fr' : '380px 1fr';
  runActive();
}

function onChange() { clearTimeout(timer); timer = setTimeout(runActive, 350); }

async function runActive() {
  overlay(true);
  try {
    if (TAB === 'simulador' || TAB === 'comparacion') {
      lastPredict = await API.post('/api/predict', { features });
    } else if (TAB === 'similares') {
      lastSimilar = await API.post('/api/similar', { features, k: 9 });
    } else if (TAB === 'recomendaciones') {
      lastRec = await API.post('/api/recommend', { features, K: 3, model: simModel });
    } else if (TAB === 'panel' && !statsData) {
      statsData = await API.get('/api/stats');
    }
    if ((TAB === 'comparacion' || TAB === 'panel') && !statsData) statsData = await API.get('/api/stats');
    render();
  } catch (e) {
    $('#content').innerHTML = `<div class="card text-steam-flop">Error: ${e.message}. ¿Está corriendo el backend? (python app/server.py)</div>`;
  } finally { overlay(false); }
}

// ── Render por pestaña ─────────────────────────────────────────────────
function render() {
  const fns = { simulador: renderSimulador, comparacion: renderComparacion, similares: renderSimilares, recomendaciones: renderRecomendaciones, panel: renderPanel };
  (fns[TAB] || renderSimulador)();
  icons();
}

function badge(cls) { return `<span class="badge badge-${cls.toLowerCase()}">${cls}</span>`; }

function probBars(probs) {
  return CLS.map(cl => `
    <div class="mb-2">
      <div class="flex justify-between text-xs mb-1"><span>${cl}</span><span class="font-semibold" style="color:${CLS_COLOR[cl]}">${pct(probs[cl])}</span></div>
      <div class="prob-track"><div class="prob-fill" style="width:${(probs[cl] * 100).toFixed(1)}%;background:${CLS_COLOR[cl]}"></div></div>
    </div>`).join('');
}

function renderSimulador() {
  if (!lastPredict) return;
  const m = lastPredict.modelos[simModel];
  const unc = lastPredict.incertidumbre;
  const noFlop = m.probs.Rentable + m.probs.Hit;
  const modelBtns = CFG.models.map(mm =>
    `<button class="px-2.5 py-1 rounded text-xs ${mm === simModel ? 'bg-steam-accent text-steam-bg font-semibold' : 'bg-steam-bg text-steam-muted'}" onclick="selModel('${mm}')">${mm.toUpperCase()}</button>`).join('');

  const validation = realGame ? `
    <div class="card fade-in border-l-4" style="border-left-color:${CLS_COLOR[realGame.clase_real]}">
      <div class="text-xs text-steam-muted mb-1">MODO VALIDACIÓN — juego real cargado</div>
      <div class="flex items-center justify-between">
        <div><b>${realGame.name}</b> · resultado real: ${badge(realGame.clase_real)} (${fmtOwners(realGame.owners_real)} owners)</div>
        <div class="text-sm">Predicción ${simModel.toUpperCase()}: ${badge(m.clase)} ${m.clase === realGame.clase_real ? '<span class="text-steam-hit">✓ acierta</span>' : '<span class="text-steam-flop">✗ falla</span>'}</div>
      </div>
    </div>` : '';

  const uncColor = { baja: '#a4d007', media: '#f5a623', alta: '#c75450' }[unc.nivel];
  $('#content').innerHTML = `
    <div class="space-y-5 fade-in">
      ${validation}
      <div class="grid grid-cols-3 gap-4">
        ${kpi('Clasificación', `<div class="text-2xl font-bold">${badge(m.clase)}</div>`, 'target')}
        ${kpi('Prob. de no-Flop', `<div class="text-2xl font-bold text-steam-accent">${pct(noFlop)}</div>`, 'shield-check')}
        ${kpi('Owners estimados', `<div class="text-2xl font-bold text-steam-hit">${fmtOwners(lastPredict.owners_estimados)}</div>`, 'users')}
      </div>
      <div class="grid gap-4" style="grid-template-columns: 1.3fr 1fr;">
        <div class="card">
          <div class="flex items-center justify-between mb-3">
            <h3 class="font-semibold">Distribución de probabilidad</h3>
            <div class="flex gap-1">${modelBtns}</div>
          </div>
          ${probBars(m.probs)}
        </div>
        <div class="card">
          <h3 class="font-semibold mb-3">Confianza del modelo</h3>
          <div class="flex items-center gap-3 mb-3">
            <div class="w-3 h-3 rounded-full" style="background:${uncColor}"></div>
            <div>Incertidumbre <b style="color:${uncColor}">${unc.nivel.toUpperCase()}</b></div>
          </div>
          <p class="text-sm text-steam-muted mb-2">Margen entre las 2 clases más probables: <b class="text-steam-text">${pct(unc.margen_top2)}</b></p>
          <p class="text-sm text-steam-muted">${unc.desacuerdo
            ? '⚠️ Los modelos <b class="text-steam-text">no coinciden</b> en la clase: ' + CFG.models.map(mm => `${mm.toUpperCase()}=${lastPredict.modelos[mm].clase}`).join(', ') + '. Predicción inestable.'
            : '✓ Los tres modelos <b class="text-steam-text">coinciden</b> en la clasificación.'}</p>
        </div>
      </div>
    </div>`;
}

function selModel(m) { simModel = m; render(); }

function kpi(label, valueHtml, icon) {
  return `<div class="card flex items-center gap-3">
    <i data-lucide="${icon}" class="w-8 h-8 text-steam-accentd"></i>
    <div><div class="text-xs text-steam-muted">${label}</div>${valueHtml}</div></div>`;
}

function renderComparacion() {
  if (!lastPredict) return;
  const met = statsData ? statsData.metrics : {};
  const rows = CFG.models.map(m => {
    const p = lastPredict.modelos[m];
    const mm = met[m] || {};
    return `<tr class="border-t border-steam-border">
      <td class="py-2 font-semibold">${m.toUpperCase()}</td>
      <td>${badge(p.clase)}</td>
      <td class="text-steam-flop">${pct(p.probs.Flop)}</td>
      <td class="text-steam-rentable">${pct(p.probs.Rentable)}</td>
      <td class="text-steam-hit">${pct(p.probs.Hit)}</td>
      <td>${mm.auc_ovr_macro ?? '—'}</td>
      <td>${mm.f1_macro ?? '—'}</td>
      <td>${mm.accuracy ?? '—'}</td></tr>`;
  }).join('');
  const agree = new Set(CFG.models.map(m => lastPredict.modelos[m].clase)).size === 1;
  $('#content').innerHTML = `
    <div class="space-y-5 fade-in">
      <div class="card ${agree ? '' : 'border-l-4 border-steam-flop'}">
        ${agree ? '<span class="text-steam-hit">✓ Los tres modelos coinciden</span>' : '<span class="text-steam-flop">⚠️ Los modelos discrepan — zona de incertidumbre</span>'}
      </div>
      <div class="card">
        <h3 class="font-semibold mb-3">Probabilidad por clase y modelo</h3>
        <div id="cmp-bars" class="chart"></div>
      </div>
      <div class="card overflow-x-auto">
        <h3 class="font-semibold mb-3">Predicción y desempeño en test</h3>
        <table class="w-full text-sm text-left">
          <thead class="text-steam-muted text-xs"><tr>
            <th class="py-1">Modelo</th><th>Predice</th><th>P(Flop)</th><th>P(Rentable)</th><th>P(Hit)</th><th>AUC</th><th>F1</th><th>Acc</th>
          </tr></thead><tbody>${rows}</tbody>
        </table>
        <p class="text-xs text-steam-muted mt-2">AUC/F1/Accuracy medidos en el conjunto de test (split 80/20). P(·) es la predicción para el juego simulado.</p>
      </div>
    </div>`;
  Charts.modelBars($('#cmp-bars'), lastPredict.modelos);
}

function renderSimilares() {
  if (!lastSimilar) return;
  const cards = lastSimilar.juegos.map(j => `
    <div class="card flex flex-col gap-1">
      <div class="flex justify-between items-start gap-2">
        <b class="text-sm leading-tight">${j.name}</b>${badge(j.clase)}
      </div>
      <div class="text-xs text-steam-muted">${j.developer || '—'}</div>
      <div class="flex justify-between text-xs mt-1">
        <span>${fmtOwners(j.owners)} owners</span>
        <span class="text-steam-accent">${pct(j.similitud)} similar</span>
      </div>
    </div>`).join('');
  $('#content').innerHTML = `
    <div class="space-y-4 fade-in">
      <div class="card-h card"><h3 class="font-semibold flex items-center gap-2"><i data-lucide="users" class="w-4 h-4 text-steam-accent"></i> Juegos del mismo camino</h3>
        <p class="text-sm text-steam-muted mt-1">Juegos reales con un perfil de features parecido al tuyo, y cómo les fue comercialmente.</p></div>
      <div class="grid grid-cols-3 gap-3">${cards}</div>
    </div>`;
}

function renderRecomendaciones() {
  if (!lastRec) return;
  const r = lastRec;
  const arrow = r.delta >= 0 ? '#a4d007' : '#c75450';
  const pkg = r.cambios.length ? r.cambios.map((c, i) => `
    <div class="flex items-center gap-3 py-2 border-t border-steam-border">
      <span class="w-6 h-6 rounded-full bg-steam-bg flex items-center justify-center text-xs text-steam-accent">${i + 1}</span>
      <span class="flex-1 text-sm">${c.desc}</span>
      <span class="text-xs text-steam-muted">P(Hit) → ${pct(c.phit)}</span>
    </div>`).join('') : '<p class="text-sm text-steam-muted py-2">Tu configuración ya es buena; no se hallaron mejoras claras.</p>';
  const singles = r.cambios_individuales.map(c => `
    <div class="flex justify-between text-sm py-1.5 border-t border-steam-border">
      <span>${c.desc}</span>
      <span style="color:${c.delta >= 0 ? '#a4d007' : '#c75450'}">${c.delta >= 0 ? '+' : ''}${pct(c.delta)}</span>
    </div>`).join('');
  $('#content').innerHTML = `
    <div class="space-y-5 fade-in">
      <div class="grid grid-cols-3 gap-4">
        ${kpi('P(Hit) actual', `<div class="text-2xl font-bold">${pct(r.phit_antes)}</div>`, 'flag')}
        ${kpi('P(Hit) optimizada', `<div class="text-2xl font-bold text-steam-hit">${pct(r.phit_despues)}</div>`, 'rocket')}
        ${kpi('Mejora', `<div class="text-2xl font-bold" style="color:${arrow}">${r.delta >= 0 ? '+' : ''}${pct(r.delta)}</div>`, 'trending-up')}
      </div>
      <div class="grid gap-4" style="grid-template-columns:1.2fr 1fr;">
        <div class="card"><h3 class="font-semibold mb-2 flex items-center gap-2"><i data-lucide="package" class="w-4 h-4 text-steam-accent"></i> Mejor paquete de cambios <span class="text-xs text-steam-muted">(modelo ${r.modelo.toUpperCase()})</span></h3>${pkg}</div>
        <div class="card"><h3 class="font-semibold mb-2">Impacto de cambios individuales</h3>${singles}</div>
      </div>
      <p class="text-xs text-steam-muted">Solo se consideran cambios realistas (tags con prevalencia 5–95%, precio en rango, idiomas, control, trimestre). No se togglean features degeneradas como la plataforma.</p>
    </div>`;
}

function renderPanel() {
  if (!statsData) return;
  const s = statsData, met = s.metrics;
  const metRows = ['lr', 'svm', 'mlp'].filter(m => met[m]).map(m =>
    `<tr class="border-t border-steam-border"><td class="py-1 font-semibold">${m.toUpperCase()}</td><td>${met[m].auc_ovr_macro}</td><td>${met[m].f1_macro}</td><td>${met[m].accuracy}</td></tr>`).join('');
  $('#content').innerHTML = `
    <div class="space-y-4 fade-in">
      <div class="grid grid-cols-4 gap-4">
        ${kpi('Juegos analizados', `<div class="text-2xl font-bold">${s.n_total.toLocaleString()}</div>`, 'database')}
        ${kpi('Mejor AUC', `<div class="text-2xl font-bold text-steam-accent">${Math.max(...['lr', 'svm', 'mlp'].filter(m => met[m]).map(m => met[m].auc_ovr_macro))}</div>`, 'award')}
        ${kpi('Clases', `<div class="text-2xl font-bold">3</div>`, 'layers')}
        ${kpi('Features', `<div class="text-2xl font-bold">${met._meta ? met._meta.features : '—'}</div>`, 'list')}
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div class="card"><h3 class="font-semibold mb-2">Distribución del mercado</h3><div id="c-market" class="chart-sm"></div></div>
        <div class="card"><h3 class="font-semibold mb-2">Distribución de precios</h3><div id="c-price" class="chart-sm"></div></div>
        <div class="card"><h3 class="font-semibold mb-2">Owners mediana por género</h3><div id="c-genre" class="chart"></div></div>
        <div class="card"><h3 class="font-semibold mb-2">Variables que empujan a Hit (LR)</h3><div id="c-imp" class="chart"></div></div>
        <div class="card"><h3 class="font-semibold mb-2">Precio vs Owners (muestra)</h3><div id="c-scatter" class="chart"></div></div>
        <div class="card"><h3 class="font-semibold mb-2">Matriz de confusión (MLP)</h3><div id="c-conf" class="chart"></div></div>
      </div>
      <div class="card"><h3 class="font-semibold mb-2">Desempeño de los modelos (test)</h3>
        <table class="w-full text-sm text-left"><thead class="text-steam-muted text-xs"><tr><th class="py-1">Modelo</th><th>AUC</th><th>F1</th><th>Accuracy</th></tr></thead><tbody>${metRows}</tbody></table>
      </div>
    </div>`;
  Charts.marketDonut($('#c-market'), s.market);
  Charts.priceHist($('#c-price'), s.price_hist);
  Charts.ownersByGenre($('#c-genre'), s.owners_by_genre);
  Charts.importance($('#c-imp'), s.lr_importance);
  Charts.scatter($('#c-scatter'), s.scatter);
  if (s.confusion.mlp) Charts.confusion($('#c-conf'), s.confusion.mlp);
}

// ── Búsqueda y carga de juego real (modo validación) ───────────────────
function bindSearch() {
  const inp = $('#game-search'), box = $('#game-results');
  let t = null;
  inp.addEventListener('input', () => {
    clearTimeout(t);
    const q = inp.value.trim();
    if (q.length < 2) { box.classList.add('hidden'); return; }
    t = setTimeout(async () => {
      const r = await API.get('/api/games?q=' + encodeURIComponent(q));
      box.innerHTML = r.juegos.map(j =>
        `<div class="px-3 py-2 hover:bg-steam-border cursor-pointer text-sm flex justify-between" onclick="loadGame(${j.appid})">
          <span>${j.name}</span>${badge(j.clase)}</div>`).join('') || '<div class="px-3 py-2 text-sm text-steam-muted">Sin resultados</div>';
      box.classList.remove('hidden');
    }, 300);
  });
  document.addEventListener('click', e => { if (!inp.contains(e.target) && !box.contains(e.target)) box.classList.add('hidden'); });
}

async function loadGame(appid) {
  $('#game-results').classList.add('hidden');
  $('#game-search').value = '';
  overlay(true);
  const g = await API.get('/api/game/' + appid);
  overlay(false);
  if (g.error) return;
  realGame = g;
  for (const k in g.features) if (k in features) features[k] = g.features[k];
  buildForm();
  setTab('simulador');
}

window.selModel = selModel;
window.loadGame = loadGame;
document.addEventListener('DOMContentLoaded', init);
