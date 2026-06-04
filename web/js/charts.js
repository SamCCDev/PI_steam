/* ==========================================================================
   charts.js — Gráficos ECharts con estética Steam (para el panel y comparación)
   ========================================================================== */
const Charts = (function () {
  const AX = '#8f98a0', GRID = '#2a475e', TXT = '#c7d5e0';
  const CLS = { Flop: '#c75450', Rentable: '#66c0f4', Hit: '#a4d007' };
  const palette = ['#66c0f4', '#a4d007', '#c75450', '#f5a623', '#9b59b6', '#1abc9c'];
  const reg = {};

  function base() { return { textStyle: { color: TXT, fontFamily: 'system-ui' }, color: palette, backgroundColor: 'transparent' }; }
  function get(el) {
    if (reg[el.id]) { reg[el.id].dispose(); }
    const c = echarts.init(el, null, { renderer: 'canvas' });
    reg[el.id] = c; return c;
  }

  // Barras agrupadas: 3 modelos x 3 clases
  function modelBars(el, modelos) {
    const c = get(el), classes = ['Flop', 'Rentable', 'Hit'];
    const series = Object.keys(modelos).map(m => ({
      name: m.toUpperCase(), type: 'bar',
      data: classes.map(cl => +(modelos[m].probs[cl] * 100).toFixed(1)),
      label: { show: true, position: 'top', formatter: '{c}%', color: TXT, fontSize: 10 }
    }));
    c.setOption({
      ...base(), tooltip: { trigger: 'axis', valueFormatter: v => v + '%' },
      legend: { top: 0, textStyle: { color: TXT } }, grid: { top: 36, left: 44, right: 12, bottom: 24 },
      xAxis: { type: 'category', data: classes, axisLine: { lineStyle: { color: GRID } }, axisLabel: { color: AX } },
      yAxis: { type: 'value', max: 100, axisLabel: { color: AX, formatter: '{value}%' }, splitLine: { lineStyle: { color: GRID } } },
      series
    });
  }

  function marketDonut(el, market) {
    const c = get(el);
    c.setOption({
      ...base(), tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      legend: { bottom: 0, textStyle: { color: TXT } },
      series: [{
        type: 'pie', radius: ['45%', '70%'], center: ['50%', '45%'],
        itemStyle: { borderColor: '#16202d', borderWidth: 2 },
        label: { color: TXT, formatter: '{b}\n{d}%' },
        data: market.map(m => ({ name: m.clase, value: m.n, itemStyle: { color: CLS[m.clase] } }))
      }]
    });
  }

  function priceHist(el, ph) {
    const c = get(el);
    c.setOption({
      ...base(), tooltip: { trigger: 'axis' }, grid: { top: 16, left: 48, right: 12, bottom: 40 },
      xAxis: { type: 'category', data: ph.labels, axisLine: { lineStyle: { color: GRID } }, axisLabel: { color: AX, rotate: 30 } },
      yAxis: { type: 'value', axisLabel: { color: AX }, splitLine: { lineStyle: { color: GRID } } },
      series: [{ type: 'bar', data: ph.counts, itemStyle: { color: '#66c0f4' } }]
    });
  }

  function ownersByGenre(el, rows) {
    const c = get(el); const r = rows.slice().reverse();
    c.setOption({
      ...base(), tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { top: 16, left: 130, right: 30, bottom: 24 },
      xAxis: { type: 'value', axisLabel: { color: AX, formatter: v => (v / 1000) + 'k' }, splitLine: { lineStyle: { color: GRID } } },
      yAxis: { type: 'category', data: r.map(x => x.genero), axisLabel: { color: AX }, axisLine: { lineStyle: { color: GRID } } },
      series: [{
        type: 'bar', data: r.map(x => x.owners_mediana), itemStyle: { color: '#a4d007' },
        label: { show: true, position: 'right', color: TXT, fontSize: 10, formatter: p => (p.value / 1000) + 'k' }
      }]
    });
  }

  function importance(el, imp) {
    const c = get(el);
    const pos = imp.positivos.map(x => ({ name: x.feature, value: x.coef }));
    const neg = imp.negativos.map(x => ({ name: x.feature, value: x.coef }));
    const all = neg.concat(pos);
    c.setOption({
      ...base(), tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { top: 10, left: 160, right: 20, bottom: 24 },
      xAxis: { type: 'value', axisLabel: { color: AX }, splitLine: { lineStyle: { color: GRID } } },
      yAxis: { type: 'category', data: all.map(x => x.name), axisLabel: { color: AX, fontSize: 11 }, axisLine: { lineStyle: { color: GRID } } },
      series: [{
        type: 'bar', data: all.map(x => ({ value: x.value, itemStyle: { color: x.value >= 0 ? '#a4d007' : '#c75450' } })),
      }]
    });
  }

  function confusion(el, conf) {
    const c = get(el); const labels = conf.labels; const m = conf.matrix;
    const data = [];
    let max = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { data.push([j, i, m[i][j]]); max = Math.max(max, m[i][j]); }
    c.setOption({
      ...base(), tooltip: { formatter: p => `real ${labels[p.value[1]]} → pred ${labels[p.value[0]]}: ${p.value[2]}` },
      grid: { top: 24, left: 70, right: 12, bottom: 40 },
      xAxis: { type: 'category', data: labels, name: 'Predicho', nameLocation: 'middle', nameGap: 28, axisLabel: { color: AX }, splitArea: { show: true } },
      yAxis: { type: 'category', data: labels, name: 'Real', axisLabel: { color: AX }, splitArea: { show: true } },
      visualMap: { min: 0, max, show: false, inRange: { color: ['#16202d', '#2a475e', '#66c0f4'] } },
      series: [{ type: 'heatmap', data, label: { show: true, color: '#fff', formatter: p => p.value[2] } }]
    });
  }

  function scatter(el, pts) {
    const c = get(el); const byCls = { Flop: [], Rentable: [], Hit: [] };
    pts.forEach(p => { if (byCls[p[2]]) byCls[p[2]].push([p[0], p[1]]); });
    c.setOption({
      ...base(), tooltip: { trigger: 'item', formatter: p => `$${p.value[0]} · ${(p.value[1] / 1000).toFixed(0)}k owners` },
      legend: { top: 0, textStyle: { color: TXT } }, grid: { top: 36, left: 60, right: 16, bottom: 40 },
      xAxis: { type: 'value', name: 'Precio (USD)', nameLocation: 'middle', nameGap: 26, axisLabel: { color: AX }, splitLine: { lineStyle: { color: GRID } } },
      yAxis: { type: 'log', name: 'Owners', axisLabel: { color: AX, formatter: v => v >= 1e6 ? (v / 1e6) + 'M' : (v / 1000) + 'k' }, splitLine: { lineStyle: { color: GRID } } },
      series: ['Flop', 'Rentable', 'Hit'].map(cl => ({
        name: cl, type: 'scatter', symbolSize: 6, data: byCls[cl],
        itemStyle: { color: CLS[cl], opacity: .55 }
      }))
    });
  }

  function resizeAll() { Object.values(reg).forEach(c => c.resize()); }
  return { modelBars, marketDonut, priceHist, ownersByGenre, importance, confusion, scatter, resizeAll };
})();
window.addEventListener('resize', () => Charts.resizeAll());
