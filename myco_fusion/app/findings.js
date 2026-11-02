'use strict';

// Charts and the Findings page. Shared by the local app and the public site, with no dependencies.
// Labels come from data files, so everything is inserted with textContent, never as HTML.
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const COLORS = ['#1f7a54', '#c2622a', '#2f77b5'];
  const INK = { axis: '#55645a', grid: '#e3e7dc', surface: '#fffefa', text: '#21382f' };

  function el(tag, cls, text) { const node = document.createElement(tag); if (cls) node.className = cls; if (text !== undefined) node.textContent = text; return node; }
  function svg(tag, attrs, text) { const node = document.createElementNS(NS, tag); Object.entries(attrs || {}).forEach(([key, value]) => node.setAttribute(key, value)); if (text !== undefined) node.textContent = text; return node; }
  function format(value, chart) { return Number.isFinite(value) ? `${value.toFixed(chart.digits ?? 2)}${chart.unit || ''}` : '—'; }
  function niceMax(value) { const power = 10 ** Math.floor(Math.log10(value || 1)); return Math.ceil(value / power * 2) / 2 * power; }
  // Round tick values, so an axis never reads -0.04, 0.22, 0.48.
  function ticks(low, high) {
    const range = high - low || 1, power = 10 ** Math.floor(Math.log10(range));
    const step = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => range / s <= 5) || range;
    const out = [];
    for (let k = Math.ceil(low / step - 1e-9); k * step <= high + 1e-9; k += 1) out.push(k * step);
    return out;
  }

  function shell(container, chart, height) {
    const width = Math.max(container.clientWidth || 0, 280);
    const root = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': chart.title || 'Chart' });
    if (chart.title) root.append(svg('title', {}, chart.title));
    const tip = el('div', 'chart-tip'); tip.hidden = true;
    const wrap = el('div', 'chart-wrap'); wrap.append(root, tip);
    container.replaceChildren(wrap);
    const show = (rows, x, y) => {
      tip.replaceChildren(...rows.map(([value, label, color]) => {
        const row = el('div', 'chart-tip-row');
        if (color) { const key = el('i', 'chart-tip-key'); key.style.background = color; row.append(key); }
        row.append(el('strong', '', value)); if (label) row.append(el('span', '', label));
        return row;
      }));
      tip.hidden = false;
      const box = tip.getBoundingClientRect();
      tip.style.left = `${Math.max(0, Math.min(x + 12, width - box.width))}px`;
      tip.style.top = `${Math.max(0, y - box.height - 8)}px`;
    };
    const hide = () => { tip.hidden = true; };
    return { root, width, show, hide };
  }

  function legend(series) {
    const box = el('div', 'chart-legend');
    series.forEach((item, index) => { const entry = el('span'); const key = el('i', 'chart-key'); key.style.background = item.color || COLORS[index]; entry.append(key, document.createTextNode(item.label)); box.append(entry); });
    return box;
  }

  function bars(container, chart) {
    const stacked = (container.clientWidth || 0) < 520;
    const rowHeight = stacked ? 52 : 40, top = 6, bottom = 8, height = top + bottom + chart.rows.length * rowHeight;
    const { root, width, show, hide } = shell(container, chart, height);
    const labelWidth = stacked ? 0 : Math.min(Math.round(width * 0.42), 230), valueSpace = 64;
    const max = chart.max ?? niceMax(Math.max(...chart.rows.map((row) => row.value)));
    const scale = (value) => (value / max) * (width - labelWidth - valueSpace - 8);
    if (!stacked) root.append(svg('line', { x1: labelWidth, x2: labelWidth, y1: top, y2: height - bottom, stroke: INK.grid }));
    chart.rows.forEach((row, index) => {
      const y = top + index * rowHeight, barHeight = 20, center = stacked ? y + 34 : y + rowHeight / 2;
      const label = stacked
        ? svg('text', { x: 0, y: y + 14, fill: INK.text, 'font-size': 13 }, row.label)
        : svg('text', { x: labelWidth - 10, y: center + 5, 'text-anchor': 'end', fill: INK.text, 'font-size': 13 }, row.label);
      const w = Math.max(scale(row.value), 2), r = Math.min(4, w);
      const bar = svg('path', { d: `M${labelWidth},${center - barHeight / 2} h${w - r} a${r},${r} 0 0 1 ${r},${r} v${barHeight - 2 * r} a${r},${r} 0 0 1 ${-r},${r} h${r - w} z`, fill: row.color || COLORS[0], tabindex: 0, class: 'chart-mark' });
      const value = svg('text', { x: labelWidth + w + 8, y: center + 5, fill: INK.text, 'font-size': 13, 'font-weight': 600 }, format(row.value, chart));
      const reveal = () => show([[format(row.value, chart), row.label]], labelWidth + w, y + 6);
      bar.addEventListener('pointermove', reveal); bar.addEventListener('focus', reveal); bar.addEventListener('pointerleave', hide); bar.addEventListener('blur', hide);
      root.append(label, bar, value);
    });
    fitLabels(root, labelWidth - 14);
  }

  function dots(container, chart) {
    const stacked = (container.clientWidth || 0) < 520;
    const rowHeight = stacked ? 54 : 38, top = 30, bottom = 44;
    const height = top + bottom + chart.rows.length * rowHeight;
    const { root, width, show, hide } = shell(container, chart, height);
    const labelWidth = stacked ? 14 : Math.min(Math.round(width * 0.36), 240), right = 22;
    const values = chart.rows.flatMap((row) => chart.series.map((s) => row.values[s.key]));
    const min = chart.min ?? 0, max = chart.max ?? niceMax(Math.max(...values));
    const x = (value) => labelWidth + (value - min) / (max - min) * (width - labelWidth - right);
    ticks(min, max).forEach((tick) => {
      root.append(svg('line', { x1: x(tick), x2: x(tick), y1: top - 8, y2: height - bottom + 6, stroke: INK.grid }));
      root.append(svg('text', { x: x(tick), y: height - bottom + 24, 'text-anchor': 'middle', fill: INK.axis, 'font-size': 12 }, `${tick.toFixed(chart.digits === 0 ? 0 : (chart.digits ?? 2))}${chart.unit || ''}`));
    });
    if (chart.reference) {
      const rx = x(chart.reference.value);
      root.append(svg('line', { x1: rx, x2: rx, y1: top - 10, y2: height - bottom + 6, stroke: INK.axis, 'stroke-width': 1.5 }));
      const anchor = rx > width * 0.7 ? 'end' : rx < width * 0.3 ? 'start' : 'middle';
      root.append(svg('text', { x: rx, y: top - 15, 'text-anchor': anchor, fill: INK.axis, 'font-size': 12 }, `${chart.reference.label}: ${format(chart.reference.value, chart)}`));
    }
    chart.rows.forEach((row, index) => {
      const rowTop = top + index * rowHeight, cy = stacked ? rowTop + 36 : rowTop + rowHeight / 2;
      root.append(stacked
        ? svg('text', { x: 0, y: rowTop + 15, fill: INK.text, 'font-size': 13 }, row.label)
        : svg('text', { x: labelWidth - 12, y: cy + 5, 'text-anchor': 'end', fill: INK.text, 'font-size': 13 }, row.label));
      const points = chart.series.map((s) => row.values[s.key]);
      root.append(svg('line', { x1: x(Math.min(...points)), x2: x(Math.max(...points)), y1: cy, y2: cy, stroke: '#c9d1c2', 'stroke-width': 2 }));
      const readout = chart.series.map((s, i) => [format(row.values[s.key], chart), s.label, COLORS[i]]);
      const reveal = () => show([[row.label, ''], ...readout], x(Math.max(...points)), cy - 6);
      const hit = svg('rect', { x: labelWidth, y: rowTop, width: width - labelWidth, height: rowHeight, fill: 'transparent', tabindex: 0, class: 'chart-hit', 'aria-label': `${row.label}: ${readout.map((r) => `${r[1]} ${r[0]}`).join(', ')}` });
      hit.addEventListener('pointermove', reveal); hit.addEventListener('focus', reveal); hit.addEventListener('pointerleave', hide); hit.addEventListener('blur', hide);
      root.append(hit);
      chart.series.forEach((s, i) => root.append(svg('circle', { cx: x(row.values[s.key]), cy, r: 6, fill: COLORS[i], stroke: INK.surface, 'stroke-width': 2, 'pointer-events': 'none' })));
    });
    fitLabels(root, labelWidth - 16);
    container.prepend(legend(chart.series));
  }

  function lines(container, chart) {
    const narrow = (container.clientWidth || 0) < 520;
    const height = narrow ? 260 : 320, pad = { top: 18, right: 22, bottom: 50, left: 52 };
    const { root, width, show, hide } = shell(container, chart, height);
    const allY = chart.series.flatMap((s) => s.values).concat(chart.reference ? [chart.reference.value] : []);
    const yMin = chart.yMin ?? Math.min(...allY), yMax = chart.yMax ?? niceMax(Math.max(...allY));
    const positions = chart.ordinal ? chart.x.map((_, i) => i) : chart.x;
    const xLow = Math.min(...positions), xHigh = Math.max(...positions);
    const px = (index) => pad.left + (positions[index] - xLow) / (xHigh - xLow || 1) * (width - pad.left - pad.right);
    const py = (value) => height - pad.bottom - (value - yMin) / (yMax - yMin || 1) * (height - pad.top - pad.bottom);
    ticks(yMin, yMax).forEach((tick) => {
      root.append(svg('line', { x1: pad.left, x2: width - pad.right, y1: py(tick), y2: py(tick), stroke: INK.grid }));
      root.append(svg('text', { x: pad.left - 8, y: py(tick) + 4, 'text-anchor': 'end', fill: INK.axis, 'font-size': 12 }, `${tick.toFixed(chart.digits === 0 ? 0 : 2)}${chart.unit || ''}`));
    });
    const step = chart.ordinal ? 1 : Math.ceil(chart.x.length / (narrow ? 4 : 7));
    chart.x.forEach((value, index) => { if (index % step === 0 || index === chart.x.length - 1 && chart.ordinal) root.append(svg('text', { x: px(index), y: height - pad.bottom + 20, 'text-anchor': 'middle', fill: INK.axis, 'font-size': 12 }, String(value))); });
    if (chart.xLabel) root.append(svg('text', { x: (pad.left + width - pad.right) / 2, y: height - 8, 'text-anchor': 'middle', fill: INK.axis, 'font-size': 12 }, chart.xLabel));
    if (chart.reference) {
      root.append(svg('line', { x1: pad.left, x2: width - pad.right, y1: py(chart.reference.value), y2: py(chart.reference.value), stroke: INK.axis, 'stroke-width': 1.5 }));
      root.append(svg('text', { x: width - pad.right, y: py(chart.reference.value) - 6, 'text-anchor': 'end', fill: INK.axis, 'font-size': 12 }, chart.reference.label));
    }
    chart.series.forEach((series, s) => {
      const color = series.color || COLORS[s];
      const d = series.values.map((value, i) => (Number.isFinite(value) ? `${i && Number.isFinite(series.values[i - 1]) ? 'L' : 'M'}${px(i).toFixed(2)},${py(value).toFixed(2)}` : '')).join(' ');
      root.append(svg('path', { d, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': series.dashed ? '6 4' : '' }));
      if (chart.ordinal) series.values.forEach((value, i) => root.append(svg('circle', { cx: px(i), cy: py(value), r: 4, fill: color, stroke: INK.surface, 'stroke-width': 2 })));
    });
    const guide = svg('line', { y1: pad.top, y2: height - pad.bottom, stroke: '#9aa794', visibility: 'hidden' });
    root.append(guide);
    const overlay = svg('rect', { x: pad.left, y: pad.top, width: width - pad.left - pad.right, height: height - pad.top - pad.bottom, fill: 'transparent', tabindex: 0, class: 'chart-hit' });
    let cursor = 0;
    const reveal = (index) => {
      cursor = index;
      guide.setAttribute('x1', px(index)); guide.setAttribute('x2', px(index)); guide.setAttribute('visibility', 'visible');
      show([[`${chart.x[index]}`, chart.xLabel || ''], ...chart.series.map((s, i) => [format(s.values[index], chart), s.label, s.color || COLORS[i]])], px(index), pad.top + 40);
    };
    overlay.addEventListener('pointermove', (event) => {
      const rect = root.getBoundingClientRect(), mx = event.clientX - rect.left;
      let best = 0; chart.x.forEach((_, i) => { if (Math.abs(px(i) - mx) < Math.abs(px(best) - mx)) best = i; });
      reveal(best);
    });
    overlay.addEventListener('focus', () => reveal(cursor));
    overlay.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowRight') { reveal(Math.min(cursor + 1, chart.x.length - 1)); event.preventDefault(); }
      if (event.key === 'ArrowLeft') { reveal(Math.max(cursor - 1, 0)); event.preventDefault(); }
    });
    const clear = () => { hide(); guide.setAttribute('visibility', 'hidden'); };
    overlay.addEventListener('pointerleave', clear); overlay.addEventListener('blur', clear);
    root.append(overlay);
    if (!chart.hideLegend) container.prepend(legend(chart.series.map((s, i) => ({ label: s.label, color: s.color || COLORS[i] }))));
  }

  // Shrink long row labels to fit their column rather than letting them run off the chart.
  function fitLabels(root, limit) {
    root.querySelectorAll('text[text-anchor="end"]').forEach((text) => {
      if (Number(text.getAttribute('font-size')) !== 13) return;
      const full = text.textContent;
      if (text.getComputedTextLength() <= limit) return;
      const title = svg('title', {}, full); text.append(title);
      let size = 13;
      while (size > 11 && text.getComputedTextLength() > limit) { size -= 0.5; text.setAttribute('font-size', size); }
      if (text.getComputedTextLength() > limit) {
        let cut = full;
        while (cut.length > 4) { cut = cut.slice(0, -1); text.firstChild.textContent = `${cut}…`; if (text.getComputedTextLength() <= limit) break; }
      }
    });
  }

  const observed = new WeakMap();
  function draw(container, chart) {
    const render = { bars, dots, lines }[chart.type];
    if (!render) return;
    const paint = () => render(container, chart);
    paint();
    if (!window.ResizeObserver) return;
    const known = observed.get(container);
    if (known) { known.paint = paint; known.last = container.clientWidth; return; }
    const entry = { paint, last: container.clientWidth, pending: null };
    observed.set(container, entry);
    new ResizeObserver(() => {
      if (Math.abs(container.clientWidth - entry.last) < 8) return;
      entry.last = container.clientWidth; clearTimeout(entry.pending); entry.pending = setTimeout(() => entry.paint(), 80);
    }).observe(container);
  }

  function chartTable(chart) {
    const details = el('details', 'chart-table'); details.append(el('summary', '', 'Show as a table'));
    const table = el('table'), head = el('thead'), body = el('tbody'), headRow = el('tr');
    const value = (v) => format(v, chart);
    let headers, rows;
    if (chart.type === 'bars') { headers = ['', 'Value']; rows = chart.rows.map((row) => [row.label, value(row.value)]); }
    else if (chart.type === 'dots') { headers = ['', ...chart.series.map((s) => s.label)]; rows = chart.rows.map((row) => [row.label, ...chart.series.map((s) => value(row.values[s.key]))]); }
    else { headers = [chart.xLabel || '', ...chart.series.map((s) => s.label)]; rows = chart.x.map((x, i) => [String(x), ...chart.series.map((s) => value(s.values[i]))]); }
    headers.forEach((text) => headRow.append(el('th', '', text))); head.append(headRow);
    rows.forEach((cells) => { const tr = el('tr'); cells.forEach((text) => tr.append(el('td', '', text))); body.append(tr); });
    table.append(head, body);
    const scroll = el('div', 'table-scroll'); scroll.append(table); details.append(scroll);
    return details;
  }

  function dataTable(spec) {
    const panel = el('div', 'finding-table');
    if (spec.title) panel.append(el('p', 'chart-title', spec.title));
    const table = el('table'), head = el('thead'), body = el('tbody'), headRow = el('tr');
    spec.headers.forEach((text) => headRow.append(el('th', '', text))); head.append(headRow);
    spec.rows.forEach((cells) => { const tr = el('tr'); cells.forEach((text) => tr.append(el('td', '', String(text)))); body.append(tr); });
    table.append(head, body);
    const scroll = el('div', 'table-scroll'); scroll.append(table); panel.append(scroll);
    return panel;
  }

  function renderFindings(root, data) {
    const heading = el('div', 'page-heading compact findings-heading'), intro = el('div');
    const title = el('h1'); title.id = 'findings-title'; title.textContent = data.title;
    intro.append(el('div', 'eyebrow', data.eyebrow), title, ...data.answer.map((text) => el('p', 'lede', text)));
    heading.append(intro);
    const safety = el('div', 'notice safety'); safety.append(el('span', 'badge amber', 'READ THIS FIRST'), el('p', '', data.safety));
    const tiles = el('div', 'stat-tiles');
    data.tiles.forEach((tile) => { const card = el('article', 'panel stat-tile'); card.append(el('strong', '', tile.value), el('span', '', tile.label)); if (tile.note) card.append(el('small', 'stat-note', tile.note)); tiles.append(card); });
    const list = el('div', 'finding-list');
    const pending = [];
    data.findings.forEach((finding, index) => {
      const article = el('article', 'panel finding'); article.id = `finding-${finding.id}`;
      article.append(el('div', 'eyebrow', `FINDING ${String(index + 1).padStart(2, '0')}`), el('h2', '', finding.title));
      const copy = el('div', 'finding-copy');
      finding.body.forEach((text) => copy.append(el('p', '', text)));
      article.append(copy);
      if (finding.chart) {
        article.append(el('p', 'chart-title', finding.chart.title));
        const holder = el('div', 'finding-chart'); article.append(holder, chartTable(finding.chart));
        pending.push([holder, finding.chart]);
      }
      if (finding.table) article.append(dataTable(finding.table));
      if (finding.caveat) article.append(el('p', 'microcopy finding-caveat', finding.caveat));
      list.append(article);
    });
    const allTests = el('details', 'panel all-tests');
    allTests.append(el('summary', '', `${data.all_tests.title} (${data.all_tests.rows.length})`), el('p', 'microcopy', data.all_tests.note), dataTable({ headers: data.all_tests.headers, rows: data.all_tests.rows }));
    const closing = el('div', 'calibration-grid');
    const limits = el('article', 'panel'); limits.append(el('h2', '', data.limits_title)); const limitList = el('ul', 'clean-list'); data.limits.forEach((text) => limitList.append(el('li', '', text))); limits.append(limitList);
    const next = el('article', 'panel'); next.append(el('h2', '', data.next_title)); const nextList = el('ul', 'clean-list'); data.next.forEach((text) => nextList.append(el('li', '', text))); next.append(nextList);
    closing.append(limits, next);
    const sources = el('article', 'panel findings-sources'); sources.append(el('h2', '', 'Data and sources'), el('p', 'microcopy', data.plan_note));
    const sourceList = el('ul', 'clean-list');
    data.sources.forEach((source) => { const li = el('li'), link = el('a', 'source-link', source.label); link.href = source.url; link.target = '_blank'; link.rel = 'noopener noreferrer'; li.append(link); sourceList.append(li); });
    sources.append(sourceList);
    root.replaceChildren(heading, safety, tiles, el('p', 'microcopy tiles-note', data.tiles_note), list, allTests, closing, sources);
    pending.forEach(([holder, chart]) => draw(holder, chart));
  }

  window.MycoCharts = { draw, lines, renderFindings, colors: COLORS };
})();
