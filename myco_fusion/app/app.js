'use strict';

const $ = (selector) => document.querySelector(selector);
const state = { experiment: null, selected: 0, example: 'typical', validation: null, manifest: null, importDemo: false, calibration: null, phone: null, phoneDemo: false, roi: { files: [], bitmap: null, box: null, scale: 1 } };
const viewNames = { findings: 'FINDINGS', bench: 'SPECTRAL BENCH', phone: 'PHONE TEST', calibration: 'CALIBRATION LAB', validate: 'VALIDATE MEASUREMENTS', capture: 'CAPTURE RECORD' };

function setView(name) {
  if (!viewNames[name]) name = 'findings';
  document.querySelectorAll('.view').forEach((el) => { el.hidden = el.id !== `view-${name}`; });
  document.querySelectorAll('[data-view]').forEach((el) => {
    const active = el.dataset.view === name;
    el.classList.toggle('active', active);
    if (active) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current');
  });
  $('#breadcrumb').textContent = viewNames[name];
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
  window.scrollTo({ top: 0 });
  // Charts drawn while a tab was hidden have no width; redraw the bench when it appears.
  if (name === 'bench' && state.experiment) renderExperiment(state.experiment);
}
document.querySelectorAll('[data-view]').forEach((el) => el.addEventListener('click', () => setView(el.dataset.view)));
document.querySelectorAll('[data-go]').forEach((el) => el.addEventListener('click', () => { setView(el.dataset.go); window.scrollTo({ top: 0 }); }));
window.addEventListener('hashchange', () => setView(location.hash.slice(1)));
setView(location.hash.slice(1));

function message(id, text, type = '') { const el = $(id); el.textContent = text; el.className = `message ${type}`; }
function fmt(value, digits = 2) { return Number.isFinite(value) ? Number(value).toFixed(digits) : '—'; }
function node(tag, cls, text) { const el = document.createElement(tag); if (cls) el.className = cls; if (text !== undefined) el.textContent = text; return el; }
function listItems(selector, values) { const el = $(selector); el.replaceChildren(...(values || []).map((text) => node('li', '', String(text)))); }
async function api(path, body) {
  const encoded = body === undefined ? undefined : JSON.stringify(body);
  if (encoded && new TextEncoder().encode(encoded).length > 2000000) throw new Error('This request exceeds the 2 MB local-app limit. Use a smaller dataset or the command-line tools.');
  const response = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: encoded });
  let result;
  try { result = await response.json(); } catch { throw new Error(`The local server returned an unreadable response (${response.status}).`); }
  if (!response.ok) throw new Error(result.error || result.message || `Request failed (${response.status}).`);
  return result;
}
function download(value, filename) {
  const blob = new Blob([JSON.stringify(value, null, 2) + '\n'], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function chart(container, bands, first, second, title, compact = false, labels = ['Measured', 'Estimated']) {
  if (!Array.isArray(bands) || !bands.length || !Array.isArray(first) || !Array.isArray(second)) return;
  window.MycoCharts.draw($(container), { type: 'lines', title, xLabel: 'Wavelength (nm)', digits: 3, yMin: Math.min(0, ...first, ...second), yMax: compact ? undefined : Math.max(1, ...first, ...second), hideLegend: true, x: bands,
    series: [{ label: labels[0], values: first, color: '#1f7a54' }, { label: labels[1], values: second, color: '#c2622a', dashed: true }] });
}

function percent(value) { return Number.isFinite(value) ? `${Math.round(value * 100)}%` : '—'; }
function drawExample() {
  const experiment = state.experiment, scenario = experiment.scenarios[state.selected];
  const hasWorst = Boolean(scenario.examples.worst);
  if (!hasWorst) state.example = 'typical';
  $('#example-worst').disabled = !hasWorst;
  ['typical', 'worst'].forEach((name) => { const button = $(`#example-${name}`); const on = state.example === name; button.classList.toggle('active', on); button.setAttribute('aria-pressed', String(on)); });
  const example = scenario.examples[state.example];
  const detail = Number.isFinite(example.mae) ? ` · angle ${fmt(example.sam_deg, 1)}° · MAE ${fmt(example.mae, 3)}` : '';
  $('#example-name').textContent = `${example.name}${detail}`;
  chart('#spectrum-chart', experiment.bands_nm, example.reference, example.estimate, `${scenario.label}: measured and estimated reflectance`);
}
function selectScenario(index) {
  const experiment = state.experiment, scenario = experiment.scenarios[index];
  if (!scenario) return;
  state.selected = index;
  document.querySelectorAll('.scenario-card').forEach((el, i) => { el.classList.toggle('selected', i === index); el.setAttribute('aria-pressed', String(i === index)); });
  $('#spectrum-title').textContent = scenario.label;
  $('#scenario-description').textContent = scenario.description || '';
  drawExample();
  const rank = Number(scenario.independent), total = Number(scenario.channels);
  $('#rank-number').textContent = Number.isFinite(rank) ? rank : '—';
  $('.rank-total').textContent = `/ ${total}`;
  $('#rank-bars').replaceChildren(...Array.from({ length: total }, (_, i) => node('i', i < rank ? 'filled' : '')));
  $('#rank-state').textContent = rank < total ? `${total} numbers recorded, ${rank} independent` : `All ${total} numbers are independent`;
  $('#rank-explanation').textContent = experiment.source === 'measured' ? 'Counts directions of the camera-and-light system above 2% of the strongest one. Recording more numbers only helps if they measure something new.' : 'Counts independent directions within the model’s four coefficients, assuming the camera and lights are known exactly.';
  const stats = $('#scenario-stats'); stats.replaceChildren();
  appendDiagnostic(stats, 'Median MAE', fmt(scenario.median_mae, 4));
  if (Number.isFinite(scenario.p90_mae)) appendDiagnostic(stats, '90th percentile MAE', fmt(scenario.p90_mae, 3));
  if (Number.isFinite(scenario.worst_mae)) appendDiagnostic(stats, 'Worst MAE', fmt(scenario.worst_mae, 3));
  if (Number.isFinite(scenario.share_within_targets)) appendDiagnostic(stats, 'Meeting both draft targets', percent(scenario.share_within_targets));
}
function renderExperiment(result) {
  state.experiment = result;
  const measured = result.source === 'measured';
  $('#bench-badge').textContent = measured ? 'SIMULATED CAMERA' : 'BEST CASE ONLY';
  $('#bench-notice').textContent = measured ? 'Reflectance spectra are measured. The camera and lights are illustrative curves and the RGB values are simulated, so these results do not validate a phone.' : 'These spectra are generated by the same four-number model the solver assumes. This checks the code and shows a best case. It is not a performance estimate.';
  $('#scenarios').replaceChildren(...result.scenarios.map((scenario, index) => {
    const button = node('button', 'scenario-card'); button.type = 'button'; button.setAttribute('aria-pressed', 'false');
    button.append(node('span', 'scenario-number', `SCENARIO ${String(index + 1).padStart(2, '0')}`), node('span', 'scenario-name', scenario.label));
    const val = node('span', 'scenario-value', fmt(scenario.median_sam_deg)); val.append(node('small', '', '°')); button.append(val);
    const bottom = node('span', 'scenario-bottom'); bottom.append(node('span', '', 'MEDIAN ANGLE'), node('span', '', Number.isFinite(scenario.share_within_targets) ? `${percent(scenario.share_within_targets)} MEET TARGETS` : `MAE ${fmt(scenario.median_mae, 3)}`)); button.append(bottom);
    button.setAttribute('aria-label', `${scenario.label}. Median spectral angle ${fmt(scenario.median_sam_deg)} degrees. ${scenario.independent} independent measurements of ${scenario.channels} recorded.`);
    button.addEventListener('click', () => { state.example = 'typical'; selectScenario(index); }); return button;
  }));
  selectScenario(Math.min(state.selected, result.scenarios.length - 1));
  const metamer = result.metamer;
  if (metamer) {
    const real = Boolean(metamer.rgb_a);
    $('#metamer-title').replaceChildren(document.createTextNode(real ? 'Two real objects.' : 'Different spectra.'), node('br'), document.createTextNode(real ? 'Nearly the same RGB.' : 'The same RGB.'));
    $('#metamer-text').textContent = metamer.description || '';
    $('#metamer-a').textContent = metamer.first_name; $('#metamer-b').textContent = metamer.second_name;
    if (real) {
      const largest = Math.max(...metamer.rgb_a.map((value, i) => Math.abs(value - metamer.rgb_b[i])));
      $('#metamer-rgb').textContent = fmt(largest, 3); $('#metamer-rgb-label').textContent = 'Largest RGB difference';
      $('#metamer-rmse').textContent = fmt(metamer.mae, 3); $('#metamer-rmse-label').textContent = 'Mean reflectance difference';
    } else {
      $('#metamer-rgb').textContent = Number.isFinite(metamer.rgb_difference) ? metamer.rgb_difference.toExponential(2) : '—'; $('#metamer-rgb-label').textContent = 'RGB difference';
      $('#metamer-rmse').textContent = fmt(metamer.spectral_rmse, 3); $('#metamer-rmse-label').textContent = 'Spectral RMSE';
    }
    chart('#metamer-chart', metamer.bands_nm || result.bands_nm, metamer.first, metamer.second, 'Two spectra with nearly the same RGB', true, [metamer.first_name, metamer.second_name]);
  }
  listItems('#experiment-notes', result.notes);
  $('#export-experiment').disabled = false;
}
['typical', 'worst'].forEach((name) => $(`#example-${name}`).addEventListener('click', () => { state.example = name; drawExample(); }));
$('#bench-source').addEventListener('change', () => {
  $('#noise').value = $('#bench-source').value === 'measured' ? '0.005' : '0.003';
  state.example = 'typical'; state.selected = 0; $('#experiment-form').requestSubmit();
});
$('#experiment-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('#run-experiment'); button.disabled = true; message('#bench-message', 'Running the numerical experiment…', 'loading');
  try {
    const result = await api('/api/bench', { seed: Number($('#seed').value), noise: Number($('#noise').value), source: $('#bench-source').value });
    renderExperiment(result); message('#bench-message', 'Experiment complete. Select a scenario to inspect one held-out object.');
  } catch (error) { message('#bench-message', error.message, 'error'); }
  finally { button.disabled = false; }
});
$('#export-experiment').addEventListener('click', () => { if (state.experiment) download(state.experiment, 'myco-fusion-bench.json'); });

function clearValidationResult() { $('#validation-results').hidden = true; state.validation = null; }
function markInputChanged() { state.importDemo = false; $('#import-kind').textContent = 'Imported values · source provenance must be confirmed by the researcher.'; clearValidationResult(); }
function uncertaintyHelp(mode) {
  if (mode === 'session_cluster') return 'Uses one median per session. Every specimen must belong to one session; crossed designs are rejected. It takes 6 sessions to reach 95%; with fewer, the true lower level is shown.';
  if (mode === 'specimen') return 'Treats specimens as independent. If specimens share sessions, this interval can be too narrow.';
  return 'Uses sessions when they are supplied and nested, otherwise specimens. Intervals state the confidence they actually reach.';
}
$('#validation-uncertainty').addEventListener('change', () => { clearValidationResult(); $('#uncertainty-help').textContent = uncertaintyHelp($('#validation-uncertainty').value); message('#validate-message', 'Uncertainty method changed. Evaluate again to update the report.'); });
$('#csv-input').addEventListener('input', markInputChanged);
$('#csv-file').addEventListener('change', async (event) => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 2000000) { message('#validate-message', 'Please choose a CSV smaller than 2 MB, or use the command-line validator.', 'error'); return; }
  try { $('#csv-input').value = await file.text(); $('#file-name').textContent = file.name; markInputChanged(); message('#validate-message', 'CSV loaded. Select Evaluate spectra to check it.'); }
  catch { message('#validate-message', 'The selected file could not be read.', 'error'); }
});
$('#load-demo').addEventListener('click', async () => {
  const button = $('#load-demo'); button.disabled = true;
  try { const result = await api('/api/example'); $('#csv-input').value = result.csv; $('#file-name').textContent = 'Simulated example · measured reference spectra, simulated camera'; state.importDemo = true; state.validation = null; $('#validation-results').hidden = true; $('#import-kind').textContent = 'SIMULATED EXAMPLE · Reference spectra are measured; estimates come from a simulated camera.'; message('#validate-message', 'Simulated example loaded. Select Evaluate spectra to inspect the report.'); }
  catch (error) { message('#validate-message', error.message, 'error'); }
  finally { button.disabled = false; }
});
function interval(value, digits = 3) { return value ? `${Math.round(value.level * 1000) / 10}% interval [${fmt(value.low, digits)}, ${fmt(value.high, digits)}]` : 'No interval'; }
function signed(value, digits = 4) { return Number.isFinite(value) ? `${value > 0 ? '+' : value < 0 ? '−' : ''}${fmt(Math.abs(value), digits)}` : 'Unavailable'; }
function plainTableRow(values) { const tr = node('tr'); values.forEach((value) => tr.append(node('td', '', value))); return tr; }
function emptyTableRow(columns, text) { const tr = node('tr'), td = node('td', 'empty-table', text); td.colSpan = columns; tr.append(td); return tr; }
function renderValidationDetails(result) {
  const counts = result.counts || {}, clustered = result.uncertainty_method === 'session_cluster';
  $('#validation-specimen-count').textContent = counts.n_specimens ?? '—';
  $('#validation-session-count').textContent = counts.n_sessions ?? '—';
  $('#validation-session-count-label').textContent = counts.n_sessions == null ? 'Session IDs not supplied' : 'Unique session IDs';
  $('#validation-observation-count').textContent = counts.n_observations ?? result.rows?.length ?? '—';
  $('#validation-uncertainty-kind').textContent = clustered ? 'SESSION INTERVAL' : 'SPECIMEN INTERVAL';
  $('#validation-uncertainty-summary').textContent = `${counts.n_specimen_regime_units ?? result.specimen_summaries?.length ?? '—'} specimen/regime summaries. Point estimates give each specimen equal weight within a regime. ${clustered ? 'Intervals use one median per session, so specimens must be nested within sessions.' : 'Intervals treat specimens as independent; shared session and site effects are not modeled.'}`;
  listItems('#validation-interval-notes', result.interval_notes || []);
  $('#validation-tail-table').replaceChildren(...(result.regimes || []).map((row) => plainTableRow([row.regime, fmt(row.worst_specimen_mae, 4), fmt(row.p90_specimen_mae, 4)])));
  const specimens = [...(result.specimen_summaries || [])].sort((a, b) => b.mean_mae - a.mean_mae);
  $('#validation-specimen-summary').textContent = `Inspect specimen summaries · ${specimens.length} specimen/regime units`;
  $('#validation-specimen-table').replaceChildren(...(specimens.length ? specimens.map((row) => plainTableRow([row.specimen_id, row.regime, row.session_ids?.join(', ') || 'Not supplied', row.n_observations, `${fmt(row.mean_sam_deg, 3)}°`, fmt(row.mean_mae, 4)])) : [emptyTableRow(6, 'No specimen summaries were returned.')]));
  const sessions = [...(result.session_summaries || [])].sort((a, b) => b.median_mae - a.median_mae);
  $('#validation-session-summary').textContent = `Inspect session summaries · ${sessions.length} session/regime units`;
  $('#validation-session-note').textContent = sessions.length ? 'Sorted by median MAE, largest first. Each row summarizes specimen means within a session and regime; these descriptive rows are not additional independent specimens.' : 'No session IDs were supplied, so acquisition-session summaries are unavailable.';
  $('#validation-session-table').replaceChildren(...(sessions.length ? sessions.map((row) => plainTableRow([row.session_id, row.regime, row.n_specimens, row.n_observations, `${fmt(row.median_sam_deg, 3)}°`, fmt(row.median_mae, 4)])) : [emptyTableRow(6, 'Add complete session_id values to inspect acquisition sessions.')]));
  const paired = result.paired_r1_r3 || {}, pairs = paired.rows || [], dl = $('#validation-paired-summary'); dl.replaceChildren();
  appendDiagnostic(dl, 'Matched physical specimens', paired.n_specimens ?? 0);
  appendDiagnostic(dl, 'Median paired Δ MAE', signed(paired.median_delta_mae));
  if (paired.delta_mae_interval) appendDiagnostic(dl, 'Δ MAE interval', interval(paired.delta_mae_interval, 4));
  appendDiagnostic(dl, 'Median paired Δ SAM', Number.isFinite(paired.median_delta_sam_deg) ? `${signed(paired.median_delta_sam_deg, 3)}°` : 'Unavailable');
  $('#validation-paired-note').textContent = pairs.length ? 'Each delta compares one physical specimen’s mean error in R3 with its mean error in R1. The displayed median is taken over those matched deltas.' : 'No physical specimen has observations in both R1 and R3. A paired comparison is unavailable.';
  const r1Only = paired.r1_only_specimens || [], r3Only = paired.r3_only_specimens || [];
  $('#validation-unmatched-note').textContent = `Unmatched specimens are excluded from paired deltas. R1 only (${r1Only.length}): ${r1Only.join(', ') || 'none'}. R3 only (${r3Only.length}): ${r3Only.join(', ') || 'none'}.`;
  $('#validation-paired-detail-summary').textContent = `Inspect matched R1 / R3 specimens · ${pairs.length} pairs`;
  $('#validation-paired-table').replaceChildren(...(pairs.length ? pairs.map((row) => plainTableRow([row.specimen_id, fmt(row.r1_mean_mae, 4), fmt(row.r3_mean_mae, 4), signed(row.delta_mae_r3_minus_r1), `${fmt(row.r1_mean_sam_deg, 3)}°`, `${fmt(row.r3_mean_sam_deg, 3)}°`, `${signed(row.delta_sam_deg_r3_minus_r1, 3)}°`])) : [emptyTableRow(7, 'No matched R1 / R3 specimens.')]));
}
$('#validate-csv').addEventListener('click', async () => {
  const csv = $('#csv-input').value, uncertainty = $('#validation-uncertainty').value, importedDemo = state.importDemo;
  if (!csv.trim()) { message('#validate-message', 'Choose a CSV, paste data, or load the simulated example first.', 'error'); return; }
  const button = $('#validate-csv'); button.disabled = true; clearValidationResult(); message('#validate-message', 'Checking the paired spectra and computing intervals…', 'loading');
  try {
    const result = await api('/api/validate', { csv, uncertainty });
    if ($('#csv-input').value !== csv || $('#validation-uncertainty').value !== uncertainty) throw new Error('The input or uncertainty method changed during evaluation. Evaluate again to update the report.');
    const synthetic = result.data_kind === 'synthetic' || importedDemo;
    if (importedDemo) { result.ui_provenance = 'synthetic_example'; result.notes = ['This report uses the bundled example: measured reference spectra with estimates from a simulated camera.', ...(result.notes || [])]; }
    $('#validation-kind').textContent = synthetic ? 'SIMULATED INPUT' : 'SOURCE UNVERIFIED';
    $('#validation-provenance-text').textContent = synthetic ? 'This report uses simulated camera values. It exercises the analysis and is not instrument or field evidence.' : 'These are user-supplied values. The app has not verified their acquisition, calibration, or provenance.';
    state.validation = result;
    $('#validation-table').replaceChildren(...regimeRows(result.regimes));
    renderValidationDetails(result);
    listItems('#validation-notes', (result.notes || []).filter((note) => !(result.interval_notes || []).includes(note))); $('#input-hash').textContent = result.input_sha256 ? `Input SHA-256: ${result.input_sha256}` : '';
    $('#validation-results').hidden = false;
    message('#validate-message', synthetic ? 'Simulated example evaluated. No phone was tested.' : 'Evaluation complete. Confirm source provenance before interpreting the results.');
  } catch (error) { message('#validate-message', error.message, 'error'); }
  finally { button.disabled = false; }
});
$('#export-validation').addEventListener('click', () => { if (state.validation) download(state.validation, 'myco-fusion-spectrum-validation.json'); });

$('#capture-form').addEventListener('input', () => { state.manifest = null; $('#manifest-result').hidden = true; });
$('#capture-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = $('#create-manifest'); button.disabled = true;
  try {
    const fields = Object.fromEntries(new FormData(event.currentTarget).entries()); const result = await api('/api/manifest', fields);
    state.manifest = result.manifest; $('#manifest-preview').textContent = JSON.stringify(result.manifest, null, 2); listItems('#manifest-warnings', result.warnings); $('#manifest-result').hidden = false;
    message('#capture-message', 'Draft created. Download it to retain the record; this form does not save automatically.');
  } catch (error) { message('#capture-message', error.message, 'error'); }
  finally { button.disabled = false; }
});
$('#export-manifest').addEventListener('click', () => { if (state.manifest) download(state.manifest, 'myco-fusion-capture-draft.json'); });

$('#experiment-form').requestSubmit();

// Calibration operates on supplied linear values; it never decodes a camera file.
function clearCalibrationResult() { state.calibration = null; $('#calibration-results').hidden = true; }
function datasetInputChanged() {
  clearCalibrationResult(); $('#calibration-input-kind').textContent = 'Input edited · acquisition and calibration provenance must be independently confirmed.';
  try { const value = JSON.parse($('#calibration-input').value).uncertainty; if (['auto', 'specimen', 'session_cluster'].includes(value)) $('#calibration-uncertainty').value = value; } catch { /* Keep the selected method while JSON is incomplete. */ }
  $('#calibration-uncertainty-help').textContent = uncertaintyHelp($('#calibration-uncertainty').value);
}
$('#calibration-input').addEventListener('input', datasetInputChanged);
$('#calibration-uncertainty').addEventListener('change', () => { clearCalibrationResult(); $('#calibration-uncertainty-help').textContent = uncertaintyHelp($('#calibration-uncertainty').value); message('#calibration-message', 'Uncertainty method changed. Fit & evaluate again to update the report.'); });
$('#calibration-file').addEventListener('change', async (event) => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 2000000) { message('#calibration-message', 'Choose a JSON dataset smaller than 2 MB, or use the command-line calibration tools.', 'error'); return; }
  try { $('#calibration-input').value = await file.text(); $('#calibration-file-name').textContent = file.name; datasetInputChanged(); message('#calibration-message', 'Dataset loaded. Select Fit & evaluate to validate the splits and fit the model.'); }
  catch { message('#calibration-message', 'The selected JSON file could not be read.', 'error'); }
});
$('#load-calibration-demo').addEventListener('click', async () => {
  const button = $('#load-calibration-demo'); button.disabled = true;
  try {
    const result = await api('/api/calibration-example');
    $('#calibration-input').value = JSON.stringify(result, null, 2); $('#calibration-file-name').textContent = 'Simulated example · measured reference spectra, simulated camera';
    $('#calibration-uncertainty').value = result.uncertainty || 'auto'; $('#calibration-uncertainty-help').textContent = uncertaintyHelp($('#calibration-uncertainty').value);
    clearCalibrationResult(); $('#calibration-input-kind').textContent = 'SIMULATED CAMERA · Reference spectra are measured. Separate training, validation and test specimens and sessions.';
    message('#calibration-message', 'Simulated example loaded. Select Fit & evaluate to run the calibration.');
  } catch (error) { message('#calibration-message', error.message, 'error'); }
  finally { button.disabled = false; }
});
function parseCalibrationInput() {
  const text = $('#calibration-input').value.trim();
  if (!text) throw new Error('Choose a dataset, paste JSON, or load the simulated example first.');
  let dataset;
  try { dataset = JSON.parse(text); } catch { throw new Error('The dataset is not valid JSON. Check quotes, commas and brackets, or load the simulated example to see the format.'); }
  if (!dataset || typeof dataset !== 'object' || Array.isArray(dataset)) throw new Error('The dataset must be a JSON object containing observations and calibration metadata.');
  dataset.uncertainty = $('#calibration-uncertainty').value;
  return dataset;
}
$('#export-calibration-input').addEventListener('click', () => {
  try { download(parseCalibrationInput(), 'myco-fusion-calibration-input.json'); }
  catch (error) { message('#calibration-message', error.message, 'error'); }
});
function appendDiagnostic(list, name, value) { list.append(node('dt', '', name), node('dd', '', value)); }
function regimeRows(regimes) {
  return (regimes || []).map((row) => {
    const tr = node('tr'), counts = node('td', '', `${row.n_specimens} / ${row.n_observations}`);
    counts.append(node('small', '', row.n_sessions == null ? 'Sessions not supplied' : `${row.n_sessions} session${row.n_sessions === 1 ? '' : 's'}`));
    counts.append(node('small', '', `Interval over ${row.interval_units} ${row.interval_unit}${row.interval_units === 1 ? '' : 's'}`));
    tr.append(node('td', '', row.regime), counts);
    const sam = node('td', '', `${fmt(row.median_sam_deg, 2)}°`); sam.append(node('small', '', interval(row.sam_interval, 2))); if (row.interval_note) sam.append(node('small', 'interval-unavailable', row.interval_note)); tr.append(sam);
    const mae = node('td', '', fmt(row.median_mae, 4)); mae.append(node('small', '', interval(row.mae_interval, 4))); tr.append(mae);
    const status = node('td', '', `${row.n_within_targets} of ${row.n_specimens}`); status.append(node('small', '', `${percent(row.share_within_targets)}${row.median_within_targets ? ' · median is within targets' : ' · median is outside targets'}`)); tr.append(status); return tr;
  });
}
function renderCalibration(result) {
  state.calibration = result;
  const synthetic = result.data_kind === 'synthetic';
  $('#calibration-kind').textContent = synthetic ? 'SIMULATED INPUT' : 'SOURCE UNVERIFIED';
  $('#calibration-provenance').textContent = synthetic ? 'These results use a simulated camera on measured reflectance. They are not a phone calibration or evidence of physical performance.' : 'These results use user-supplied RGB and reference spectra. The app has not authenticated instrument calibration, linearity, acquisition or specimen identity.';
  $('#calibration-mae').textContent = fmt(result.test_mean_specimen_mae, 5);
  $('#calibration-baseline').textContent = fmt(result.median_baseline_mean_specimen_mae, 5);
  $('#calibration-median-baseline').textContent = `Training-mean baseline: ${fmt(result.baseline_mean_specimen_mae, 5)}`;
  const contrast = result.model_vs_median_baseline;
  $('#calibration-difference').textContent = signed(contrast.median_difference, 4);
  $('#calibration-difference-label').textContent = `${contrast.median_difference <= 0 ? 'Model error is lower at the median. ' : 'Model error is higher at the median. '}${interval(contrast.interval, 4)}`;
  $('#calibration-counts').replaceChildren(...['train', 'validation', 'test'].map((split) => {
    const counts = result.counts[split], tr = node('tr');
    tr.append(node('td', '', split[0].toUpperCase() + split.slice(1)), node('td', '', counts.specimens), node('td', '', counts.sessions), node('td', '', counts.observations)); return tr;
  }));
  const diagnostics = result.diagnostics, dl = $('#calibration-diagnostics'); dl.replaceChildren();
  appendDiagnostic(dl, 'Affine design rank', `${diagnostics.design_rank} / 4`);
  appendDiagnostic(dl, 'Design condition number', fmt(diagnostics.design_condition_number, 2));
  appendDiagnostic(dl, 'Test RGB outside training range', String(diagnostics.test_rgb_outside_training_range));
  appendDiagnostic(dl, 'Estimated bands outside [0, 1]', String(diagnostics.test_estimated_bands_outside_0_1));
  appendDiagnostic(dl, 'Shared sessions across splits', String((diagnostics.session_overlap || []).length));
  const first = result.evaluation.rows && result.evaluation.rows[0];
  if (first) {
    $('#calibration-spectrum-title').textContent = `${first.specimen_id} · ${first.regime}`;
    chart('#calibration-chart', result.model.bands_nm, first.reference, first.estimate, 'First held-out observation: reference and estimated spectra', false, ['Reference', 'Estimated']);
  } else { $('#calibration-chart').replaceChildren(node('p', 'microcopy', 'No observation curve was returned.')); }
  $('#calibration-regimes').replaceChildren(...regimeRows(result.evaluation.regimes));
  const clustered = result.evaluation.uncertainty_method === 'session_cluster';
  $('#calibration-uncertainty-kind').textContent = clustered ? 'SESSION INTERVAL' : 'SPECIMEN INTERVAL';
  $('#calibration-interval-summary').textContent = clustered ? 'Test intervals use one median per acquisition session. A regime with fewer than two test sessions has no session-based interval.' : 'Test intervals treat physical specimens as independent. They do not cover shared acquisition-session or site effects, even when sessions are held out between splits.';
  listItems('#calibration-interval-notes', result.evaluation.interval_notes || []);
  $('#calibration-alpha').textContent = result.tuning.length ? `Selected alpha: ${result.model.alpha}. Selection uses validation mean specimen MAE only; the final model is fitted on the training split only.` : `Fixed alpha: ${result.model.alpha}. No validation split was supplied; test data did not choose this value.`;
  $('#calibration-tuning').replaceChildren(...result.tuning.map((candidate) => {
    const tr = node('tr'); tr.append(node('td', '', candidate.alpha), node('td', '', fmt(candidate.validation_mean_specimen_mae, 6)), node('td', '', candidate.alpha === result.model.alpha ? 'Selected' : '')); return tr;
  }));
  listItems('#calibration-notes', [...(result.notes || []), ...(result.evaluation.notes || []).filter((note) => !(result.evaluation.interval_notes || []).includes(note))]);
  $('#calibration-model-hash').textContent = `Model SHA-256: ${result.model.model_sha256}`;
  $('#calibration-input-hash').textContent = `Input SHA-256: ${result.input_sha256}`;
  $('#calibration-results').hidden = false;
}
$('#fit-calibration').addEventListener('click', async () => {
  const button = $('#fit-calibration'); clearCalibrationResult();
  const submittedText = $('#calibration-input').value, submittedUncertainty = $('#calibration-uncertainty').value;
  let dataset;
  try { dataset = parseCalibrationInput(); } catch (error) { message('#calibration-message', error.message, 'error'); return; }
  button.disabled = true; message('#calibration-message', 'Checking specimen/session splits, fitting calibration, and evaluating held-out observations…', 'loading');
  try {
    const result = await api('/api/calibrate', { dataset });
    if ($('#calibration-input').value !== submittedText || $('#calibration-uncertainty').value !== submittedUncertainty) throw new Error('The dataset or uncertainty method changed while calibration was running. Fit again to evaluate the current input.');
    renderCalibration(result);
    message('#calibration-message', result.data_kind === 'synthetic' ? 'Simulated calibration complete. This shows the workflow; no phone was tested.' : 'Calibration evaluation complete. Review provenance, baseline performance and regime-specific results before interpreting the fit.');
  } catch (error) { message('#calibration-message', error.message, 'error'); }
  finally { button.disabled = false; }
});
$('#export-calibration-model').addEventListener('click', () => { if (state.calibration) download(state.calibration.model, 'myco-fusion-calibration-model.json'); });
$('#export-calibration-report').addEventListener('click', () => { if (state.calibration) download(state.calibration, 'myco-fusion-calibration-report.json'); });
function downloadText(value, filename, mime) {
  const url = URL.createObjectURL(new Blob([value], { type: mime })); const link = node('a');
  link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('#export-calibration-csv').addEventListener('click', () => { if (state.calibration) downloadText(state.calibration.prediction_csv, 'myco-fusion-held-out-predictions.csv', 'text/csv;charset=utf-8'); });


// Findings load first because they are the landing page.
(async function loadFindings() {
  const root = $('#findings-root');
  try { window.MycoCharts.renderFindings(root, await api('/api/findings')); }
  catch (error) { root.replaceChildren(node('p', 'message error', `The findings could not be loaded: ${error.message}`)); }
})();

// Phone test: everything here runs in the browser or on the local server.
const LIGHT_COLORS = [['Red', '#ff0000'], ['Green', '#00ff00'], ['Blue', '#0000ff'], ['White', '#ffffff']];
let lightIndex = 0;
function paintLight() {
  const [name, color] = LIGHT_COLORS[lightIndex];
  $('#light-panel').style.background = color; $('#light-name').textContent = name;
}
async function closeLightPanel() {
  // Leave full screen first: hiding a full-screen element can leave the page unclickable.
  if (document.fullscreenElement) { try { await document.exitFullscreen(); } catch { /* Hide it anyway. */ } }
  $('#light-panel').hidden = true;
}
$('#open-light-panel').addEventListener('click', async () => {
  const panel = $('#light-panel'); lightIndex = 0; paintLight(); panel.hidden = false; panel.focus();
  if (panel.requestFullscreen) { try { await panel.requestFullscreen(); } catch { /* A window-sized panel still works. */ } }
});
$('#light-panel').addEventListener('click', () => { lightIndex = (lightIndex + 1) % LIGHT_COLORS.length; paintLight(); });
$('#light-panel').addEventListener('keydown', (event) => {
  if (event.key === 'Escape') closeLightPanel();
  if (event.key === 'ArrowRight' || event.key === ' ') { lightIndex = (lightIndex + 1) % LIGHT_COLORS.length; paintLight(); event.preventDefault(); }
  if (event.key === 'ArrowLeft') { lightIndex = (lightIndex + LIGHT_COLORS.length - 1) % LIGHT_COLORS.length; paintLight(); event.preventDefault(); }
});
document.addEventListener('fullscreenchange', () => {
  const panel = $('#light-panel');
  if (!document.fullscreenElement) panel.hidden = true;
  // Esc can arrive before full screen has opened; never stay full screen on a hidden panel.
  else if (panel.hidden) document.exitFullscreen().catch(() => {});
});

// Labels come from names like mug_s1_flash_r2.jpg. Anything else is listed as skipped, never guessed.
function photoLabels(name) {
  const match = name.replace(/\.[^.]+$/, '').match(/^(.+?)_(s\d+)_(.+)_r(\d+)$/i);
  if (!match) return null;
  return { object: match[1], session: match[2].toLowerCase(), light: match[3].toLowerCase(), repeat: match[4] };
}
function srgbToLinear(value) { const v = value / 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }
const MAX_PREVIEW = 1600;
async function loadBitmap(file) {
  const probe = await createImageBitmap(file);
  const ratio = Math.min(1, MAX_PREVIEW / Math.max(probe.width, probe.height));
  if (ratio === 1) return probe;
  const scaled = await createImageBitmap(file, { resizeWidth: Math.round(probe.width * ratio), resizeHeight: Math.round(probe.height * ratio), resizeQuality: 'medium' });
  probe.close();
  return scaled;
}
function drawRoi() {
  const { bitmap, box } = state.roi, canvas = $('#roi-canvas');
  if (!bitmap) return;
  const context = canvas.getContext('2d');
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  if (box) {
    context.lineWidth = Math.max(2, canvas.width / 400); context.strokeStyle = '#ffffff'; context.strokeRect(box.x, box.y, box.w, box.h);
    context.setLineDash([8, 8]); context.strokeStyle = '#000000'; context.strokeRect(box.x, box.y, box.w, box.h); context.setLineDash([]);
  }
}
$('#roi-files').addEventListener('change', async (event) => {
  const files = [...event.target.files];
  state.roi.files = files; state.roi.box = null; $('#roi-measure').disabled = true;
  if (!files.length) { $('#roi-status').textContent = 'No photos selected'; $('#roi-stage').hidden = true; $('#roi-actions').hidden = true; return; }
  $('#roi-status').textContent = `Opening ${files[0].name}…`;
  try {
    state.roi.bitmap = await loadBitmap(files[0]);
    const canvas = $('#roi-canvas'); canvas.width = state.roi.bitmap.width; canvas.height = state.roi.bitmap.height;
    $('#roi-stage').hidden = false; $('#roi-actions').hidden = false; drawRoi();
    $('#roi-status').textContent = `${files.length} photo${files.length === 1 ? '' : 's'} selected. Showing ${files[0].name}.`;
    $('#roi-box-text').textContent = 'Drag on the photo to draw a box. The same box is used for every selected photo.';
  } catch {
    $('#roi-status').textContent = `This browser could not open ${files[0].name}. Safari can open HEIC photos; other browsers need JPEG or PNG.`;
    $('#roi-stage').hidden = true; $('#roi-actions').hidden = true;
  }
});
function canvasPoint(event) {
  const canvas = $('#roi-canvas'), rect = canvas.getBoundingClientRect();
  return { x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * canvas.width / rect.width)), y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * canvas.height / rect.height)) };
}
let roiStart = null;
$('#roi-canvas').addEventListener('pointerdown', (event) => { roiStart = canvasPoint(event); $('#roi-canvas').setPointerCapture(event.pointerId); });
$('#roi-canvas').addEventListener('pointermove', (event) => {
  if (!roiStart) return;
  const point = canvasPoint(event);
  state.roi.box = { x: Math.min(roiStart.x, point.x), y: Math.min(roiStart.y, point.y), w: Math.abs(point.x - roiStart.x), h: Math.abs(point.y - roiStart.y) };
  drawRoi();
});
$('#roi-canvas').addEventListener('pointerup', () => {
  roiStart = null;
  const box = state.roi.box, ok = box && box.w >= 4 && box.h >= 4;
  $('#roi-measure').disabled = !ok;
  if (ok) $('#roi-box-text').textContent = `Box set: ${Math.round(box.w)} × ${Math.round(box.h)} pixels.`;
});
$('#roi-measure').addEventListener('click', async () => {
  const button = $('#roi-measure'), { files, box, bitmap } = state.roi;
  if (!box || !bitmap) return;
  button.disabled = true;
  const relative = { x: box.x / bitmap.width, y: box.y / bitmap.height, w: box.w / bitmap.width, h: box.h / bitmap.height };
  const added = [], problems = [];
  let clippedPhotos = 0;
  for (const [index, file] of files.entries()) {
    $('#roi-status').textContent = `Measuring ${index + 1} of ${files.length}…`;
    const labels = photoLabels(file.name);
    if (!labels) { problems.push(`${file.name}: name should look like mug_s1_flash_r2.jpg`); continue; }
    try {
      const image = await loadBitmap(file);
      const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext('2d', { willReadFrequently: true }); context.drawImage(image, 0, 0);
      const x = Math.round(relative.x * image.width), y = Math.round(relative.y * image.height);
      const w = Math.max(1, Math.round(relative.w * image.width)), h = Math.max(1, Math.round(relative.h * image.height));
      const pixels = context.getImageData(x, y, w, h).data;
      image.close();
      const sum = [0, 0, 0]; let clipped = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i] >= 254 || pixels[i + 1] >= 254 || pixels[i + 2] >= 254) clipped += 1;
        for (let c = 0; c < 3; c += 1) sum[c] += srgbToLinear(pixels[i + c]);
      }
      const count = pixels.length / 4;
      if (clipped / count > 0.01) clippedPhotos += 1;
      added.push([labels.object, labels.session, labels.light, labels.repeat, ...sum.map((value) => (value / count).toFixed(5))].join(','));
    } catch { problems.push(`${file.name}: could not be opened`); }
  }
  const area = $('#phone-input');
  const existing = area.value.trim();
  const header = 'object_id,session_id,light,repeat,r,g,b';
  area.value = `${existing && !state.phoneDemo ? existing : header}\n${added.join('\n')}\n`;
  state.phoneDemo = false;
  $('#phone-file-name').textContent = 'From your photos';
  $('#phone-kind').textContent = 'Values from tone-mapped browser images. Fine for a stability check; treat the independent-measurement count as exploratory.';
  const parts = [`Added ${added.length} row${added.length === 1 ? '' : 's'} to the table.`];
  if (clippedPhotos) parts.push(`${clippedPhotos} photo${clippedPhotos === 1 ? ' has' : 's have'} over 1% clipped pixels in the box. Lower the exposure or light for those.`);
  if (problems.length) parts.push(`Skipped ${problems.length}: ${problems.slice(0, 3).join('; ')}${problems.length > 3 ? '; …' : ''}.`);
  $('#roi-status').textContent = parts.join(' ');
  button.disabled = false;
});

function clearPhoneResult() { state.phone = null; $('#phone-results').hidden = true; }
$('#phone-input').addEventListener('input', () => { state.phoneDemo = false; clearPhoneResult(); $('#phone-file-name').textContent = 'Edited table'; $('#phone-kind').textContent = 'Values are used as supplied. The app cannot verify how they were captured.'; });
$('#phone-file').addEventListener('change', async (event) => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 2000000) { message('#phone-message', 'Please choose a CSV smaller than 2 MB, or use the command-line tool.', 'error'); return; }
  try { $('#phone-input').value = await file.text(); $('#phone-file-name').textContent = file.name; state.phoneDemo = false; clearPhoneResult(); message('#phone-message', 'CSV loaded. Select Analyze to run it.'); }
  catch { message('#phone-message', 'The selected file could not be read.', 'error'); }
});
$('#load-phone-demo').addEventListener('click', async () => {
  const button = $('#load-phone-demo'); button.disabled = true;
  try { const result = await api('/api/repeatability-example'); $('#phone-input').value = result.csv; state.phoneDemo = true; clearPhoneResult(); $('#phone-file-name').textContent = 'Simulated example · measured reflectance, simulated phone'; $('#phone-kind').textContent = 'SIMULATED EXAMPLE · 10 objects, 3 sessions, 4 lights, with known noise.'; message('#phone-message', 'Simulated example loaded. Select Analyze to see the report.'); }
  catch (error) { message('#phone-message', error.message, 'error'); }
  finally { button.disabled = false; }
});
function renderPhone(result) {
  state.phone = result;
  const synthetic = result.data_kind === 'synthetic', counts = result.counts;
  $('#phone-data-kind').textContent = synthetic ? 'SIMULATED INPUT' : 'SOURCE UNVERIFIED';
  $('#phone-provenance').textContent = synthetic ? 'This report uses simulated photos with known noise. It shows the analysis working and says nothing about any real phone.' : 'These values were supplied to the app. It has not verified how the photos were taken or processed.';
  $('#phone-photos').textContent = counts.n_photos;
  $('#phone-photos-note').textContent = counts.n_missing_combinations ? `${counts.n_missing_combinations} object/session/light combinations missing` : 'No combinations missing';
  $('#phone-design').textContent = `${counts.n_objects} × ${counts.n_sessions}`;
  $('#phone-design-note').textContent = counts.crossed ? 'Crossed design' : 'Objects are not repeated across sessions';
  $('#phone-lights').textContent = counts.n_lights;
  const rank = $('#phone-rank'); rank.replaceChildren();
  const groups = result.independent_measurements || [];
  if (!groups.length) rank.append(node('p', 'microcopy', 'Not enough repeated photos to count independent measurements. Each object needs at least two photos under each light.'));
  groups.forEach((group) => {
    const row = node('div', 'phone-rank-row');
    const label = group.lights.length > 1 ? `All ${group.lights.length} lights together` : group.lights[0];
    const value = node('strong', '', `${group.independent} of ${group.channels}`);
    const bar = node('div', 'rank-bars');
    bar.append(...Array.from({ length: group.channels }, (_, i) => node('i', i < group.independent ? 'filled' : '')));
    row.append(node('span', 'phone-rank-label', label), value, bar);
    rank.append(row);
  });
  const rows = [];
  (result.lights || []).forEach((light) => {
    ['r', 'g', 'b'].forEach((channel) => {
      const c = light.channels[channel];
      const session = node('td', '', fmt(c.sd_session, 4));
      session.append(node('small', '', c.sd_session == null ? 'Needs 2 or more sessions' : c.sd_session_interval ? `[${fmt(c.sd_session_interval.low, 4)}, ${fmt(c.sd_session_interval.high, 4)}]` : c.sd_session > 0 ? 'Too few sessions for an interval' : 'Estimated at zero'));
      const tr = node('tr');
      tr.append(node('td', '', light.light), node('td', '', channel.toUpperCase()), node('td', '', fmt(c.mean, 3)), node('td', '', fmt(c.sd_residual, 4)), session,
        node('td', '', fmt(c.sd_object_by_session, 4)), node('td', '', fmt(c.repeatability_coefficient, 4)), node('td', '', fmt(c.reproducibility_coefficient, 4)), node('td', '', fmt(c.icc_agreement, 3)));
      rows.push(tr);
    });
  });
  $('#phone-table').replaceChildren(...rows);
  listItems('#phone-notes', result.notes);
  $('#phone-hash').textContent = `Input SHA-256: ${result.input_sha256}`;
  $('#phone-results').hidden = false;
}
$('#analyze-phone').addEventListener('click', async () => {
  const csv = $('#phone-input').value;
  if (!csv.trim()) { message('#phone-message', 'Add photos with the box tool, choose a CSV, paste a table, or load the simulated example first.', 'error'); return; }
  const button = $('#analyze-phone'); button.disabled = true; clearPhoneResult(); message('#phone-message', 'Estimating variance components and the noise floor…', 'loading');
  try {
    const result = await api('/api/repeatability', { csv });
    renderPhone(result);
    message('#phone-message', result.data_kind === 'synthetic' ? 'Simulated example analyzed. No phone was tested.' : 'Analysis complete. Check the notes before interpreting the results.');
  } catch (error) { message('#phone-message', error.message, 'error'); }
  finally { button.disabled = false; }
});
$('#export-phone').addEventListener('click', () => { if (state.phone) download(state.phone, 'myco-fusion-phone-test.json'); });
