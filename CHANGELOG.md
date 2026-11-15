# Changelog

## Unreleased

- **Component admission trial (2026-11-14).** A plan addendum, approved and committed before it ran, tested whether four added estimators improve the one-photo estimate: a polynomial fit, a kernel fit, a linear fit with a smoothness penalty, and a forward model from the camera and flash curves. It also tested a weighted combination of estimators and a forward-model choice of screen color. Checks of the admission rule on parts of known value ran first; one rounding defect in those checks was fixed and recorded under Deviations before the trial. Only the kernel fit passed, and its median gain was far below the 10% bar. Code in `study/admission_trial.py`, results in `results/admission_rule_checks.json` and `results/admission_trial.json`, and finding 10 on the Findings page.
- Added `fit_smooth_tm` (a second-difference smoothness penalty) to `myco_fusion.spectra`, and `myco_fusion.sensor` (signal-dependent sensor noise, a cross-polarization model and photo-reading helpers). Both were written in an earlier session and checked here; the trial uses them.

## 0.3.1

- Repeated the study with measured sensitivities of six iPhones, up to the 12 Pro Max (plan addendum, Q7). Results stayed close to the DSLR's.

## 0.3.0

- Added a reproducible study on measured spectra, with a plan written before the main results were computed (`study/`, `results/`, `docs/FINDINGS.md`).
- Made Findings the landing tab and added a public Findings page (`site/`).
- Added the Phone test tab: repeatability across sessions, a count of independent measurements, a light panel and a tool that measures a box in your photos.
- Replaced percentile-bootstrap intervals labeled 95% with exact intervals that state their true level. In simulation the old session interval reached about 50% with 2 sessions and 75% with 3.
- Replaced the single pass label with a count of specimens meeting both draft targets, because a median can sit inside the targets while many specimens do not.
- Bench, calibration and validation demos now use measured reflectance with a simulated camera. The old demos were generated from the same model the solver assumes.
- Fixed a broken install: `pip install .` left the app without its interface and data.
- Raised text sizes and contrast across the app.
- Narrowed the repository to the measurement question.

## 0.2.0

- Added the calibration lab, session-aware validation and a research log.

## 0.1.0

- First runnable workbench, with a synthetic spectral bench, paired-spectrum validation and draft capture records.
