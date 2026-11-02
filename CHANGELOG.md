# Changelog

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
