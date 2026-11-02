# Paired capture protocol (needs a reference instrument)

This draft protocol evaluates a specific camera and calibration pipeline against reference reflectance. It does not evaluate species identity, toxicity, consumption decisions or continual learning. For a test that needs no instrument, start with the [phone test](PHONE_TEST.md).

## Aim and first experiment

Start with one documented phone/camera mode, a contact spectrometer with a documented calibration procedure, and physical reflectance targets. Establish repeatable paired measurements under controlled flash (R1), then repeat the held-out targets under shielded outdoor flash (R3). The workbench can assess imported spectra; acquisition and spectral reconstruction must currently be performed externally and documented.

Use separate calibration targets and held-out evaluation targets spanning dark, neutral, colored, and glossy surfaces. A calibration chart tested on the same patches is a fitting check, not evidence of generalization. Initially use three independently repositioned repeats per target/regime/session to expose acquisition variability. This repeat count is a pilot choice, not a sample-size justification.

## Regimes

| Code | Acquisition | Purpose |
| --- | --- | --- |
| R1 | Controlled low ambient light plus fixed phone flash | Establish the baseline measurement chain. |
| R3 | Outdoor canopy or comparable field setting, standardized shield plus flash | Test the proposed field protocol with the R1 reconstruction frozen. |
| R2 | Ambient field light without flash, optional comparison | Measure domain shift and whether the shield/flash improves performance. No failure is required. |
| R4 | Documented overcast condition, optional comparison | Sample another illumination condition without calling it a universal worst case. |

Acquire R1 and R3 on the same held-out targets in multiple sessions. Record order and randomize it where practical. Do not refit on R3 evaluation spectra. A separately calibrated R3 model is a new, explicitly named experimental arm.

## Measurement and preprocessing controls

1. Record phone model, camera/lens, software versions, RAW mode, exposure time, ISO, focus, white balance, flash settings, distance, angle, illumination, and ambient conditions. Lock settings where supported and log unsupported controls.
2. Save original files. Record black/white levels, demosaicing, linearization, color transforms, white-balance gains, normalization, and every processing version. Linear gamma alone does not ensure a sensor-linear or exposure-comparable signal. Avoid undocumented automatic tone mapping, denoising, HDR, or clipping.
3. Check dark signal, white-reference stability, saturation, underexposure, and instrument repeatability before evaluation. Retain flagged acquisitions with explicit exclusion reasons and counts; never silently discard them.
4. Measure reference spectra of the same marked spot with matched timing. Record aperture/spot size, illumination/collection geometry, reference standard, integration time, and instrument spectral resolution. Contact and camera geometries can differ, especially on glossy or curved targets; quantify that mismatch.
5. Record wavelength grid and units. The current example grid is 420–680 nm in 10 nm increments (27 values). Sampling interval is not instrument spectral resolution. Document resampling and prohibit extrapolation beyond the reference instrument's supported range.
6. Store reconstructed reflectance and measured reference reflectance separately. Include calibration/model identifiers and hashes. Record clipping before and after reconstruction; clipping estimates into [0,1] must not hide model failure.

Flash intensity repeats are an exposure/noise experiment. An identifiability experiment requires independently characterized, spectrally different illumination and a rank/conditioning analysis with explicit sensor and reflectance-basis assumptions. The app's simulated camera says nothing about a real phone's response.

## Split and record design

- Assign `target_id` or `specimen_id`, `spot_id`, `session_id`, `regime`, `repeat_id`, and `split` before fitting. Use stable unique physical identities, not filenames as independence units.
- Keep all spots, views, and repeats of a physical target/specimen in one training/tuning/test partition. Reserve acquisition sessions for external evaluation; prevent repeated targets from leaking across session-held-out partitions.
- Choose calibration hyperparameters on training/tuning data. Freeze the transform and processing pipeline before evaluating held-out targets. Never learn a spectral basis from test references.
- Preserve provenance, file hashes, licenses, instrument uncertainty, quality flags, and reasons for missing observations in the manifest. The current importer is a numerical comparison tool; it cannot infer missing provenance or certify split independence.

## Analysis and provisional thresholds

Compute spectral angle mapper (SAM, degrees) and mean absolute reflectance error (MAE, reflectance fraction) for each paired observation on an aligned wavelength grid. Zero-norm spectra make SAM undefined: report them as invalid for SAM with a reason, without converting them to a successful zero-angle result.

For each required regime, first take the arithmetic mean of repeat/spot errors within each unique physical target, then the median across targets so targets receive equal weight. Do not let many frames from one specimen inflate n. Report target counts, missingness, distributions, and worst-target errors alongside medians. For uncertainty, use the app's exact intervals, which report their true level. With few targets or sessions that level falls below 95%.

The importer accepts `session_id` and, when specimens are nested in sessions, computes intervals from one median per session. If the same targets are measured across sessions the design is crossed: analyze its repeatability in the Phone test tab, and treat the specimen-based accuracy interval as too narrow.

The draft engineering targets are SAM ≤10° and MAE ≤0.05. They are unvalidated placeholders. Report how many specimens meet both in R1 and R3, not only whether the medians do. Meeting them is not evidence of species discrimination or field safety. Missing R1/R3 or invalid measurements leave the comparison unassessed. No classifier-transfer gate is specified because its estimand and operating rule have not been designed.

Before a confirmatory experiment, finalize the sample-size rationale, aggregation, exclusions, uncertainty method, thresholds, and permitted adaptations, and create a dated immutable record. This document is a draft, not a preregistration.

## Evidence needed to advance

- **Measurement feasibility:** held-out R1/R3 observations, calibrated reference instrument, full provenance, repeatability, failure accounting, and results under the frozen pipeline.
- **Domain transfer:** independent voucher specimens with adjudicated identity if later classification is intended; held-out specimens/sessions and fair RGB baselines. Do not infer tissue performance from color tiles alone.
- **Spatial fusion:** verified camera/geometry coordinate transforms, stable scale, registration residuals, point visibility and missingness; later validate Gaussian footprint/opacity handling and dimensional error against reference geometry.
- **Continual learning:** a separate protocol with pre-update predictions, controlled label availability, session isolation, retention measurements, and frozen/replay baselines.

If R1 fails, diagnose this sensor/preprocessing/calibration protocol. If R1 succeeds and R3 fails, investigate illumination and shielding or test another documented acquisition arm. R2 success is useful evidence, not a failed experiment. Keep the broader project viable while narrowing claims to what was actually tested.
