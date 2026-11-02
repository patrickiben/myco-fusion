# Study plan: how far can a phone camera go on measured spectra?

Written on 2026-11-01, before any accuracy metric was computed on the natural-object or mushroom data. The file structure of those datasets (columns, counts, value ranges) had been inspected to write the loaders.

**What had already been seen.** A first probe used the 190-patch and 24-chart reflectance sets, the Nikon sensitivity and CIE illuminants. Its results motivated these questions, so results on those sets are exploratory. The natural-object and mushroom results are the first look at those data.

This plan has no independent timestamp, so treat it as a statement of order, not proof. Any change made after seeing results is listed under Deviations at the end.

## Scope

This study simulates a camera on measured reflectance spectra. It does not test a real phone capture, and it does not test identification of any species. Nothing here supports decisions about eating fungi.

## Data

| Set | Use | Source |
|---|---|---|
| Natural objects: 532 samples, 10 domains, repeats averaged per sample | Primary test set | Gutiérrez et al. 2025, CC BY 4.0 |
| Mushrooms: 250 samples, mean reflectance per sample, species not named | Primary test set | SpectroFood, Malounas et al. 2024 |
| 24-patch color chart, BabelColor average | Calibration only | via colour-science |
| 190-patch mixed set | Exploratory comparison | via colour-science (rawtoaces) |

Metrics use 27 bands, 420 to 680 nm in 10 nm steps. Reference values are clipped to [0, 1], and the number of clipped values is reported.

## Forward model

- Grid: 400 to 700 nm in 5 nm steps, with linear interpolation of every curve.
- Cameras: measured sensitivities of the Nikon D5100 (primary) and Sigma SD Merrill.
- Flash stand-in: CIE LED-B4, with LED-B1 and LED-B5 as a sensitivity check.
- Response: each channel is normalized so a perfect white under an equal-energy light reads 1, and each light is normalized to mean 1 over the grid.
- Noise, applied per capture: additive noise on every channel, plus one gain error shared by all channels of that capture.

| Noise level | Additive SD | Gain error SD |
|---|---|---|
| Low | 0.002 | 0.5% |
| Moderate (primary) | 0.005 | 2% |
| High | 0.015 | 5% |

Each extra light is a separate capture with its own gain error. Screen light is dimmer than flash, so screen captures also carry additive noise multiplied by 3 in the primary analysis, and by 1 and 5 as checks.

## Estimators

- Primary: affine ridge from captured channels to 27 bands, the same form as the app's calibration, with the penalty chosen by inner cross-validation.
- Sensitivity, one-photo arm only: degree-2 polynomial ridge and RBF kernel ridge. These test whether a nonlinear model gets past three numbers.
- Baseline: constant training-median spectrum.

## Validation

- Natural objects: 5-fold cross-validation with folds split by species, or by sample when no species is recorded.
- Mushrooms: 5-fold cross-validation by sample.
- Chart-calibrated models are fitted on the chart only and tested on everything else.

## Reported metrics

Median, 90th percentile and worst spectral angle and mean absolute error (MAE), and the share of samples meeting both draft targets (angle at most 10°, MAE at most 0.05). The draft targets are placeholders. Comparisons between arms use per-sample paired differences with a 95% cluster bootstrap by species (2,000 resamples).

## Questions and what counts as an answer

**Q1. Is three numbers a hard limit?** Compare the one-photo estimate with the error from the best linear k-number summary (PCA fitted on training folds, k from 1 to 8) and with the instrument's own repeat error. *The limit holds on a dataset if neither nonlinear estimator lowers median MAE by more than 10% below the affine model.*

**Q2. Does chart calibration transfer to biological material?** Calibrate on the chart, test on each natural-object domain and on mushrooms. Compare with calibration on the same domain or set. *Transfer is adequate on a domain if chart-calibrated median MAE is within 20% of same-domain cross-validation.*

**Q3. How much does changed light hurt, and what restores it?** Train under flash, test under D65, A, LED-B1, LED-B5, FL2 and an illustrative green shade (D65 multiplied by the median leaf reflectance). Arms: no correction, white-patch gain correction, recalibration on the chart under the test light. *A correction helps if the 95% interval of the paired median MAE reduction excludes zero and the median reduction is at least 20%.*

**Q4. Do extra lights the phone already has add real measurements?** Arms: flash; three flash brightness levels; flash plus stable ambient D65; screen red, green and blue; flash plus screen red, green and blue. Each arm is calibrated two ways: cross-validation on the test set, which assumes spectrometer-measured training targets, and the 24-patch chart alone, which is what a person without a spectrometer could do. Screen primaries: three Gaussian sets (narrow, medium, wide; medium is primary) because no measured iPhone screen curve is available here, plus the measured Apple Studio Display curves from colour-science as a real-display check. *An arm helps if the 95% interval of the paired median MAE reduction versus flash alone excludes zero and the median reduction is at least 20%, at moderate noise with screen noise multiplied by 3.* Also reported: the number of independent measurement directions, counted as singular values of the stacked sensitivity operator above 2% of the largest.

**Q5. Do real objects form metamer pairs, and do extra lights separate them?** Among natural-object samples from different species, count pairs whose flash RGB differences are within noise on every channel but whose spectra differ by an angle above 10° or an MAE above 0.05. Report how many of those pairs flash plus screen light can tell apart. The clearest pair replaces the constructed pair in the app.

**Q6. Do the app's intervals mean what they say?** Simulation of specimens nested in sessions, with the original percentile-bootstrap methods against exact order-statistic intervals. Sample sizes: 3 to 40 specimens, 2 to 20 sessions. Scenarios: lognormal with session share of variance 36%, smaller session share, heavier-tailed session effects, and unequal session sizes.

## Addendum: measured iPhone curves (Q7)

Added on 2026-11-01, after every result above and before any result with these curves. It does not change Q1 to Q6.

**Question.** Do the Nikon conclusions hold with measured iPhone sensitivities?

**Curves.** iPhone 6s, SE, 8, X, 11 and 12 Pro Max from Tominaga et al. 2021, measured at 400 to 700 nm in 10 nm steps. The source states no license, so they are downloaded with a hash check and never stored in this repository. No curve for a newer iPhone was found.

**Design.** Natural objects only, with the same noise, flash stand-in, folds, lights and screen colors as above. Each light is exposed to match flash on a white target for that phone. For each phone:

- One photo, calibrated on similar objects: share meeting both targets and median MAE, plus the nonlinear gain against the 10% rule.
- One photo, chart only: share and median MAE.
- Changed light on chart calibration: no correction, white card, chart under that light, with the 20% bar counted across the six lights.
- Flash plus the medium screen colors against flash, under both calibrations, against the 20% bar.

**What counts as an answer.** A Nikon conclusion holds for phones if it has the same outcome under the same rule on at least five of the six phones. The 12 Pro Max is reported on its own as the newest available. The flash is still a stand-in, so this tests the camera curves only.

## Deviations

Added before the first run, with no results seen:

- Chart-only calibration for the Q4 arms.
- Calibration on natural objects as an extra row in the mushroom transfer test (Q2).
- A count of mushroom and natural-object look-alike pairs (Q5, exploratory).

Changed after seeing results. A separate AI code review of the first full run found the first two problems below:

- **Screen light was too strong.** Every light had been normalized to a mean of 1, which packs a narrow screen color's energy into one channel. On a white target the screen read 2 to 3 times what flash read, so tripling its noise left it about as clean as flash. Each light is now scaled as exposure would scale it: its strongest channel on a white target matches flash, and screen photos then carry three times the noise. This erased most of the reported screen-light gain.
- **The comparison statistic now matches this plan.** The first run used the change in medians. Q3 and Q4 now use the median per-sample difference, as a percent of the baseline median. Q1 keeps the change in medians, as its rule is written, and now reports a cluster bootstrap interval.
- **Noise draws** went from 5 to 20 per arm, because chart-calibrated estimates of one arm differed by up to 5% between labels. Summaries now use each sample's error averaged over draws, and the flash arm reuses one run throughout.
- **Grouping** now also keeps patches of one numbered specimen together, and corrects one misspelled species. Results barely moved.
- **Q5 reports more.** It now breaks down why pairs exceed a target, and adds a stricter look-alike test that allows for gain errors shared across a photo's channels. The example pair is the largest spectral angle passing that test, among objects with mean reflectance above 0.08. Under the original rule the same pair was picked; a later rule that preferred bright objects was dropped because its pair failed the stricter test.
- **Share intervals** were added for the headline shares, using the same cluster bootstrap.
