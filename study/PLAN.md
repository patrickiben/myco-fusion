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

## Addendum: component admission trial

**Status on 2026-11-14:** approved by the author before any part of it was run. This addendum does not change Q1 to Q7 or their results.

An AI coding agent drafted this addendum at the author's request, and the author approved it on 2026-11-14, before any part of it was run. Before the run, this addendum is committed, and the SHA-256 (Secure Hash Algorithm, 256-bit) hashes of this file and of the trial's code are written into the results file.

**Questions.** Several components could be added to the one-photo estimate. Does each one improve held-out accuracy over the simpler method that already does its job? Does a combination of estimators outperform the best single one? Can the admission rule itself tell a useful component from a useless one on this problem?

**What had already been seen.** Q1 compared the polynomial and kernel estimators with the affine estimator on these natural objects, and Q4 tested flash plus all three screen colors together. No result had been computed for the smoothing estimator, the forward-model estimator, any combination of estimators, any single added screen color, or the shot-noise model. The smoothing code had been checked against a direct solution on random numbers, which says nothing about its accuracy. Every natural-object sample was used in Q1 to Q7, so this trial is not a first look at the data.

### Data, split, and simulation

- Natural objects only: 532 samples in 314 groups. The mushroom set is not used, because its curves have nearly one shape.
- One split by group, drawn once with seed 20261114: half of the samples for development and half held out, with every group kept whole on one side.
- Camera, flash, and noise as above: Nikon D5100, LED-B4, moderate noise, and 20 noise draws. Each sample's error is its MAE over the 27 bands, averaged over the draws.
- Every penalty, setting, and weight is chosen on the development half. The held-out half is used once, after everything is fixed.

### Components

| Component | What it is | Simpler method it must outperform |
|---|---|---|
| Affine ridge | The study's primary estimator | None: it is the reference and is always included |
| Polynomial ridge | Degree 2, as in Q1 | Affine ridge |
| Kernel ridge | RBF kernel, as in Q1 | Affine ridge |
| Smoothing ridge | Affine ridge with an added penalty on the second difference across bands. The penalty weight (0.0001, 0.01, 1, or 100) is chosen with the ridge penalty by inner cross-validation. | Affine ridge |
| Forward-model estimator | It fits no map from photo to spectrum. It describes a spectrum by the development mean and its first k principal directions (k from 3 to 8, chosen by inner cross-validation), predicts the photo from the camera and flash curves, and returns the most probable spectrum given the photo and the moderate noise. The part of a spectrum that the k directions leave out is treated as extra noise. | Affine ridge |
| Combination | A weighted average of the five estimators above, with one weight per estimator. The weights are at least zero, sum to one, and are fitted by least squares to development predictions that each estimator made for samples outside its own training folds. | The best single estimator on development, and the plain average |

The forward-model estimator is given the true camera and flash curves, which is its best case. A second arm gives it the measured iPhone 12 Pro Max curves while the photos come from the Nikon, to show the cost of a wrong model. That arm is reported and is not tested for admission.

### Comparison statistic

For two systems, the difference is the mean over held-out samples of the paired difference in MAE. Its 95% interval is the percentile interval over 2,000 bootstrap resamples of held-out groups. The one-sided p-value is (1 + the number of resamples at or below zero) / 2,001. The statistic used in Q3 and Q4 (the median paired difference as a percent of the baseline median) and the share of samples meeting both draft targets are reported beside it and decide nothing.

### Admission rule

A component passes only if all four conditions hold on the held-out half. Otherwise, it does not pass.

1. **Frozen items unchanged.** The Q1 to Q7 result files are byte-identical after the run, the draft targets are unchanged, and every test passes.
2. **Unique gain.** Removing the component and refitting the combination on development data worsens held-out error: the interval of the error without the component minus the error with it lies above zero, and the Holm-adjusted p-value across the four candidate components is below 0.05.
3. **Corruption check.** Replacing the component's predictions with corrupted ones and refitting does not lower held-out error. Two corruptions are used: predictions shuffled across samples, and predictions reflected about the development mean spectrum. For each, the paired difference (corrupted minus real) must be at least zero.
4. **Better than the simpler method.** On its own, the component has lower held-out error than affine ridge, with the interval above zero.

The combination passes if its held-out error is lower than that of the best single estimator on development, with the interval above zero. Its difference from the plain average is reported.

A pass is an offline result on simulated photos. It does not put a component into the app or into the primary analysis of Q1 to Q7; that would need paired phone and spectrometer measurements. Size is reported against the rule in Q1: a component is called practically useful only if it also lowers median MAE by more than 10% below affine ridge.

### Checks of the rule itself, run first

A rule that passes nothing, or passes everything, gives no information. Before the verdicts on the real components are read, the rule is run on components whose value is known. The same 200 replicates serve all three checks. Each replicate draws its own split of groups and one noise draw, with the five real estimators present, so these checks are somewhat harder than the main trial, which averages 20 draws. In these checks, the Holm adjustment covers the four candidate components and the added ones.

- **Useless component:** affine predictions shuffled across samples, so they carry no information about the sample.
- **Planted component:** the affine prediction moved toward the true spectrum by a fraction *a* of its error, for *a* of 0.05, 0.1, 0.2, and 0.5, one at a time.
- **Two near duplicates:** two planted components with *a* of 0.5, each with independent noise of SD 0.005 per band. Reported: how often at least one passes when each is removed alone, and how often the pair passes when the two are removed together.

*The rule is called usable here if the useless component passes in at most 5% of replicates and the planted component with a of 0.2 passes in at least 80%.* If the rule is not usable, the verdicts on the real components are reported with that statement and are not called findings. If the near duplicates pass alone in fewer than half of the replicates, the polynomial and kernel estimators are also tested as a pair removed together, and that result is reported as exploratory.

### One added screen color

- Arms: flash plus one medium screen color (red, green, or blue, as defined in Q4), with exposure matched to flash and screen noise multiplied by 3, estimated by affine ridge on the six numbers.
- Forward-model choice: the color with the lowest predicted mean squared error over the 27 bands, computed from the development half and the camera and light curves. The choice is written to the results file before any held-out error is computed.
- Simple rule: always blue, a fixed choice made here without any calculation.

*The forward-model choice passes if it is not blue and its held-out error is lower than blue's, with the interval above zero.* Each color is also reported against flash alone under the rule in Q4.

### Shot-noise check

The estimator trial and the screen-color comparison are repeated with photos simulated under signal-dependent noise: read noise of SD 0.002, a shot-noise factor of 0.0065 (so the total at a signal of 0.5 equals the moderate additive noise), gain error of 2%, and 12-bit rounding. The forward-model estimator keeps the moderate noise in its own model. *A verdict is called robust to the noise model if it is the same under both.*

### Limits stated in advance

- This is a simulation on spectra already used in Q1 to Q7, with one split.
- Intervals are pointwise 95% intervals. Only the unique-gain p-values are Holm-adjusted.
- Admission by unique gain can reject each of two useful components that overlap. The near-duplicate check measures how often that happens here.
- If the checks of the rule would take more than about 30 minutes of computing, the replicate count is reduced before any result is read, and the reduction is listed under Deviations.

### Details fixed before the run

Written on 2026-11-14, after the code in `study/admission_trial.py` was built and tested on synthetic numbers, and before any part of the trial or its checks was run. They settle points the text above leaves open and change no rule.

- **Split.** The study's group-balancing function assigns whole groups to two halves with seed 20261114, and the first half is development. Development is split into five folds by group for the out-of-fold predictions. Inside every fit, penalties and settings are chosen by the study's own four-fold search.
- **Smoothing ridge.** `fit_smooth_tm` as written, which penalizes the second differences of every row of the fitted map, including the constant row. The ridge penalty grid is the study's.
- **Forward-model estimator.** The description of spectra and the photo prediction use the 5 nm simulation grid from 400 to 700 nm, and the estimate is read on the 27 bands. The gain error enters its noise as 2% of the observed photo, shared by the three channels. The part of a spectrum outside the k directions enters as noise through the camera and light curves. The number of directions is the one with the lowest mean MAE over the four folds.
- **Combination weights.** Found by an exact search over every subset of the estimators.
- **Best single estimator.** Chosen in each noise draw by its out-of-fold development MAE.
- **Bootstrap.** One set of 2,000 resamples of held-out groups serves every contrast within an analysis, and each check replicate draws its own set.
- **Corruptions.** A new random order is drawn in each noise draw and applied separately to the development and held-out predictions. The reflection uses the mean development spectrum on the 27 bands.
- **Checks of the rule.** Condition 1 is checked once for the whole run, not in each replicate. Besides the full rule, the pass rate of each condition, and of conditions 2 and 3 together, is reported. Condition 4 alone rejects the useless component and passes the planted ones, so the rate for conditions 2 and 3 is the one that tests the statistics. For the pair, condition 2 removes both together, with the pair and the four candidates as the Holm family; condition 3 corrupts both together; and condition 4 requires each member to outperform affine ridge.
- **Screen colors.** The flash photo of each draw is reused, and one screen photo is added. The forward-model choice uses the best linear estimate from the full development covariance, with the moderate noise and its gain term, and with the additive noise tripled for the screen photo.
- **Shot-noise check.** Photos come from `myco_fusion.sensor.simulate_sensor`, which clips values to the range 0 to 1 before the 12-bit rounding. For screen photos, the read and shot noise are tripled and the gain error is not, as in the study's rule.
- **Frozen items.** The files are `results/study.json`, `results/coverage.json`, `results/phone_power.json`, and `myco_fusion/data/findings.json`. Each must match its committed copy before and after the run, the draft targets must still be an angle of 10° and an MAE of 0.05, and the test suite runs after the trial.
- **Commands.** `python -m study.admission_trial checks` writes `results/admission_rule_checks.json`. `python -m study.admission_trial trial` refuses to run until that file exists and writes `results/admission_trial.json`.

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
