# Data sources

Every reflectance spectrum in this project is a real measurement. RGB values are simulated from those spectra, so no result here comes from a phone photo.

| Data | Used for | Source | Terms | In this repository? |
|---|---|---|---|---|
| 532 samples of natural objects from 430 specimens: barks, beetles, birds, butterflies, flowers, fruits, leaves, plant fruits, stones, vegetables | Main test set, app demos | [Gutiérrez et al. 2025](https://doi.org/10.1038/s41597-025-04675-9), [dataset](https://doi.org/10.6084/m9.figshare.25705380.v4) | CC BY 4.0 | Yes, as a 27-band table in `myco_fusion/data/` |
| 250 mushroom samples, species not named | Main test set | [Malounas et al. 2024](https://doi.org/10.1016/j.dib.2024.110040), [mean spectra](https://zenodo.org/records/8362947) | No license stated on the CSV record | No. `study/sources.py` downloads it and checks its hash |
| 24-patch color chart (BabelColor average) | Calibration only | [colour-science](https://www.colour-science.org/) | Installed with `pip install ".[study]"` | No |
| 190-patch mixed reflectance set (rawtoaces) | Exploratory comparison | [colour-science](https://www.colour-science.org/) | Installed with `pip install ".[study]"` | No |
| Nikon D5100 and Sigma SD Merrill sensitivities | Camera model | [colour-science](https://www.colour-science.org/) (NPL measurements) | Installed with `pip install ".[study]"` | No |
| iPhone 6s, SE, 8, X, 11 and 12 Pro Max sensitivities | Camera model, Q7 | [Tominaga et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8347217/), [database](http://ohlab.kic.ac.jp/index/mobilespec_database) | No license stated | No. `study/sources.py` downloads them and checks their hashes |
| CIE illuminants; Apple Studio Display primaries | Light model | [colour-science](https://www.colour-science.org/) | Installed with `pip install ".[study]"` | No |

## Things to know about the data

- **Natural objects.** Measured with a SpectraScan PR-715 spectroradiometer from 400 to 1000 nm in 4 nm steps. Objects came from northwestern Argentina, so other regions may differ. A sample is one color patch. Most have three readings taken in similar regions of the object, which give the repeat-measurement error in the findings. That figure compares each reading with the mean of the others, so it somewhat overstates the noise in the three-reading mean used as the reference.
- **Mushrooms.** Mean reflectance per sample from a SpecimIQ hyperspectral camera, with white and dark references. Dry matter runs from about 6% to 87%. The source says only "mushroom" and does not name a species, so this cannot speak to wild species or pigments.
- **Chart, camera, light and display curves.** The study loads these through colour-science at run time rather than copying them here.
- **iPhone curves.** Each file has 31 rows and 3 columns, covering 400 to 700 nm in 10 nm steps as the paper describes. The files have no wavelength column, and their peaks sit near 600, 530 and 470 nm, as expected for red, green and blue.
- **What is missing.** No public reflectance data for a variety of wild fungi was found. No measured curve was found for an iPhone newer than the 12 Pro Max, and no measured iPhone screen or flash spectrum was available.
