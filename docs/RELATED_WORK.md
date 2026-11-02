# Related work

Each entry was compared with its source on 2026-11-01. "Bearing here" is this project's reading, not the authors' claim.

**He et al. 2023, [SpeCamX](https://pmc.ncbi.nlm.nih.gov/articles/PMC10545193/).** An app that estimates a 27-channel cube from 420 to 680 nm on an unmodified Pixel 4. It computes a transformation matrix by Wiener estimation from a ColorChecker photographed under the phone's flash, and reports an average RMSE below 0.04 on a separate ColorChecker Digital SG, plus a bilirubin study in 320 patients imaged in a dark ward. *Bearing here:* the chart-calibration arm follows this recipe. This study's result that a chart transfers unevenly to other materials is about a simulated camera, and does not test SpeCamX.

**Yeo et al. 2017, [SpeCam](https://make.kaist.ac.kr/project/2017-specam).** Places a phone face down and "uses only the front-facing camera and the display as a multi-spectral light source", identifying 30 surface materials in the authors' study. *Bearing here:* prior art for using a screen as a changing light. This project asks a narrower question, about estimating reflectance, and in simulation found little median gain.

**Lin and Finlayson 2020, [Physically Plausible Spectral Reconstruction](https://www.mdpi.com/1424-8220/20/21/6399).** Describes the spectra that give one RGB as a fundamental metamer plus metameric blacks, and builds reconstruction that respects this. *Bearing here:* the reason three channels leave spectral directions unseen.

**Fu et al. 2024, [Limitations of Data-Driven Spectral Reconstruction](https://arxiv.org/abs/2401.03835).** Tests learned RGB-to-spectrum models under "metameric or near-metameric conditions" and argues that published scores overstate real-world performance. *Bearing here:* a reason to keep held-out tests and tail errors in view.

**Guo et al. 2026, [PocketGS](https://arxiv.org/abs/2601.17354).** Reports 3D Gaussian splatting trained entirely on a commodity phone, with post-capture processing averaging 255.2 s and under 3 GB of peak memory on its MobileScan data. The abstract reports no millimeter geometric accuracy. *Bearing here:* the 3D half of the original idea. It is not built or tested here.

**Behrouz et al. 2025, [Nested Learning](https://arxiv.org/abs/2512.24695).** Proposes a sequence model that learns its own update algorithm, with a continual-learning module called Hope. *Bearing here:* part of the original idea. No continual learning is implemented here, because there is no labeled stream of real captures to learn from.
