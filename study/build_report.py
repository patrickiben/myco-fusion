"""Turn results/*.json into findings.json, docs/FINDINGS.md and the public site.

Numbers in the text come from the results files. check_claims() stops the build if a
rerun no longer supports the main written conclusions. Wording still needs reading.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from myco_fusion.synthetic import run_synthetic

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DATA = ROOT / "myco_fusion" / "data" / "findings.json"
APP = ROOT / "myco_fusion" / "app"
SITE = ROOT / "site"
SOURCES = [
    {"label": "Gutiérrez et al. 2025, natural-object reflectance (CC BY 4.0)", "url": "https://doi.org/10.1038/s41597-025-04675-9"},
    {"label": "Malounas et al. 2024, SpectroFood dataset", "url": "https://doi.org/10.1016/j.dib.2024.110040"},
    {"label": "He et al. 2023, SpeCamX", "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10545193/"},
    {"label": "Yeo et al. 2017, SpeCam", "url": "https://make.kaist.ac.kr/project/2017-specam"},
    {"label": "colour-science, for chart, camera, illuminant and display curves", "url": "https://www.colour-science.org/"},
]


def pct(value, digits=0):
    return f"{100 * value:.{digits}f}"


def drop_text(reduction):
    """Whole-number percent with its interval, without a stray minus zero."""
    whole = lambda value: f"{round(value) + 0:d}".replace("-", "−")
    return f"{whole(reduction['percent'])}% ({whole(reduction['low'])} to {whole(reduction['high'])})"


def check_claims(s, c):
    """Fail loudly if results stop supporting the main written conclusions."""
    natural = s["q1"]["natural"]
    mixed = s["q1"]["mixed"]
    q3 = s["q3"]["natural"]["lights"]
    primary = s["q4"]["primary"]["natural"]
    charted = s["q4"]["chart_calibrated"]["natural"]
    mushrooms_chart = s["q4"]["chart_calibrated"]["mushrooms"]
    sensitivity = s["q4"]["sensitivity"]
    q5 = s["q5"]
    white_helps = [row["white_patch_reduction"]["helps"] for row in q3.values()]
    checks = {
        "the natural-object nonlinear gain spans the 10% line": natural["nonlinear_gain"]["low"] <= 10 <= natural["nonlinear_gain"]["high"],
        "the mixed-set nonlinear gain clears the 10% line": mixed["nonlinear_gain"]["low"] > 10,
        "the limit holds on mushrooms": s["q1"]["mushrooms"]["limit_holds"] and s["q1"]["mushrooms"]["limit_resolved"],
        "a nonlinear fit on one photo beats a linear fit on four at the median": s["q4"]["nonlinear_one_photo_vs_linear_four_photos"]["high"] < 0,
        "white card clears the bar under 4 of 6 lights": sum(white_helps) == 4 and not q3["D65 daylight"]["white_patch_reduction"]["helps"] and not q3["FL2 fluorescent"]["white_patch_reduction"]["helps"],
        "chart under the light clears the bar under every light": all(row["chart_under_test_light_reduction"]["helps"] for row in q3.values()),
        "no correction is far below the same-light share": max(row["none"]["share_within_targets"] for row in q3.values()) < s["q3"]["natural"]["same_light"]["share_within_targets"] - 0.2,
        "flash plus screen has no median gain on natural objects": primary["flash_plus_screen_rgb"]["reduction_vs_flash"]["low"] <= 0 <= primary["flash_plus_screen_rgb"]["reduction_vs_flash"]["high"],
        "no arm clears the bar on natural objects": not any(row.get("reduction_vs_flash", {}).get("helps") for block in (primary, charted) for row in block.values()),
        "flash plus screen improves the tail": primary["flash_plus_screen_rgb"]["mae_p90"] < primary["flash"]["mae_p90"] - 0.005,
        "screen alone is worse than flash": primary["screen_rgb"]["reduction_vs_flash"]["high"] < 0,
        "brightness adds no directions": primary["flash_three_brightness"]["independent_directions"] == 3,
        "extra-light arms get worse on mushrooms with chart calibration": all(mushrooms_chart[name]["reduction_vs_flash"]["high"] < 0 for name in ("flash_plus_ambient", "screen_rgb", "flash_plus_screen_rgb")),
        "only the Sigma row clears the bar in the sensitivity checks": [name for name, row in sensitivity.items() if row["reduction"]["helps"]] == ["camera: Sigma SD Merrill"],
        "camera curves move the share more than the model": natural["affine"]["share_within_targets"] - natural["sigma_camera"]["share_within_targets"] > natural["kernel"]["share_within_targets"] - natural["affine"]["share_within_targets"],
        "chart transfer is inadequate overall": not s["q2"]["natural_all"]["transfer_adequate"],
        "leaves are the worst transfer": max(s["q2"]["domains"], key=lambda d: s["q2"]["domains"][d]["ratio"]) == "Leaves",
        "chart transfer fails on mushrooms": not s["q2"]["mushrooms"]["transfer_adequate"],
        "constant spectrum meets the angle target on every mushroom": s["q1"]["mushrooms"]["constant_baseline"]["sam_worst"] <= 10,
        "one number nearly explains the mushroom data": s["q1"]["mushrooms"]["pca_floor"]["1"] < 0.01,
        "most above-target look-alikes differ by angle alone": q5["above_target_by_angle_only"] > 0.9 * q5["look_alike_above_a_draft_target"],
        "most above-target look-alikes include a dark patch": q5["of_those_with_a_dark_patch"] > 0.9 * q5["look_alike_above_a_draft_target"],
        "screen light separates few look-alikes": q5["share_separated"] < 0.1,
        "the example pair is the plum and the tourmaline": {q5["example_pair"]["a"]["object"], q5["example_pair"]["b"]["object"]} == {"Prunus domestica", "Black Tourmaline. Brazil"},
        "session bootstrap is well below 95% with 3 sessions": c["sessions"]["session share 36%"][1]["session_bootstrap_coverage"] < 0.8,
        "exact intervals match their stated level": all(abs(r["exact_coverage"] - r["exact_stated_level"]) < 0.03 for r in c["sessions"]["session share 36%"]),
    }
    failed = [claim for claim, holds in checks.items() if not holds]
    if failed:
        raise SystemExit("Results no longer support: " + "; ".join(failed))


ARM_LABELS = {
    "flash": "Flash only",
    "flash_three_brightness": "Three flash brightness levels",
    "flash_plus_ambient": "Flash plus steady daylight",
    "screen_rgb": "Screen red, green, blue",
    "flash_plus_screen_rgb": "Flash plus screen red, green, blue",
}
SIMILAR_LABEL = "Calibrated on hundreds of similar objects measured by spectrometer"
DATASET_LABELS = {"natural": "Natural objects", "mushrooms": "Mushrooms", "mixed": "Mixed set (exploratory)"}


def interval_text(reduction):
    return f"{drop_text(reduction)}"


def all_tests_table(s):
    """Every pre-stated test by dataset, whatever the result, plus every sensitivity row."""
    rows = []
    for name in ("natural", "mushrooms", "mixed"):
        entry = s["q1"][name]
        verdict = "Limit holds" if entry["limit_holds"] and entry["limit_resolved"] else "Limit fails" if entry["limit_resolved"] else "Unresolved: interval spans 10%"
        rows.append(["Q1 hard limit", DATASET_LABELS[name], f"Best nonlinear fit against linear fit (drop in median MAE)", drop_text(entry["nonlinear_gain"]), "At most 10%", verdict])
    for name, row in sorted(s["q2"]["domains"].items(), key=lambda item: -item[1]["ratio"]):
        rows.append(["Q2 chart transfer", f"{name} ({row['n']})", "Chart MAE over same-kind MAE", f"{row['ratio']:.2f} times", "At most 1.2 times", "Adequate" if row["transfer_adequate"] else "Not adequate"])
    rows.append(["Q2 chart transfer", "Natural objects, all", "Chart MAE over pooled MAE", f"{s['q2']['natural_all']['ratio']:.2f} times", "At most 1.2 times", "Adequate" if s["q2"]["natural_all"]["transfer_adequate"] else "Not adequate"])
    rows.append(["Q2 chart transfer", "Mushrooms", "Chart MAE over within-set MAE", f"{s['q2']['mushrooms']['ratio']:.2f} times", "At most 1.2 times", "Adequate" if s["q2"]["mushrooms"]["transfer_adequate"] else "Not adequate"])
    for dataset in ("natural", "mushrooms"):
        for light, row in s["q3"][dataset]["lights"].items():
            for key, label in (("white_patch", "White card"), ("chart_under_test_light", "Chart under that light")):
                reduction = row[f"{key}_reduction"]
                rows.append(["Q3 changed light", f"{DATASET_LABELS[dataset]}, {light}", f"{label} against no correction", drop_text(reduction), "At least 20%, interval above 0", "Met" if reduction["helps"] else "Not met"])
    for block, calibration in (("primary", "similar objects"), ("chart_calibrated", "chart only")):
        for dataset in ("natural", "mushrooms"):
            for name, label in ARM_LABELS.items():
                if name == "flash":
                    continue
                reduction = s["q4"][block][dataset][name]["reduction_vs_flash"]
                rows.append(["Q4 extra lights", f"{DATASET_LABELS[dataset]}, {calibration}", f"{label} against flash", drop_text(reduction), "At least 20%, interval above 0", "Met" if reduction["helps"] else "Not met"])
    for name, row in s["q4"]["sensitivity"].items():
        rows.append(["Q4 sensitivity", "Natural objects", f"Flash plus screen against flash, {name}", drop_text(row["reduction"]), "At least 20%, interval above 0", "Met" if row["reduction"]["helps"] else "Not met"])
    return {"title": "Every pre-stated test, whatever the result",
            "note": "Drops are the median per-sample drop in MAE as a percent of the baseline median, with a 95% cluster bootstrap interval. Q1 uses the drop in the median itself, as its rule is written. A negative drop means the arm did worse.",
            "headers": ["Question", "Test set", "Comparison", "Result (95% interval)", "Rule", "Outcome"], "rows": rows}


def build(s, c):
    natural = s["q1"]["natural"]
    mixed = s["q1"]["mixed"]
    mushrooms = s["q1"]["mushrooms"]
    repeat = s["q1"]["repeat_measurement"]
    q2 = s["q2"]
    q3 = s["q3"]["natural"]
    q4 = s["q4"]
    q5 = s["q5"]
    coverage = c["sessions"]["session share 36%"]
    original = next(row for row in run_synthetic()["scenarios"] if row["id"] == "matched")
    lights = q3["lights"]
    none_shares = [row["none"]["share_within_targets"] for row in lights.values()]
    white_shares = [row["white_patch"]["share_within_targets"] for row in lights.values()]
    chart_shares = [row["chart_under_test_light"]["share_within_targets"] for row in lights.values()]
    primary = q4["primary"]["natural"]
    chart_arms = q4["chart_calibrated"]["natural"]
    mushroom_primary = q4["primary"]["mushrooms"]
    mushroom_chart = q4["chart_calibrated"]["mushrooms"]
    flash, flash_screen = primary["flash"], primary["flash_plus_screen_rgb"]
    screen_gain = flash_screen["reduction_vs_flash"]
    sensitivity = q4["sensitivity"]
    pair = q5["example_pair"]
    domains = sorted(q2["domains"].items(), key=lambda item: -item[1]["ratio"])
    by_sessions = {row["n_sessions"]: row for row in coverage}
    specimen_rows = [row["specimen_bootstrap_coverage"] for row in coverage]
    affine_share = natural["affine"]["share_interval"]
    chart_share = q2["natural_all"]["chart"]["share_interval"]
    plain_light = {"D65 daylight": "daylight", "A incandescent": "incandescent light", "LED-B1 warm white": "warm white LED", "LED-B5 cool white": "cool white LED", "FL2 fluorescent": "fluorescent light", "Green shade (illustrative)": "green shade"}
    white_misses = [plain_light[name] for name, row in lights.items() if not row["white_patch_reduction"]["helps"]]
    name_a, name_b = "Purple plum (Prunus domestica)", "Black tourmaline (stone)"
    if pair["a"]["object"] != "Prunus domestica":
        name_a, name_b = name_b, name_a
    sens = lambda key: sensitivity[key]["reduction"]["percent"]

    data = {
        "title": "How far can a phone camera go toward a spectrum?",
        "eyebrow": "FINDINGS · SIMULATION ON MEASURED REFLECTANCE",
        "answer": [
            "A phone camera records three numbers per pixel, so it cannot measure a 27-band spectrum directly. This study asks how well a calibrated camera could estimate one, by simulating cameras on real measured reflectance. It used a DSLR's measured color curves, because no recent iPhone curve was available.",
            f"With calibration on hundreds of similar objects measured by spectrometer, {pct(natural['affine']['share_within_targets'])}% of 532 samples of natural objects met both draft targets. With only a 24-patch color chart, {pct(q2['natural_all']['chart']['share_within_targets'])}% did. When the light changed and nothing corrected for it, {pct(min(none_shares))}–{pct(max(none_shares))}% did, and a white reference card brought that back to {pct(min(white_shares))}–{pct(max(white_shares))}%. Using a screen as an extra light added almost nothing at the median.",
        ],
        "safety": "No phone photo and no wild fungus was tested. Never use this to decide whether a mushroom is safe to eat.",
        "tiles": [
            {"value": f"{pct(natural['affine']['share_within_targets'])}%", "label": SIMILAR_LABEL, "note": f"95% interval {pct(affine_share['low'])}–{pct(affine_share['high'])}%"},
            {"value": f"{pct(q2['natural_all']['chart']['share_within_targets'])}%", "label": "Calibrated on a 24-patch color chart", "note": f"95% interval {pct(chart_share['low'])}–{pct(chart_share['high'])}%"},
            {"value": f"{pct(min(none_shares))}–{pct(max(none_shares))}%", "label": "Light changed, no correction", "note": "Range across 6 lights"},
            {"value": f"{pct(min(white_shares))}–{pct(max(white_shares))}%", "label": "Light changed, white card", "note": "Range across 6 lights"},
        ],
        "tiles_note": "Share of held-out samples meeting both draft targets: spectral angle at most 10° and mean absolute error (MAE) at most 0.05. Angle measures the spectrum's shape. MAE measures reflectance error, so 0.05 is 5 percentage points. The targets are placeholders, not validated limits.",
        "findings": [
            {
                "id": "light",
                "title": "In this simulation, changed light hurt most, and a white card recovered most of the loss",
                "body": [
                    f"Calibrating on a chart under flash and then shooting under other light dropped the share meeting both targets from {pct(q3['same_light']['share_within_targets'])}% to {pct(min(none_shares))}–{pct(max(none_shares))}%. A gain correction from one white patch in the same scene brought it back to {pct(min(white_shares))}–{pct(max(white_shares))}%. Photographing the whole chart under that light reached {pct(min(chart_shares))}–{pct(max(chart_shares))}%.",
                    f"Against the study's 20% bar, the white card cleared it under {sum(row['white_patch_reduction']['helps'] for row in lights.values())} of 6 lights. It missed under {' and '.join(white_misses)}, where there was less to recover. The full chart cleared it under all 6.",
                ],
                "caveat": "This assumes one light falling evenly on the card and the object. Dappled light under a canopy is not modeled, and the green shade is an illustrative stand-in.",
                "chart": {
                    "type": "dots", "title": "Share of samples meeting both targets, by light", "unit": "%", "digits": 0, "min": 0, "max": 100,
                    "series": [{"key": "none", "label": "No correction"}, {"key": "white", "label": "White card"}, {"key": "chart", "label": "Chart under that light"}],
                    "rows": [{"label": name.replace(" (illustrative)", ""), "values": {"none": 100 * row["none"]["share_within_targets"], "white": 100 * row["white_patch"]["share_within_targets"], "chart": 100 * row["chart_under_test_light"]["share_within_targets"]}} for name, row in lights.items()],
                    "reference": {"value": 100 * q3["same_light"]["share_within_targets"], "label": "Same light as calibration"},
                },
            },
            {
                "id": "real",
                "title": "Measured spectra are harder than the first demo suggested",
                "body": [
                    f"The app's first version reported a median spectral angle of {original['median_sam_deg']:.2f}° because it tested spectra generated by the same model its solver assumes. On measured spectra, with a real camera's measured sensitivity, the same kind of fit gives {natural['affine']['sam_median']:.2f}° on natural objects and {mixed['affine']['sam_median']:.2f}° on an exploratory mixed set of 190 samples.",
                    f"The typical error is small, but the tail is long: the worst natural-object sample reached {natural['affine']['sam_worst']:.0f}°.",
                ],
                "chart": {
                    "type": "bars", "title": "Median spectral angle, one photo under flash", "unit": "°", "digits": 2,
                    "rows": [{"label": "First demo: generated spectra", "value": original["median_sam_deg"]}, {"label": "Natural objects, measured", "value": natural["affine"]["sam_median"]}, {"label": "Mixed set, measured", "value": mixed["affine"]["sam_median"]}],
                },
            },
            {
                "id": "camera",
                "title": "In this comparison, the camera's color curves moved results more than the model did",
                "body": [
                    f"With the Nikon D5100's measured sensitivities and a linear fit, {pct(natural['affine']['share_within_targets'])}% of samples met both targets. A nonlinear fit raised that to {pct(natural['kernel']['share_within_targets'])}%. The Sigma SD Merrill's curves lowered it to {pct(natural['sigma_camera']['share_within_targets'])}%. A constant spectrum, the baseline to beat, managed {pct(natural['constant_baseline']['share_within_targets'])}%.",
                    "This is only two cameras, and neither is a phone. The Sigma uses a layered Foveon sensor, so treat it as a range check. Measured curves for phones up to the iPhone 12 Pro Max are published but not used here yet, and none was found for newer iPhones, so where a current iPhone lands is unknown.",
                ],
                "chart": {
                    "type": "bars", "title": "Share of samples meeting both targets", "unit": "%", "digits": 0, "max": 100,
                    "rows": [{"label": "Constant spectrum", "value": 100 * natural["constant_baseline"]["share_within_targets"]}, {"label": "Sigma SD Merrill, linear fit", "value": 100 * natural["sigma_camera"]["share_within_targets"]}, {"label": "Nikon D5100, linear fit", "value": 100 * natural["affine"]["share_within_targets"]}, {"label": "Nikon D5100, nonlinear fit", "value": 100 * natural["kernel"]["share_within_targets"]}],
                },
            },
            {
                "id": "limit",
                "title": "A nonlinear model does better, but whether three numbers are a hard limit is unresolved",
                "body": [
                    f"The study's test for a hard limit was whether a nonlinear model lowers median MAE by more than 10%. On natural objects it did so by {natural['nonlinear_gain']['percent']:.0f}%, but the 95% interval ({natural['nonlinear_gain']['low']:.0f}–{natural['nonlinear_gain']['high']:.0f}%) spans that line, so this stays unresolved. On the exploratory mixed set the gain was {mixed['nonlinear_gain']['percent']:.0f}% ({mixed['nonlinear_gain']['low']:.0f}–{mixed['nonlinear_gain']['high']:.0f}%). On mushrooms it was {mushrooms['nonlinear_gain']['percent']:.0f}%.",
                    f"A nonlinear fit on one photo also beat a linear fit on four photos at the median ({natural['kernel']['mae_median']:.4f} against {flash_screen['mae_median']:.4f}). The chart's lower bars show the best linear summaries for scale. They don't bound what a nonlinear model could do. The repeat error is how much the reference measurements themselves differ, taken in similar regions of each object.",
                ],
                "chart": {
                    "type": "bars", "title": "Median MAE on natural objects, lower is better", "digits": 4,
                    "rows": [{"label": "Camera, linear fit", "value": natural["affine"]["mae_median"]}, {"label": "Camera, nonlinear fit", "value": natural["kernel"]["mae_median"]}, {"label": "Best 3 numbers, known exactly", "value": natural["pca_floor"]["3"]}, {"label": "Repeat-measurement error", "value": repeat["mae_median"]}, {"label": "Best 5 numbers, known exactly", "value": natural["pca_floor"]["5"]}],
                },
            },
            {
                "id": "lights",
                "title": "Using a screen as extra light added almost nothing at the median",
                "body": [
                    f"Each screen color was exposed so its strongest channel matched flash on a white target, with three times the noise because a screen is dimmer. Adding three screen colors to the flash changed median MAE by {drop_text(screen_gain)}. The tail did improve: 90th percentile MAE fell from {flash['mae_p90']:.3f} to {flash_screen['mae_p90']:.3f}, and the share meeting both targets rose from {pct(flash['share_within_targets'])}% to {pct(flash_screen['share_within_targets'])}%. With chart-only calibration the drop was {drop_text(chart_arms['flash_plus_screen_rgb']['reduction_vs_flash'])}, and the share rose from {pct(chart_arms['flash']['share_within_targets'])}% to {pct(chart_arms['flash_plus_screen_rgb']['share_within_targets'])}%. No arm cleared the 20% bar on natural objects.",
                    f"Screen colors alone did worse than flash ({drop_text(primary['screen_rgb']['reduction_vs_flash'])}). Steady daylight as a second light gave {drop_text(primary['flash_plus_ambient']['reduction_vs_flash'])}. Three flash brightness levels gave {drop_text(primary['flash_three_brightness']['reduction_vs_flash'])}; they add no independent measurements, so that gain comes from averaging noise.",
                    f"On mushrooms, brightness levels ({mushroom_primary['flash_three_brightness']['reduction_vs_flash']['percent']:.0f}%) and steady daylight ({mushroom_primary['flash_plus_ambient']['reduction_vs_flash']['percent']:.0f}%) cleared the bar with calibration on similar samples. With a chart alone, the daylight and screen arms made mushrooms worse ({mushroom_chart['flash_plus_ambient']['reduction_vs_flash']['percent']:.0f}%, {mushroom_chart['screen_rgb']['reduction_vs_flash']['percent']:.0f}% and {mushroom_chart['flash_plus_screen_rgb']['reduction_vs_flash']['percent']:.0f}%).",
                    f"What moved the natural-object result: lower noise gave {sens('noise: low'):.0f}%, screen photos as clean as flash gave {sens('screen noise x1'):.0f}%, five times the screen noise gave {sens('screen noise x5'):.0f}%, narrow screen colors gave {sens('screen: narrow'):.0f}%, wide ones {sens('screen: wide'):.0f}%, and the measured curves of an older Apple display {sens('screen: Apple Studio Display (measured)'):.0f}%. Only the Sigma camera's curves cleared the bar ({sens('camera: Sigma SD Merrill'):.0f}%).",
                ],
                "caveat": "Screen colors other than that older display are illustrative, because no measured iPhone screen curve was available. Only a real phone test can settle this.",
                "chart": {
                    "type": "dots", "title": "Share of samples meeting both targets, by how the photos were taken", "unit": "%", "digits": 0, "min": 0, "max": 100,
                    "series": [{"key": "similar", "label": "Calibrated on similar objects"}, {"key": "chart", "label": "Color chart only"}],
                    "rows": [{"label": ARM_LABELS[name], "values": {"similar": 100 * primary[name]["share_within_targets"], "chart": 100 * chart_arms[name]["share_within_targets"]}} for name in ARM_LABELS],
                },
                "table": {
                    "title": "Capture arms, calibrated on similar objects",
                    "headers": ["Capture", "Numbers recorded", "Independent directions", "Median MAE", "90th percentile MAE", "Worst MAE", "Median per-sample drop in MAE (95% interval)"],
                    "rows": [[ARM_LABELS[name], primary[name]["channels"], primary[name]["independent_directions"], f"{primary[name]['mae_median']:.4f}", f"{primary[name]['mae_p90']:.3f}", f"{primary[name]['mae_worst']:.3f}",
                              "Baseline" if name == "flash" else drop_text(primary[name]["reduction_vs_flash"])] for name in ARM_LABELS],
                },
            },
            {
                "id": "chart",
                "title": "A color chart transfers unevenly",
                "body": [
                    f"Calibrating on the chart alone raised median MAE by {100 * (q2['natural_all']['ratio'] - 1):.0f}% overall, past the study's 20% limit. It held up on flowers and fruits. On leaves the error was {q2['domains']['Leaves']['ratio']:.1f} times that of a calibration on leaves, and only {pct(q2['domains']['Leaves']['chart']['share_within_targets'])}% of leaf samples met both targets, against {pct(q2['domains']['Leaves']['same_domain']['share_within_targets'])}%. On bark the error was {q2['domains']['Barks']['ratio']:.1f} times, and on mushrooms {q2['mushrooms']['ratio']:.1f} times.",
                ],
                "caveat": "Beetles and vegetables have only 9 samples each, so their same-kind results are unstable.",
                "chart": {
                    "type": "dots", "title": "Median MAE by kind of object, lower is better", "digits": 3, "min": 0,
                    "series": [{"key": "same", "label": "Calibrated on the same kind"}, {"key": "chart", "label": "Color chart only"}],
                    "rows": [{"label": f"{name} ({row['n']})", "values": {"same": row["same_domain"]["mae_median"], "chart": row["chart"]["mae_median"]}} for name, row in domains],
                },
            },
            {
                "id": "mushrooms",
                "title": "The mushroom data cannot answer the mushroom question",
                "body": [
                    f"The only public mushroom reflectance data found covers 250 samples. The source does not name the species, and dry matter runs from {pct(s['data']['mushrooms']['dry_matter_min'])}% to {pct(s['data']['mushrooms']['dry_matter_max'])}%. Their spectra barely change shape. One number gets a median MAE of {mushrooms['pca_floor']['1']:.4f}, and a constant spectrum already meets the angle target on every sample (median {mushrooms['constant_baseline']['sam_median']:.2f}°).",
                    f"So the {pct(mushrooms['affine']['share_within_targets'])}% meeting both targets here says little. Wild fungi with varied pigments remain untested.",
                ],
            },
            {
                "id": "metamer",
                "title": "Look-alikes are real but rare, and mostly dark",
                "body": [
                    f"Of {q5['pairs_from_independent_groups']:,} pairs of samples from different species or specimens, {q5['look_alike_pairs']:,} looked alike under flash at this noise level, and {q5['look_alike_above_a_draft_target']} of those differed by more than a draft target. Most of that is shape in dark patches: {q5['above_target_by_angle_only']} exceeded the angle target alone, only {q5['above_target_by_mae']} exceeded the MAE target, and {q5['of_those_with_a_dark_patch']} include a patch with mean reflectance below 0.15. Screen light would separate {q5['of_those_separated_by_screen_light']} of the {q5['look_alike_above_a_draft_target']}.",
                    f"An example: a purple plum and a black tourmaline stone. Their simulated flash responses are nearly the same, yet their spectra differ in shape by {pair['sam_deg']:.0f}°. The difference in reflectance is small, {pair['mae']:.3f} on average.",
                ],
                "chart": {
                    "type": "lines", "title": "Measured reflectance of two look-alike samples", "xLabel": "Wavelength (nm)", "digits": 3, "yMin": 0,
                    "x": pair["bands_nm"],
                    "series": [{"label": name_a, "values": pair["spectrum_a"]}, {"label": name_b, "values": pair["spectrum_b"]}],
                },
            },
            {
                "id": "intervals",
                "title": "The app's first “95%” intervals were not 95%",
                "body": [
                    f"An interval labeled 95% from the original session bootstrap contained the truth about {pct(by_sessions[2]['session_bootstrap_coverage'])}% of the time with 2 sessions, and {pct(by_sessions[3]['session_bootstrap_coverage'])}% with 3. Ignoring sessions, the original default, stayed near {pct(sum(specimen_rows) / len(specimen_rows))}% however many sessions were added.",
                    "The app now uses exact order-statistic intervals that state their true level: 75% for 3 sessions, and at least 95% from 6 sessions. They are wider, and that is the price of few sessions.",
                ],
                "caveat": "Simulated with 5 specimens per session and sessions making up 36% of the variance. Other scenarios are in results/coverage.json.",
                "chart": {
                    "type": "lines", "title": "How often an interval contained the truth", "xLabel": "Number of sessions", "unit": "%", "digits": 0, "yMin": 40, "yMax": 100, "ordinal": True,
                    "x": [row["n_sessions"] for row in coverage],
                    "series": [{"label": "Exact interval (states its level)", "values": [100 * row["exact_coverage"] for row in coverage]},
                               {"label": "Session bootstrap (labeled 95%)", "values": [100 * row["session_bootstrap_coverage"] for row in coverage]},
                               {"label": "Ignoring sessions (labeled 95%)", "values": [100 * row["specimen_bootstrap_coverage"] for row in coverage]}],
                    "reference": {"value": 95, "label": "95%"},
                },
            },
        ],
        "all_tests": all_tests_table(s),
        "limits_title": "What this does not show",
        "limits": [
            "No phone photo was used. RGB values are simulated from the measured curves of two other cameras.",
            "The noise model is simple. It leaves out clipping, tone mapping, uneven light, gloss and viewing angle, so real results should be expected to be worse.",
            "Calibrating on similar objects takes hundreds of spectrometer readings, which most people do not have.",
            "The spectra come from specific collections: natural objects from one region of Argentina, and mushrooms of an unnamed species. Other materials may behave differently.",
            "The draft targets are placeholders. Meeting them is not evidence of fitness for any purpose.",
            "No human expert has reviewed this work.",
            "Nothing here identifies a species, and nothing here supports a decision about eating anything.",
            "The first idea behind this project included 3D capture. No 3D work is reported here; the code contains only a tested point-projection function.",
        ],
        "next_title": "What would settle more of this",
        "next": [
            "Run the phone test. It needs no purchase and measures how stable a real phone is. It may also show whether screen light adds real measurements, though the simulation says that needs many objects.",
            "Repeat this study with measured phone sensitivity curves.",
            "Pair phone photos with spectrometer readings of the same marked spots, starting with stable matte targets. Only that can test spectral accuracy.",
        ],
        "plan_note": "The analysis plan was written before the natural-object and mushroom results were computed. It has no independent timestamp. A separate AI code review of the first full run found a flaw in how screen light was modeled, which erased most of the first reported gain. That and every other change are listed at the end of study/PLAN.md. No human expert has reviewed this work yet.",
        "sources": SOURCES,
        "metamer": {
            "bands_nm": pair["bands_nm"], "spectrum_a": pair["spectrum_a"], "spectrum_b": pair["spectrum_b"],
            "name_a": name_a, "name_b": name_b,
            "sam_deg": pair["sam_deg"], "mae": pair["mae"], "flash_response_a": pair["flash_response_a"], "flash_response_b": pair["flash_response_b"],
            "description": "Measured reflectance of a purple plum and a black tourmaline stone. Under a simulated flash and the study's Nikon sensitivity, their RGB values are nearly the same.",
        },
    }
    return data


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def chart_table(chart):
    unit = chart.get("unit", "")
    digits = chart.get("digits", 2)
    fmt = lambda v: f"{v:.{digits}f}{unit}"
    if chart["type"] == "bars":
        return markdown_table(["", "Value"], [[row["label"], fmt(row["value"])] for row in chart["rows"]])
    if chart["type"] == "dots":
        return markdown_table(["", *[s["label"] for s in chart["series"]]],
                              [[row["label"], *[fmt(row["values"][s["key"]]) for s in chart["series"]]] for row in chart["rows"]])
    if chart["type"] == "lines" and chart.get("ordinal"):
        return markdown_table([chart["xLabel"], *[s["label"] for s in chart["series"]]],
                              [[x, *[fmt(s["values"][i]) for s in chart["series"]]] for i, x in enumerate(chart["x"])])
    return ""


def write_markdown(data):
    parts = ["<!-- Generated by study/build_report.py. Do not edit by hand. -->", f"# {data['title']}", "", *[p + "\n" for p in data["answer"]],
             f"**{data['safety']}**", "", markdown_table(["Condition", "Share meeting both draft targets", ""], [[t["label"], t["value"], t["note"]] for t in data["tiles"]]), "", data["tiles_note"], ""]
    for number, finding in enumerate(data["findings"], start=1):
        parts += [f"## {number}. {finding['title']}", "", *[p + "\n" for p in finding["body"]]]
        if "chart" in finding:
            table = chart_table(finding["chart"])
            if table:
                parts += [f"*{finding['chart']['title']}*", "", table, ""]
        if "table" in finding:
            parts += [f"*{finding['table']['title']}*", "", markdown_table(finding["table"]["headers"], finding["table"]["rows"]), ""]
        if finding.get("caveat"):
            parts += [f"*Caveat: {finding['caveat']}*", ""]
    tests = data["all_tests"]
    parts += [f"## {tests['title']}", "", tests["note"], "", markdown_table(tests["headers"], tests["rows"]), ""]
    parts += [f"## {data['limits_title']}", "", *[f"- {item}" for item in data["limits"]], "", f"## {data['next_title']}", "", *[f"- {item}" for item in data["next"]], "",
              "## How this was done", "", data["plan_note"], "", "The plan is in [study/PLAN.md](../study/PLAN.md), the raw numbers are in [results/](../results/), and the data sources are in [DATA.md](DATA.md).", "",
              "## Sources", "", *[f"- [{item['label']}]({item['url']})" for item in data["sources"]], ""]
    (ROOT / "docs" / "FINDINGS.md").write_text("\n".join(parts), encoding="utf-8")


PHONE_TEST = """<!-- Generated by study/build_report.py. Do not edit by hand. -->
# Phone test: no spectrometer needed

A phone can answer two questions on its own. How stable are its photos from day to day? And do extra lights add independent measurements, or only more numbers?

This test cannot show spectral accuracy, which needs a reference instrument, and it says nothing about identifying fungi. The simulation study found little median gain from screen light, so treat the second question as an open check, not a promise.

## What you need

- A phone on a fixed stand, and a room you can make dark.
- A laptop or tablet to use as a colored light. The app's Phone test tab has a full-screen light panel.
- About 30 stable, matte objects with varied colors. The table below shows why fewer may not settle the second question. The stability measures work with fewer.

## Taking the photos

1. Mark where each object sits, so it goes back in the same place.
2. Use one rear camera. Lock focus and exposure if your camera allows it, and turn off anything that changes the look between shots, such as portrait or macro switching.
3. Photograph each object under each light: the phone's flash, then the screen showing red, green and blue. Keep the screen close and in the same position.
4. Take 3 photos per object and light.
5. Repeat the whole thing in at least 3 sessions, on different days or after restarting the setup. Keep the same objects: this crossed design is what the analysis expects.
6. Name each file by object, session, light and repeat, like `mug_s1_flash_r2.jpg` or `mug_s1_screen-red_r2.jpg`. The session must look like `s1` and the repeat like `r2`. Other names are skipped and listed.

Avoid clipping. If a box has over 1% clipped pixels, the app warns you, and you should lower the exposure or move the light back.

## Analyzing them

With the app installed (see the [README](../README.md)), run `.venv/bin/myco-fusion serve --open` and open the Phone test tab. Add your photos, drag a box over the object, and select Analyze. Photos stay on your computer.

Browsers give tone-mapped values, which suit a stability check. Tone curves can add directions that are not real measurements, so treat the independent-measurement count from JPEG or HEIC photos as exploratory. If you can extract linear values from RAW files with another tool, paste a table with the columns `object_id, session_id, light, repeat, r, g, b`. Commas or tabs both work. Safari can open HEIC photos; other browsers need JPEG or PNG. The photo tool has not been tried on real iPhone photos yet.

## Reading the report

- **Photo-to-photo SD** is the noise between repeats in one session.
- **Session SD** and **object × session SD** show drift between sessions. With 3 sessions these are rough, so more sessions help.
- **Repeatability coefficient:** two photos in one session should differ by less than this 95% of the time.
- **Reproducibility coefficient:** the same, across sessions.
- **ICC:** near 1 means objects differ far more than repeat photos do.
- **Independent measurements:** how many directions in the photo table are stronger than the strongest direction noise produces on its own. Flash alone tops out at 3. More than 3 with all lights together suggests the extra lights measured something new.

## What to expect

Simulated with known truth, {replicates} times per row: {design}. The controls use spectra restricted to three dimensions, so any count above 3 there is a false alarm.

{table}

No control run went above 3 here, but the test also often misses real extra directions, even with 30 objects. So seeing 3 with all lights does not prove the lights add nothing. The simulation uses illustrative camera and light curves, and other kinds of drift could still raise false alarms. Your phone's numbers are the point of running the test.
"""


def write_phone_doc():
    power = json.loads((ROOT / "results" / "phone_power.json").read_text())
    above = lambda cell: f"{100 * cell['share_above_3']:.0f}%"
    table = markdown_table(["Objects", "Flash alone", "All lights", "Control", "Control plus placement drift"],
                           [[row["n_objects"], above(row["flash_only"]), above(row["all_lights"]), above(row["control_three_dimensional"]), above(row["control_with_placement_drift"])] for row in power["rows"]])
    note = "Cells show the share of runs that counted more than 3 independent measurements."
    (ROOT / "docs" / "PHONE_TEST.md").write_text(PHONE_TEST.format(replicates=power["replicates"], design=power["design"], table=f"{table}\n\n{note}"), encoding="utf-8")


def write_site(data):
    SITE.mkdir(exist_ok=True)
    (SITE / "findings.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copyfile(APP / "findings.js", SITE / "findings.js")
    shutil.copyfile(APP / "styles.css", SITE / "styles.css")


def main():
    study = json.loads((ROOT / "results" / "study.json").read_text())
    coverage = json.loads((ROOT / "results" / "coverage.json").read_text())
    check_claims(study, coverage)
    data = build(study, coverage)
    PACKAGE_DATA.write_text(json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    write_markdown(data)
    write_phone_doc()
    write_site(data)
    print(f"Wrote {PACKAGE_DATA}, docs/FINDINGS.md, docs/PHONE_TEST.md and site/")


if __name__ == "__main__":
    main()
