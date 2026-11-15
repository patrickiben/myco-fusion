"""Component admission trial: the "Addendum: component admission trial" in study/PLAN.md.

The checks of the rule run first, on components whose value is known. The trial refuses to start
until their results exist:

    python -m study.admission_trial checks    # writes results/admission_rule_checks.json
    python -m study.admission_trial trial     # writes results/admission_trial.json

Both need colour-science for the camera and light curves (pip install ".[study]").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from myco_fusion.spectra import apply_tm, fit_smooth_tm
from myco_fusion.validation import DRAFT_TARGETS

from .pipeline import (ALPHAS, METRIC_NM, NOISE, SIM_NM, Noise, _inner_score, camera_curves, capture, errors,
                       fit_selected, gaussian_light, group_folds, light_curve, match_exposure, paired_reduction)

ROOT = Path(__file__).resolve().parent.parent
CHECKS_PATH = ROOT / "results" / "admission_rule_checks.json"
TRIAL_PATH = ROOT / "results" / "admission_trial.json"

SEED = 20261114
DRAWS = 20
REPLICATES = 200
RESAMPLES = 2000
INNER_FOLDS = 4  # the study's own penalty search (pipeline._inner_score)
DEV_FOLDS = 5
BETAS = (1e-4, 1e-2, 1.0, 100.0)
K_VALUES = tuple(range(3, 9))
PLANTED = (0.05, 0.1, 0.2, 0.5)
DUPLICATE_A = 0.5
DUPLICATE_SD = 0.005
ESTIMATORS = ("affine", "polynomial", "kernel", "smoothing", "forward_model")
CANDIDATES = ESTIMATORS[1:]
SCREEN_COLORS = {"blue": (458, 30), "green": (530, 45), "red": (625, 30)}  # the medium set of Q4
SCREEN_NOISE_FACTOR = 3.0
SHOT_NOISE = {"read_noise_sd": 0.002, "shot_noise_factor": 0.0065, "gain_error_sd": 0.02, "bit_depth": 12}
PUBLISHED_TARGETS = {"sam_deg": 10.0, "mae": 0.05}
ROUNDING = 1e-12  # per-sample MAE differences below this are floating-point rounding and count as zero
FROZEN_FILES = ("results/study.json", "results/coverage.json", "results/phone_power.json", "myco_fusion/data/findings.json")
CODE_FILES = ("study/admission_trial.py", "study/pipeline.py", "study/sources.py", "study/run_study.py",
              "myco_fusion/spectra.py", "myco_fusion/sensor.py", "myco_fusion/validation.py")
DATA_FILES = ("Database_Plants.csv", "Database_Animals.csv", "Database_Stones.csv", "phone_curves/iPhone_12_Pro_Max.txt")

METRIC_INDEX = np.searchsorted(SIM_NM, METRIC_NM)
if not np.allclose(SIM_NM[METRIC_INDEX], METRIC_NM):
    raise ImportError("the 27 metric bands must lie on the simulation grid")


def rng_for(*names):
    return np.random.default_rng([SEED, *[zlib.crc32(str(name).encode()) for name in names]])


# ----------------------------------------------------------------------------- inputs and photos

@dataclass
class Inputs:
    ids: list
    sim: np.ndarray       # spectra on the 5 nm simulation grid, 400 to 700 nm
    metric: np.ndarray    # the same spectra on the 27 metric bands
    groups: np.ndarray
    camera: np.ndarray
    flash: np.ndarray
    phone_camera: np.ndarray
    screens: dict
    noise: Noise

    def operator(self, kind):
        camera = self.phone_camera if kind == "forward_model_wrong_camera" else self.camera
        return camera * self.flash


def load_inputs() -> Inputs:
    from .run_study import prepare
    from .sources import load_colour_tables, load_natural_objects, load_phone_curves

    tables = load_colour_tables()
    raw = load_natural_objects()
    sim, metric, _ = prepare(raw["wavelengths"], raw["spectra"])
    if not np.array_equal(sim[:, METRIC_INDEX], metric):
        raise ValueError("simulation and metric spectra disagree on shared bands")
    camera = camera_curves(*tables["cameras"]["Nikon D5100"])
    flash = light_curve(*tables["illuminants"]["LED-B4"])
    phone_camera = camera_curves(*load_phone_curves()["iPhone 12 Pro Max"])
    screens = {color: match_exposure(gaussian_light(*shape), flash, camera) for color, shape in SCREEN_COLORS.items()}
    return Inputs(list(raw["ids"]), sim, metric, np.asarray(raw["group"]), camera, flash, phone_camera, screens,
                  NOISE["moderate"])


def photos(inputs, light, rng, factor=1.0, shot=False):
    """One photo per sample: the study's additive noise, or signal-dependent noise for the shot-noise check."""
    if not shot:
        return capture(inputs.sim, light, inputs.camera, inputs.noise, rng, factor)
    from myco_fusion.sensor import RealisticNoise, simulate_sensor

    params = dict(SHOT_NOISE)
    params["read_noise_sd"] *= factor
    params["shot_noise_factor"] *= factor
    clean = (inputs.sim * light) @ inputs.camera.T
    return simulate_sensor(clean, RealisticNoise(**params), rng)


def sample_mae(truth, estimate):
    return errors(truth, estimate)[1]


def within_targets(truth, estimate):
    angle, error = errors(truth, estimate)
    return (angle <= DRAFT_TARGETS["sam_deg"]) & (error <= DRAFT_TARGETS["mae"])


# ----------------------------------------------------------------------------- estimators

class SmoothRidge:
    """Affine ridge with a penalty on second differences across bands (``fit_smooth_tm`` as written)."""

    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta

    def fit(self, x, y):
        self.weights = fit_smooth_tm(x, y, alpha=self.alpha, beta=self.beta)
        return self

    def predict(self, x):
        return apply_tm(x, self.weights)


class ForwardModel:
    """The most probable spectrum given one photo, the camera and light curves, and a description of spectra.

    Spectra are described by the training mean and the first ``k`` principal directions on the
    simulation grid, and the photo is predicted through ``operator`` (camera times light). The part
    of a spectrum outside the k directions is treated as extra noise, and so are the additive noise
    and the gain error, the latter as a share of the observed photo common to its three channels.
    Under this Gaussian description, the most probable spectrum is the posterior mean.
    """

    def __init__(self, k, operator, noise):
        self.k, self.operator, self.noise = int(k), np.asarray(operator, dtype=float), noise

    def fit(self, spectra_sim):
        s = np.asarray(spectra_sim, dtype=float)
        self.mean = s.mean(axis=0)
        cov = np.cov(s, rowvar=False)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1][: self.k]
        self.basis, self.variances = vectors[:, order], np.maximum(values[order], 0.0)
        residual = cov - (self.basis * self.variances) @ self.basis.T
        mapped = self.operator @ self.basis
        self.gain_matrix = mapped * self.variances
        self.fixed = (self.gain_matrix @ mapped.T + self.operator @ residual @ self.operator.T
                      + self.noise.additive ** 2 * np.eye(3))
        self.photo_mean = self.operator @ self.mean
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        cov = self.fixed[None] + self.noise.gain ** 2 * x[:, :, None] * x[:, None, :]
        innovation = np.linalg.solve(cov, (x - self.photo_mean)[..., None])[..., 0]
        return (self.mean + (innovation @ self.gain_matrix) @ self.basis.T)[:, METRIC_INDEX]


def select_smoothing(x, y, groups, rng):
    candidates = [lambda a=a, b=b: SmoothRidge(a, b) for a in ALPHAS for b in BETAS]
    scores = [_inner_score(x, y, groups, build, rng, INNER_FOLDS) for build in candidates]
    return candidates[int(np.argmin(scores))]().fit(x, y)


def select_forward(x, spectra_sim, groups, rng, operator, noise):
    """Choose k by mean MAE on the 27 bands over the study's four folds, then fit on all rows."""
    scores = []
    for k in K_VALUES:
        folds = group_folds(groups, INNER_FOLDS, rng)
        fold_errors = []
        for fold in np.unique(folds):
            train, test = folds != fold, folds == fold
            if train.sum() < 2:
                continue
            estimate = ForwardModel(k, operator, noise).fit(spectra_sim[train]).predict(x[test])
            fold_errors.append(np.mean(np.abs(estimate - spectra_sim[test][:, METRIC_INDEX])))
        scores.append(np.mean(fold_errors) if fold_errors else np.inf)
    return ForwardModel(K_VALUES[int(np.argmin(scores))], operator, noise).fit(spectra_sim)


def fit_predict(kind, inputs, train, test, x, rng):
    """Fit one estimator on the ``train`` rows of photos ``x`` and predict the ``test`` rows."""
    y, s, g = inputs.metric[train], inputs.sim[train], inputs.groups[train]
    if kind in ("affine", "polynomial", "kernel"):
        model = fit_selected(x[train], y, g, rng, "ridge" if kind == "affine" else kind)
    elif kind == "smoothing":
        model = select_smoothing(x[train], y, g, rng)
    elif kind.startswith("forward_model"):
        model = select_forward(x[train], s, g, rng, inputs.operator(kind), inputs.noise)
    else:
        raise ValueError(kind)
    return model.predict(x[test])


def stack(kinds, inputs, x, dev, held, dev_folds, key):
    """Out-of-fold development predictions, and held-out predictions from fits on all of development."""
    out_of_fold, held_out = {}, {}
    for kind in kinds:
        prediction = np.empty((len(dev), len(METRIC_INDEX)))
        for fold in np.unique(dev_folds):
            inside = dev_folds == fold
            prediction[inside] = fit_predict(kind, inputs, dev[~inside], dev[inside], x, rng_for(*key, kind, "fold", int(fold)))
        out_of_fold[kind] = prediction
        held_out[kind] = fit_predict(kind, inputs, dev, held, x, rng_for(*key, kind, "development"))
    return out_of_fold, held_out


# ----------------------------------------------------------------------------- combination and statistics

def simplex_weights(predictions, target):
    """Least-squares weights that are at least zero and sum to one, by exact search over subsets."""
    columns = np.column_stack([np.asarray(p, dtype=float).ravel() for p in predictions])
    t = np.asarray(target, dtype=float).ravel()
    gram, cross, scale = columns.T @ columns, columns.T @ t, float(t @ t)
    m = columns.shape[1]
    best, best_loss = None, np.inf
    for size in range(1, m + 1):
        for subset in combinations(range(m), size):
            index = list(subset)
            kkt = np.zeros((size + 1, size + 1))
            kkt[:size, :size] = gram[np.ix_(index, index)]
            kkt[:size, size] = kkt[size, :size] = 1.0
            solution = np.linalg.lstsq(kkt, np.append(cross[index], 1.0), rcond=None)[0][:size]
            if solution.min() < -1e-10 or abs(solution.sum() - 1.0) > 1e-8:
                continue
            weights = np.zeros(m)
            weights[index] = np.clip(solution, 0.0, None)
            weights /= weights.sum()
            loss = weights @ gram @ weights - 2 * cross @ weights
            if loss < best_loss - 1e-12 * scale:
                best, best_loss = weights, loss
    return best


def combine(preds, names, y_dev):
    """Weights fitted on development out-of-fold predictions, and the combined held-out prediction."""
    weights = simplex_weights([preds[n][0] for n in names], y_dev)
    return dict(zip(names, weights.tolist())), sum(w * preds[n][1] for w, n in zip(weights, names))


def corrupt(pair, how, dev_mean, rng):
    dev_pred, held_pred = pair
    if how == "shuffled":
        return dev_pred[rng.permutation(len(dev_pred))], held_pred[rng.permutation(len(held_pred))]
    if how == "reflected":
        return 2 * dev_mean - dev_pred, 2 * dev_mean - held_pred
    raise ValueError(how)


def draw_errors(preds, y_dev, y_held, removals, corrupted, rng):
    """Held-out MAE per sample: the full combination, each removal and each corruption refitted, each component alone."""
    names = list(preds)
    dev_mean = y_dev.mean(axis=0)
    weights, combined = combine(preds, names, y_dev)
    out = {"combination": sample_mae(y_held, combined)}
    for removal in removals:
        if all(weights[n] == 0.0 for n in removal):
            # The full optimum already gives these components no weight, so it is also an optimum without them.
            out["without:" + "+".join(removal)] = out["combination"].copy()
            continue
        kept = [n for n in names if n not in removal]
        out["without:" + "+".join(removal)] = sample_mae(y_held, combine(preds, kept, y_dev)[1])
    for group in corrupted:
        for how in ("shuffled", "reflected"):
            changed = dict(preds)
            for n in group:
                changed[n] = corrupt(preds[n], how, dev_mean, rng)
            out[f"{how}:" + "+".join(group)] = sample_mae(y_held, combine(changed, names, y_dev)[1])
    for n in names:
        out["alone:" + n] = sample_mae(y_held, preds[n][1])
    return out, weights, combined


class GroupBootstrap:
    """One set of resamples of held-out groups, shared by every contrast it is asked for."""

    def __init__(self, groups, rng, resamples=RESAMPLES):
        names, self.index = np.unique(np.asarray(groups), return_inverse=True)
        count = len(names)
        draws = rng.integers(0, count, size=(resamples, count))
        self.counts = np.zeros((resamples, count))
        np.add.at(self.counts, (np.repeat(np.arange(resamples), count), draws.ravel()), 1.0)
        self.sizes = np.bincount(self.index, minlength=count).astype(float)
        self.denominator = self.counts @ self.sizes

    def contrast(self, difference):
        """Mean paired difference, its percentile 95% interval, and the one-sided p-value for a difference above zero."""
        difference = without_rounding(difference)
        sums = np.bincount(self.index, weights=difference, minlength=len(self.sizes))
        boot = (self.counts @ sums) / self.denominator
        low, high = np.percentile(boot, [2.5, 97.5])
        return {"mean": float(difference.mean()), "low": float(low), "high": float(high),
                "p_one_sided": float((1 + np.sum(boot <= 0)) / (len(boot) + 1))}


def without_rounding(difference):
    """Per-sample differences, with floating-point rounding (below ``ROUNDING``) set to exactly zero."""
    difference = np.asarray(difference, dtype=float)
    return np.where(np.abs(difference) < ROUNDING, 0.0, difference)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def verdicts(maes, tested, family, boot, frozen_ok=True):
    """Conditions 2 to 4 of the rule for each tuple in ``tested``; Holm runs over ``family`` plus ``tested``."""
    tests = list(dict.fromkeys([*family, *tested]))
    unique = {t: boot.contrast(maes["without:" + "+".join(t)] - maes["combination"]) for t in tests}
    adjusted = dict(zip(tests, holm([unique[t]["p_one_sided"] for t in tests])))
    table = {}
    for t in tested:
        label = "+".join(t)
        shuffled = float(np.mean(without_rounding(maes["shuffled:" + label] - maes["combination"])))
        reflected = float(np.mean(without_rounding(maes["reflected:" + label] - maes["combination"])))
        simpler = {n: boot.contrast(maes["alone:affine"] - maes["alone:" + n]) for n in t}
        condition_2 = bool(unique[t]["low"] > 0 and adjusted[t] < 0.05)
        condition_3 = bool(shuffled >= 0 and reflected >= 0)
        condition_4 = bool(all(s["low"] > 0 for s in simpler.values()))
        table[label] = {"unique_gain": unique[t], "holm_p": float(adjusted[t]),
                        "shuffled_minus_real": shuffled, "reflected_minus_real": reflected, "better_than_affine": simpler,
                        "condition_1": bool(frozen_ok), "condition_2": condition_2, "condition_3": condition_3,
                        "condition_4": condition_4, "passes": bool(frozen_ok and condition_2 and condition_3 and condition_4)}
    family_p = {"+".join(t): {"p_one_sided": unique[t]["p_one_sided"], "holm_p": float(adjusted[t])}
                for t in tests if t not in tested}
    return table, family_p


# ----------------------------------------------------------------------------- checks of the rule

def check_replicate(r, inputs):
    """One replicate: its own split of groups and one noise draw, the five estimators, and the added components."""
    split = group_folds(inputs.groups, 2, rng_for("check-split", r))
    dev, held = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    dev_folds = group_folds(inputs.groups[dev], DEV_FOLDS, rng_for("check-development-folds", r))
    x = photos(inputs, inputs.flash, rng_for("check-photos", r))
    oof, out = stack(ESTIMATORS, inputs, x, dev, held, dev_folds, ("check", r))
    y_dev, y_held = inputs.metric[dev], inputs.metric[held]
    rng = rng_for("check-added", r)
    a_dev, a_held = oof["affine"], out["affine"]
    added = {"useless": (a_dev[rng.permutation(len(dev))], a_held[rng.permutation(len(held))])}
    for a in PLANTED:
        added[f"planted_{a:g}"] = (a_dev + a * (y_dev - a_dev), a_held + a * (y_held - a_held))
    for name in ("duplicate_1", "duplicate_2"):
        added[name] = (a_dev + DUPLICATE_A * (y_dev - a_dev) + rng.normal(0, DUPLICATE_SD, a_dev.shape),
                       a_held + DUPLICATE_A * (y_held - a_held) + rng.normal(0, DUPLICATE_SD, a_held.shape))
    boot = GroupBootstrap(inputs.groups[held], rng_for("check-bootstrap", r))
    family = [(c,) for c in CANDIDATES]
    configs = {"useless": [("useless",)], **{f"planted_{a:g}": [(f"planted_{a:g}",)] for a in PLANTED},
               "duplicates": [("duplicate_1",), ("duplicate_2",)]}
    record = {"replicate": r, "n_development": len(dev), "n_held_out": len(held), "configs": {}}
    for config, tested in configs.items():
        preds = {**{k: (oof[k], out[k]) for k in ESTIMATORS}, **{t[0]: added[t[0]] for t in tested}}
        groups_tested = list(tested) + ([("duplicate_1", "duplicate_2")] if config == "duplicates" else [])
        maes, weights, _ = draw_errors(preds, y_dev, y_held, family + groups_tested, groups_tested, rng_for("check-corrupt", r, config))
        table, family_p = verdicts(maes, tested, family, boot)
        if config == "duplicates":
            pair_table, _ = verdicts(maes, [("duplicate_1", "duplicate_2")], family, boot)
            table.update(pair_table)
        record["configs"][config] = {"tested": table, "family": family_p, "weights": weights}
    return record


def _rate(flags):
    flags = np.asarray(flags, dtype=bool)
    p = float(flags.mean())
    return {"rate": p, "mc_se": float(np.sqrt(p * (1 - p) / len(flags))), "n": int(len(flags))}


def summarize_checks(records):
    summary = {}
    for config in ("useless", *(f"planted_{a:g}" for a in PLANTED)):
        rows = [r["configs"][config]["tested"][config] for r in records]
        summary[config] = {"passes": _rate([x["passes"] for x in rows]),
                           **{f"condition_{i}": _rate([x[f"condition_{i}"] for x in rows]) for i in (2, 3, 4)},
                           "conditions_2_and_3": _rate([x["condition_2"] and x["condition_3"] for x in rows])}
    tables = [r["configs"]["duplicates"]["tested"] for r in records]
    one, two, pair = "duplicate_1", "duplicate_2", "duplicate_1+duplicate_2"
    summary["duplicates"] = {
        "at_least_one_passes_alone": _rate([t[one]["passes"] or t[two]["passes"] for t in tables]),
        "both_pass_alone": _rate([t[one]["passes"] and t[two]["passes"] for t in tables]),
        "at_least_one_passes_alone_conditions_2_and_3": _rate(
            [(t[one]["condition_2"] and t[one]["condition_3"]) or (t[two]["condition_2"] and t[two]["condition_3"]) for t in tables]),
        "pair_passes_together": _rate([t[pair]["passes"] for t in tables]),
        "pair_conditions_2_and_3": _rate([t[pair]["condition_2"] and t[pair]["condition_3"] for t in tables]),
    }
    useless, planted = summary["useless"]["passes"]["rate"], summary["planted_0.2"]["passes"]["rate"]
    summary["usability"] = {"useless_passes_at_most": 0.05, "planted_0.2_passes_at_least": 0.80,
                            "rule_usable": bool(useless <= 0.05 and planted >= 0.80)}
    return summary


def run_checks(replicates, out_path):
    started = time.time()
    meta = provenance("checks")
    inputs = load_inputs()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress = out_path.with_name(out_path.stem + ".progress.jsonl")
    records = []
    with progress.open("w", encoding="utf-8") as handle:
        for r in range(replicates):
            record = jsonable(check_replicate(r, inputs), 6)
            records.append(record)
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            print(f"replicate {r + 1}/{replicates}: {time.time() - started:.0f} s", flush=True)
    result = {
        "addendum": "study/PLAN.md, Addendum: component admission trial (Checks of the rule itself)",
        "provenance": meta,
        "settings": {"replicates": replicates, "noise_draws_per_replicate": 1, "bootstrap_resamples": RESAMPLES,
                     "development_folds": DEV_FOLDS, "estimators": list(ESTIMATORS), "candidates": list(CANDIDATES),
                     "planted": list(PLANTED), "duplicate_a": DUPLICATE_A, "duplicate_noise_sd": DUPLICATE_SD,
                     "noise": {"additive": inputs.noise.additive, "gain": inputs.noise.gain}},
        "summary": summarize_checks(records),
        "seconds": round(time.time() - started, 1),
        "replicates": records,
    }
    write_json(out_path, result)
    progress.unlink()
    return result


# ----------------------------------------------------------------------------- the trial

def screen_choice(inputs, dev):
    """The added screen color with the lowest predicted MSE on the 27 bands, from development spectra and the curves."""
    s = inputs.sim[dev]
    mean, cov = s.mean(axis=0), np.cov(s, rowvar=False)
    second = cov + np.outer(mean, mean)

    def predicted(lights):
        rows, blocks = [], []
        for light, factor in lights:
            a = inputs.camera * light
            rows.append(a)
            blocks.append((inputs.noise.additive * factor) ** 2 * np.eye(3) + inputs.noise.gain ** 2 * a @ second @ a.T)
        h = np.vstack(rows)
        noise = np.zeros((len(h), len(h)))
        for i, block in enumerate(blocks):
            noise[3 * i:3 * i + 3, 3 * i:3 * i + 3] = block
        posterior = cov - cov @ h.T @ np.linalg.solve(h @ cov @ h.T + noise, h @ cov)
        return float(np.mean(np.diag(posterior)[METRIC_INDEX]))

    predicted_mse = {"flash alone": predicted([(inputs.flash, 1.0)])}
    for color, light in inputs.screens.items():
        predicted_mse[color] = predicted([(inputs.flash, 1.0), (light, SCREEN_NOISE_FACTOR)])
    return min(SCREEN_COLORS, key=lambda c: predicted_mse[c]), predicted_mse


def trial_draws(inputs, dev, held, dev_folds, shot):
    """All noise draws of one noise model. Returns per-sample MAE and target shares averaged over draws."""
    label = "shot" if shot else "moderate"
    y_dev, y_held = inputs.metric[dev], inputs.metric[held]
    mae_sums, within_sums = defaultdict(float), defaultdict(float)
    weights_log, best_log = [], []
    for draw in range(DRAWS):
        key = ("trial", label, draw)
        x = photos(inputs, inputs.flash, rng_for(*key, "flash"), shot=shot)
        oof, out = stack(ESTIMATORS, inputs, x, dev, held, dev_folds, key)
        preds = {k: (oof[k], out[k]) for k in ESTIMATORS}
        maes, weights, combined = draw_errors(preds, y_dev, y_held, [(c,) for c in CANDIDATES] + [("polynomial", "kernel")],
                                              [(c,) for c in CANDIDATES], rng_for(*key, "corrupt"))
        estimates = {"combination": combined, **{"alone:" + k: out[k] for k in ESTIMATORS}}
        dev_errors = {k: float(np.mean(sample_mae(y_dev, oof[k]))) for k in ESTIMATORS}
        best = min(ESTIMATORS, key=dev_errors.get)
        estimates["best_single"] = out[best]
        estimates["plain_average"] = np.mean([out[k] for k in ESTIMATORS], axis=0)
        estimates["alone:forward_model_wrong_camera"] = fit_predict("forward_model_wrong_camera", inputs, dev, held, x,
                                                                    rng_for(*key, "wrong-camera"))
        for color, light in inputs.screens.items():
            both = np.column_stack((x, photos(inputs, light, rng_for(*key, "screen", color), SCREEN_NOISE_FACTOR, shot)))
            model = fit_selected(both[dev], y_dev, inputs.groups[dev], rng_for(*key, "screen-fit", color), "ridge")
            estimates["screen:" + color] = model.predict(both[held])
        for name, estimate in estimates.items():
            maes[name] = sample_mae(y_held, estimate)
            within_sums[name] = within_sums[name] + within_targets(y_held, estimate)
        for name, values in maes.items():
            mae_sums[name] = mae_sums[name] + values
        weights_log.append(weights)
        best_log.append(best)
        print(f"{label} draw {draw + 1}/{DRAWS}", flush=True)
    return ({k: v / DRAWS for k, v in mae_sums.items()}, {k: float(np.mean(v / DRAWS)) for k, v in within_sums.items()},
            weights_log, best_log)


def analyze_trial(maes, shares, groups_held, choice, frozen_ok, pair_reported, label):
    boot = GroupBootstrap(groups_held, rng_for("trial-bootstrap", label))
    tested = [(c,) for c in CANDIDATES]
    table, _ = verdicts(maes, tested, tested, boot, frozen_ok)
    for c in CANDIDATES:
        change = 100 * (1 - np.median(maes["alone:" + c]) / np.median(maes["alone:affine"]))
        table[c]["median_mae_change_percent_vs_affine"] = float(change)
        table[c]["practically_useful"] = bool(table[c]["passes"] and change > 10)
        table[c]["q4_statistic_vs_affine"] = paired_reduction(maes["alone:affine"], maes["alone:" + c], groups_held,
                                                              rng_for("trial-q4-statistic", label, c))
    combination = {"vs_best_single": boot.contrast(maes["best_single"] - maes["combination"]),
                   "vs_plain_average": boot.contrast(maes["plain_average"] - maes["combination"])}
    combination["passes"] = bool(frozen_ok and combination["vs_best_single"]["low"] > 0)
    screens = {}
    for color in SCREEN_COLORS:
        reduction = paired_reduction(maes["alone:affine"], maes["screen:" + color], groups_held, rng_for("trial-screen", label, color))
        reduction["helps_under_q4_rule"] = bool(reduction["low"] > 0 and reduction["percent"] >= 20)
        screens[color] = {"q4_statistic_vs_flash": reduction,
                          "mean_vs_flash": boot.contrast(maes["alone:affine"] - maes["screen:" + color])}
    if choice == "blue":
        chosen = {"choice": choice, "passes": False, "reason": "the forward model chose blue, the simple rule's color"}
    else:
        versus = boot.contrast(maes["screen:blue"] - maes["screen:" + choice])
        chosen = {"choice": choice, "vs_blue": versus, "passes": bool(versus["low"] > 0)}
    systems = {name: {"median_mae": float(np.median(values)), "p90_mae": float(np.percentile(values, 90)),
                      "mean_mae": float(np.mean(values)), "share_within_targets": shares.get(name)}
               for name, values in maes.items() if name.split(":")[0] in {"alone", "combination", "best_single", "plain_average", "screen"}}
    return {
        "components": table,
        "combination": combination,
        "polynomial_and_kernel_removed_together": {
            "reported": bool(pair_reported), "exploratory": True,
            "unique_gain_unadjusted": boot.contrast(maes["without:polynomial+kernel"] - maes["combination"])},
        "forward_model_wrong_camera_minus_right": boot.contrast(maes["alone:forward_model_wrong_camera"] - maes["alone:forward_model"]),
        "screen_colors": screens,
        "screen_choice": chosen,
        "systems": systems,
    }


def run_trial(out_path):
    if not CHECKS_PATH.exists():
        raise SystemExit(f"Run the checks of the rule first: {CHECKS_PATH.relative_to(ROOT)} does not exist.")
    started = time.time()
    checks = json.loads(CHECKS_PATH.read_text(encoding="utf-8"))["summary"]
    meta = provenance("trial")
    frozen_before = frozen_state()
    inputs = load_inputs()
    split = group_folds(inputs.groups, 2, np.random.default_rng(SEED))
    dev, held = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    dev_folds = group_folds(inputs.groups[dev], DEV_FOLDS, rng_for("trial", "development-folds"))
    choice, predicted = screen_choice(inputs, dev)
    write_json(out_path, {"status": "screen choice recorded; no held-out error computed yet",
                          "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "screen_choice": {"choice": choice, "predicted_mse": predicted}, "provenance": meta})
    runs = {}
    for shot in (False, True):
        runs["shot" if shot else "moderate"] = trial_draws(inputs, dev, held, dev_folds, shot)
    frozen_after = frozen_state()
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT, capture_output=True, text=True)
    frozen_ok = bool(frozen_before == frozen_after and all(v["matches_commit"] for k, v in frozen_after.items() if k != "draft_targets")
                     and frozen_after["draft_targets"] == PUBLISHED_TARGETS and tests.returncode == 0)
    pair_reported = checks["duplicates"]["at_least_one_passes_alone"]["rate"] < 0.5
    groups_held = inputs.groups[held]
    analyses = {label: analyze_trial(maes, shares, groups_held, choice, frozen_ok, pair_reported, label)
                for label, (maes, shares, _, _) in runs.items()}

    def verdict_set(analysis):
        return {**{c: analysis["components"][c]["passes"] for c in CANDIDATES},
                "combination": analysis["combination"]["passes"], "screen_choice": analysis["screen_choice"]["passes"]}

    moderate, shot = verdict_set(analyses["moderate"]), verdict_set(analyses["shot"])
    result = {
        "addendum": "study/PLAN.md, Addendum: component admission trial",
        "provenance": meta,
        "rule_checks": {"file": str(CHECKS_PATH.relative_to(ROOT)), "sha256": sha256_file(CHECKS_PATH),
                        "rule_usable": checks["usability"]["rule_usable"],
                        "verdicts_are_findings": checks["usability"]["rule_usable"]},
        "condition_1": {"passes": frozen_ok, "before": frozen_before, "after": frozen_after,
                        "test_suite_exit_code": tests.returncode, "test_suite_tail": tests.stderr.strip().splitlines()[-3:]},
        "split": {"seed": SEED, "n_development": len(dev), "n_held_out": len(held),
                  "groups_development": int(len(np.unique(inputs.groups[dev]))), "groups_held_out": int(len(np.unique(groups_held)))},
        "screen_choice": {"choice": choice, "predicted_mse": predicted},
        "moderate_noise": analyses["moderate"],
        "shot_noise": analyses["shot"],
        "robust_to_noise_model": {name: moderate[name] == shot[name] for name in moderate},
        "weights_by_draw": {label: run[2] for label, run in runs.items()},
        "best_single_by_draw": {label: run[3] for label, run in runs.items()},
        "held_out_ids": [inputs.ids[i] for i in held],
        "per_sample_mae": {label: run[0] for label, run in runs.items()},
        "seconds": round(time.time() - started, 1),
    }
    write_json(out_path, result)
    return result


# ----------------------------------------------------------------------------- provenance and output

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, text=True):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=text).stdout


def frozen_state():
    state = {}
    for rel in FROZEN_FILES:
        data = (ROOT / rel).read_bytes()
        state[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "matches_commit": data == _git("show", f"HEAD:{rel}", text=False)}
    state["draft_targets"] = dict(DRAFT_TARGETS)
    return state


def provenance(command):
    from importlib import metadata

    from .sources import FILES

    plan = ROOT / "study" / "PLAN.md"
    return {
        "command": command,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git("rev-parse", "HEAD").strip(),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "plan_matches_commit": plan.read_bytes() == _git("show", "HEAD:study/PLAN.md", text=False),
        "uncommitted_paths": len(_git("status", "--porcelain").splitlines()),
        "sha256": {rel: sha256_file(ROOT / rel) for rel in ("study/PLAN.md", *CODE_FILES)},
        "data_sha256": {name: FILES[name][1] for name in DATA_FILES},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "colour-science": metadata.version("colour-science")},
        "platform": platform.platform(),
    }


def jsonable(value, digits=7):
    if isinstance(value, dict):
        return {str(k): jsonable(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist(), digits)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return float(f"{v:.{digits}g}") if np.isfinite(v) else None
    return value


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(data), separators=(",", ":")) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Component admission trial (study/PLAN.md addendum).")
    commands = parser.add_subparsers(dest="command", required=True)
    checks = commands.add_parser("checks", help="run the checks of the rule on components of known value (first)")
    checks.add_argument("--replicates", type=int, default=REPLICATES)
    checks.add_argument("--out", type=Path, default=CHECKS_PATH)
    trial = commands.add_parser("trial", help="run the trial itself (needs the checks file)")
    trial.add_argument("--out", type=Path, default=TRIAL_PATH)
    args = parser.parse_args(argv)
    if args.command == "checks":
        run_checks(args.replicates, args.out)
    else:
        run_trial(args.out)


if __name__ == "__main__":
    main()
