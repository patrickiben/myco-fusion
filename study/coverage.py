"""Q6 of study/PLAN.md: how often do 95%-labeled intervals contain the truth?

Compares the original percentile bootstrap with the exact order-statistic interval
now used by myco_fusion.validation. Writes results/coverage.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from myco_fusion.intervals import median_interval

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261101
SIMS, BOOT = 2000, 1000


def percentile_interval(values, rng):
    index = rng.integers(0, len(values), size=(BOOT, len(values)))
    return np.percentile(np.median(values[index], axis=1), [2.5, 97.5])


def cluster_percentile_interval(groups, rng):
    """Vectorized when sessions are equal in size, otherwise a resampling loop."""
    k = len(groups)
    chosen = rng.integers(0, k, size=(BOOT, k))
    if len({len(g) for g in groups}) == 1:
        table = np.vstack(groups)
        medians = np.median(table[chosen].reshape(BOOT, -1), axis=1)
    else:
        medians = np.array([np.median(np.concatenate([groups[i] for i in row])) for row in chosen])
    return np.percentile(medians, [2.5, 97.5])


def session_effects(rng, k, scenario):
    if scenario["session_distribution"] == "t3":
        # Scale a t(3) draw to the stated standard deviation (its SD is sqrt(3)).
        return rng.standard_t(3, k) * scenario["session_sd"] / np.sqrt(3)
    return rng.normal(0, scenario["session_sd"], k)


def simulate_sessions(rng, k, per_session, scenario):
    effects = session_effects(rng, k, scenario)
    sizes = rng.integers(2, 9, k) if scenario.get("unequal") else np.full(k, per_session)
    return [np.exp(effects[j] + rng.normal(0, scenario["specimen_sd"], sizes[j])) for j in range(k)]


SCENARIOS = {
    "session share 36%": {"session_sd": 0.3, "specimen_sd": 0.4, "session_distribution": "normal"},
    "session share 6%": {"session_sd": 0.1, "specimen_sd": 0.4, "session_distribution": "normal"},
    "heavy-tailed session effects": {"session_sd": 0.3, "specimen_sd": 0.4, "session_distribution": "t3"},
    "unequal session sizes": {"session_sd": 0.3, "specimen_sd": 0.4, "session_distribution": "normal", "unequal": True},
}


def main():
    rng = np.random.default_rng(SEED)
    specimens = []
    for n in (3, 4, 5, 6, 8, 10, 15, 20, 30, 40):
        old_hits, new_hits, level = 0, 0, None
        for _ in range(SIMS):
            x = np.exp(rng.normal(0, 0.5, n))
            low, high = percentile_interval(x, rng)
            old_hits += low <= 1 <= high
            interval = median_interval(x)
            new_hits += interval["low"] <= 1 <= interval["high"]
            level = interval["level"]
        specimens.append({"n_specimens": n, "bootstrap_coverage": old_hits / SIMS, "exact_coverage": new_hits / SIMS, "exact_stated_level": level})
    sessions = {}
    for name, scenario in SCENARIOS.items():
        rows = []
        for k in (2, 3, 4, 5, 6, 8, 12, 20):
            cluster, specimen, exact, widths_cluster, widths_exact, stated = 0, 0, 0, [], [], None
            for _ in range(SIMS):
                groups = simulate_sessions(rng, k, 5, scenario)
                low, high = cluster_percentile_interval(groups, rng)
                cluster += low <= 1 <= high
                widths_cluster.append(high - low)
                low, high = percentile_interval(np.concatenate(groups), rng)
                specimen += low <= 1 <= high
                interval = median_interval([np.median(g) for g in groups])
                exact += interval["low"] <= 1 <= interval["high"]
                widths_exact.append(interval["high"] - interval["low"])
                stated = interval["level"]
            rows.append({"n_sessions": k, "session_bootstrap_coverage": cluster / SIMS, "specimen_bootstrap_coverage": specimen / SIMS,
                         "exact_coverage": exact / SIMS, "exact_stated_level": stated,
                         "session_bootstrap_median_width": float(np.median(widths_cluster)), "exact_median_width": float(np.median(widths_exact))})
        sessions[name] = rows
    out = ROOT / "results" / "coverage.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"seed": SEED, "simulations": SIMS, "bootstrap_resamples": BOOT, "scenarios": SCENARIOS,
                               "independent_specimens": specimens, "sessions": sessions}, indent=1, allow_nan=False) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
