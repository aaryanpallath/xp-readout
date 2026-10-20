"""Simulation study: does each method do what it claims? Every scenario has a known true effect."""

from __future__ import annotations

import numpy as np

from .simulate import experiment
from .stats import cuped, msprt_p_values, srm, welch


def _arms(df):
    c, t = df[df.variant == "control"], df[df.variant == "treatment"]
    return c, t


def aa_false_positives(sims: int, n: int, seed: int = 0) -> dict:
    """A/A tests (no true effect): how often is the result 'significant'? Should be about 5%."""
    raw = cup = 0
    for i in range(sims):
        c, t = _arms(experiment(n, seed=seed + i))
        raw += welch(c.hours, t.hours).significant
        cup += cuped(c.hours.values, t.hours.values, c.hours_pre.values, t.hours_pre.values)[0].significant
    return {"sims": sims, "members": n, "false_positive_rate_welch": raw / sims,
            "false_positive_rate_cuped": cup / sims}


def power(sims: int, n: int, lift: float, seed: int = 0) -> dict:
    """With a true lift, how often is it detected, and how much does CUPED narrow the confidence interval?"""
    raw = cup = 0
    width_raw, width_cup, reductions, covered = [], [], [], 0
    for i in range(sims):
        c, t = _arms(experiment(n, lift_hours=lift, seed=seed + i))
        r = welch(c.hours, t.hours)
        u, red = cuped(c.hours.values, t.hours.values, c.hours_pre.values, t.hours_pre.values)
        raw += r.significant and r.lift > 0
        cup += u.significant and u.lift > 0
        width_raw.append(r.lift_ci[1] - r.lift_ci[0])
        width_cup.append(u.lift_ci[1] - u.lift_ci[0])
        reductions.append(red)
        covered += u.lift_ci[0] <= lift <= u.lift_ci[1]
    return {"sims": sims, "members": n, "true_lift": lift, "power_welch": raw / sims, "power_cuped": cup / sims,
            "mean_variance_reduction": float(np.mean(reductions)),
            "ci_width_ratio_cuped_vs_welch": float(np.mean(width_cup) / np.mean(width_raw)),
            "cuped_ci_coverage": covered / sims}


def peeking(sims: int, n: int, lift: float = 0.0, days: int = 14, seed: int = 0, tau_frac: float = 0.1) -> dict:
    """Look at the results every day and stop at the first 'significant' result. A fixed-horizon test inflates
    false positives under peeking; the always-valid mSPRT p-value does not."""
    naive = seq = fixed = 0
    seq_days = []
    for i in range(sims):
        df = experiment(n, lift_hours=lift, days=days, seed=seed + i)
        tau2 = None
        diffs, variances, ps = [], [], []
        for day in range(1, days + 1):
            c, t = _arms(df[df.first_day <= day])
            if len(c) < 30 or len(t) < 30:
                diffs.append(0.0)
                variances.append(1.0)
                ps.append(1.0)
                continue
            e = welch(c.hours, t.hours)
            if tau2 is None:   # mixing scale: the size of effect the test is tuned for, as a share of the mean
                tau2 = (tau_frac * e.control_mean) ** 2
            diffs.append(e.diff)
            variances.append(e.se ** 2)
            ps.append(e.p_value)
        naive += any(p < 0.05 for p in ps)
        fixed += ps[-1] < 0.05          # the honest comparison: one planned look at the end
        avp = msprt_p_values(np.array(diffs), np.array(variances), tau2)
        hit = np.flatnonzero(avp < 0.05)
        if hit.size:
            seq += 1
            seq_days.append(int(hit[0]) + 1)
    out = {"sims": sims, "members": n, "true_lift": lift, "looks": days, "tau_fraction_of_mean": tau_frac,
           "reject_rate_naive_daily_peeking": naive / sims, "reject_rate_msprt": seq / sims,
           "reject_rate_fixed_horizon_single_look": fixed / sims}
    if seq_days:
        out["msprt_median_day_of_decision"] = float(np.median(seq_days))
    return out


def srm_detection(sims: int, n: int, seed: int = 0) -> dict:
    """Healthy 50/50 splits vs. a randomization bug that sends 51% to treatment."""
    healthy = broken = 0
    rng = np.random.default_rng(seed)
    for _ in range(sims):
        t = rng.binomial(n, 0.5)
        healthy += srm(n - t, t)["mismatch"]
        t = rng.binomial(n, 0.51)
        broken += srm(n - t, t)["mismatch"]
    return {"sims": sims, "members": n, "false_alarm_rate_healthy": healthy / sims,
            "detection_rate_51_49_split": broken / sims}


def tau_sensitivity(sims: int, n: int, lift: float, seed: int = 0) -> list[dict]:
    """mSPRT with different mixing scales: the false positive rate stays controlled; power depends on tuning."""
    rows = []
    for tau_frac in (0.02, 0.05, 0.1, 0.2, 0.5):
        null = peeking(sims, n, 0.0, seed=seed, tau_frac=tau_frac)
        eff = peeking(sims, n, lift, seed=seed + 100_000, tau_frac=tau_frac)
        rows.append({"tau_fraction_of_mean": tau_frac, "false_positive_rate": null["reject_rate_msprt"],
                     "power": eff["reject_rate_msprt"],
                     "median_day_of_decision": eff.get("msprt_median_day_of_decision")})
    return rows


def run_all(scale: float = 1.0) -> dict:
    s = lambda k: max(20, int(k * scale))  # noqa: E731
    return {
        "aa": aa_false_positives(s(2000), 4000, seed=10_000),
        "power": power(s(1000), 20000, lift=0.02, seed=20_000),
        "peeking_null": peeking(s(1000), 6000, lift=0.0, seed=30_000),
        "peeking_effect": peeking(s(500), 6000, lift=0.06, seed=40_000),
        "srm": srm_detection(s(5000), 100_000, seed=50_000),
        "msprt_tau_sensitivity": tau_sensitivity(s(500), 6000, lift=0.06, seed=60_000),
    }
