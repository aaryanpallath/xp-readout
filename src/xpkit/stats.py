"""Statistics for two-arm experiments: Welch tests with relative lift, CUPED, sample ratio mismatch, and
always-valid sequential p-values (mSPRT)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

ALPHA = 0.05
SRM_ALPHA = 0.001   # SRM is checked on every experiment, so a strict threshold keeps false alarms rare


@dataclass
class Effect:
    control_mean: float
    treatment_mean: float
    diff: float
    diff_ci: tuple[float, float]
    lift: float                 # relative: treatment / control - 1
    lift_ci: tuple[float, float]
    p_value: float
    se: float
    n_control: int
    n_treatment: int

    @property
    def significant(self) -> bool:
        return self.p_value < ALPHA

    @property
    def mde_80(self) -> float:
        """Smallest relative lift this experiment could detect with 80% power (two-sided 5%)."""
        return float((stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.8)) * self.se / abs(self.control_mean))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(significant=self.significant, mde_80=self.mde_80)
        return d


def welch(control: np.ndarray, treatment: np.ndarray) -> Effect:
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    mc, mt = c.mean(), t.mean()
    vc, vt = c.var(ddof=1) / len(c), t.var(ddof=1) / len(t)
    se = float(np.sqrt(vc + vt))
    diff = float(mt - mc)
    df = (vc + vt) ** 2 / (vc ** 2 / (len(c) - 1) + vt ** 2 / (len(t) - 1))
    p = float(2 * stats.t.sf(abs(diff) / se, df)) if se > 0 else 1.0
    q = stats.t.ppf(1 - ALPHA / 2, df)
    lift = mt / mc - 1
    # delta method for the ratio of two independent means
    se_lift = float(np.sqrt(vt / mc ** 2 + mt ** 2 * vc / mc ** 4))
    z = stats.norm.ppf(1 - ALPHA / 2)
    return Effect(float(mc), float(mt), diff, (diff - q * se, diff + q * se), float(lift),
                  (float(lift - z * se_lift), float(lift + z * se_lift)), p, se, len(c), len(t))


def cuped(y_c: np.ndarray, y_t: np.ndarray, x_c: np.ndarray, x_t: np.ndarray) -> tuple[Effect, float]:
    """CUPED (Deng et al., 2013): subtract theta * (pre-period metric - its mean). theta is estimated on both arms
    pooled, so the adjustment cannot bias the treatment effect. Returns the effect and the variance reduction."""
    y = np.concatenate([y_c, y_t])
    x = np.concatenate([x_c, x_t])
    theta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    mx = x.mean()
    adj_c, adj_t = y_c - theta * (x_c - mx), y_t - theta * (x_t - mx)
    reduction = 1 - np.concatenate([adj_c, adj_t]).var() / y.var()
    adj = welch(adj_c, adj_t)
    # Relative lift = adjusted difference / raw control mean, with a delta-method interval.
    mc = float(np.mean(y_c))
    var_mc = float(np.var(y_c, ddof=1) / len(y_c))
    lift = adj.diff / mc
    se_lift = float(np.sqrt(adj.se ** 2 / mc ** 2 + adj.diff ** 2 * var_mc / mc ** 4))
    z = stats.norm.ppf(1 - ALPHA / 2)
    eff = Effect(mc, mc + adj.diff, adj.diff, adj.diff_ci, lift, (lift - z * se_lift, lift + z * se_lift),
                 adj.p_value, adj.se, adj.n_control, adj.n_treatment)
    return eff, float(reduction)


def srm(n_control: int, n_treatment: int, expected_treatment_share: float = 0.5) -> dict:
    """Sample ratio mismatch: is the observed split consistent with the intended allocation?"""
    n = n_control + n_treatment
    expected = [n * (1 - expected_treatment_share), n * expected_treatment_share]
    p = float(stats.chisquare([n_control, n_treatment], expected).pvalue)
    return {"n_control": n_control, "n_treatment": n_treatment, "observed_treatment_share": n_treatment / n,
            "expected_treatment_share": expected_treatment_share, "p_value": p, "mismatch": p < SRM_ALPHA}


def msprt_p_values(diffs: np.ndarray, variances: np.ndarray, tau2: float) -> np.ndarray:
    """Always-valid p-values for a difference in means observed at successive looks (Johari et al., 2017).

    At each look the mixture likelihood ratio against "no effect", with a N(0, tau2) mixing prior on the effect, is
    sqrt(V / (V + tau2)) * exp(tau2 * diff^2 / (2 V (V + tau2))), where V is the variance of the estimated
    difference. The p-value is the running minimum of 1 / ratio, so it can be checked at every look and
    rejecting whenever it falls below alpha keeps the false positive rate at or below alpha."""
    d = np.asarray(diffs, float)
    v = np.asarray(variances, float)
    log_lr = 0.5 * np.log(v / (v + tau2)) + tau2 * d ** 2 / (2 * v * (v + tau2))
    p = np.minimum(1.0, np.exp(-log_lr))
    return np.minimum.accumulate(p)
