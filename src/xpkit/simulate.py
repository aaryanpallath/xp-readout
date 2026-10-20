"""Synthetic streaming experiments with known truth.

Each member has a latent engagement level that drives weekly viewing hours both before the experiment (the CUPED
covariate) and during it, plus a 30-day retention flag. Treatment can change hours and retention by known amounts,
and allocation can be deliberately broken to test the sample-ratio check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def experiment(n: int, lift_hours: float = 0.0, retention_change: float = 0.0, treatment_share: float = 0.5,
               pre_post_noise: float = 0.6, days: int = 14, seed: int = 0) -> pd.DataFrame:
    """n members; lift_hours is the true relative change in hours; retention_change is in percentage points."""
    rng = np.random.default_rng(seed)
    engagement = rng.normal(0.0, 0.8, n)
    treated = rng.random(n) < treatment_share
    hours_pre = np.exp(2.0 + engagement + rng.normal(0, pre_post_noise, n))
    hours = np.exp(2.0 + engagement + rng.normal(0, pre_post_noise, n)) * np.where(treated, 1 + lift_hours, 1.0)
    p_retain = 1 / (1 + np.exp(-(1.6 + 0.9 * engagement)))
    p_retain = np.clip(p_retain + np.where(treated, retention_change, 0.0), 0, 1)
    retained = (rng.random(n) < p_retain).astype(float)
    return pd.DataFrame({
        "member_id": np.arange(n), "variant": np.where(treated, "treatment", "control"),
        "first_day": rng.integers(1, days + 1, n), "hours_pre": hours_pre, "hours": hours, "retained": retained,
    })
