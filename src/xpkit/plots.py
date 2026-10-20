"""Charts for the simulation study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def study_charts(r: dict, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    pn = r["peeking_null"]
    labels = ["Fixed horizon,\none look", "Peek daily,\nstop at p<0.05", "Always-valid\n(mSPRT), daily"]
    vals = [pn["reject_rate_fixed_horizon_single_look"], pn["reject_rate_naive_daily_peeking"], pn["reject_rate_msprt"]]
    bars = a.bar(labels, [v * 100 for v in vals], color=["#5f6368", "#b42318", "#1a7f37"])
    a.axhline(5, ls="--", color="#999", lw=1)
    a.text(2.45, 5.4, "promised 5%", ha="right", fontsize=8, color="#666")
    for bar, v in zip(bars, vals, strict=True):
        a.text(bar.get_x() + bar.get_width() / 2, v * 100 + 0.5, f"{v:.1%}", ha="center", fontsize=9)
    a.set(ylabel="false positive rate (%)", title=f"No real effect, {pn['looks']} daily looks ({pn['sims']:,} runs)")
    pw = r["power"]
    vals = [pw["power_welch"], pw["power_cuped"]]
    bars = b.bar(["Welch t-test", "CUPED"], [v * 100 for v in vals], color=["#5f6368", "#e50914"])
    for bar, v in zip(bars, vals, strict=True):
        b.text(bar.get_x() + bar.get_width() / 2, v * 100 + 0.5, f"{v:.1%}", ha="center", fontsize=9)
    b.set(ylabel="detection rate (%)", ylim=(0, max(vals) * 100 * 1.3),
          title=f"True +{pw['true_lift']:.0%} lift: CUPED cuts variance {pw['mean_variance_reduction']:.0%}")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
