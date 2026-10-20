import numpy as np
import pytest
from scipy import stats as sps

from xpkit.readout import MetricSpec, ai_summary, analyze, validate_summary
from xpkit.simulate import experiment
from xpkit.stats import cuped, msprt_p_values, srm, welch
from xpkit.study import aa_false_positives, peeking

P = MetricSpec("Viewing hours per member", "hours", True, "hours_pre", " h")
G = [MetricSpec("30-day retention", "retained", True)]


def arms(df):
    return df[df.variant == "control"], df[df.variant == "treatment"]


# ------------------------------------------------------------------ statistics
def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.normal(10, 3, 500), rng.normal(10.4, 4, 700)
    e = welch(a, b)
    assert e.p_value == pytest.approx(sps.ttest_ind(b, a, equal_var=False).pvalue, rel=1e-9)
    assert e.diff_ci[0] < e.diff < e.diff_ci[1]


def test_cuped_is_unbiased_and_reduces_variance():
    lifts, reductions = [], []
    for s in range(200):
        c, t = arms(experiment(8000, lift_hours=0.03, seed=s))
        e, red = cuped(c.hours.values, t.hours.values, c.hours_pre.values, t.hours_pre.values)
        lifts.append(e.lift)
        reductions.append(red)
    assert np.mean(lifts) == pytest.approx(0.03, abs=0.004)
    assert np.mean(reductions) > 0.2


def test_srm_flags_broken_splits_only():
    assert not srm(50_100, 49_900)["mismatch"]
    assert srm(49_000, 51_000)["mismatch"]
    assert not srm(30_100, 69_900, expected_treatment_share=0.7)["mismatch"]


def test_msprt_p_values_never_increase():
    p = msprt_p_values(np.array([0.1, -0.3, 0.8, 0.2]), np.array([1.0, 0.5, 0.25, 0.1]), tau2=1.0)
    assert np.all(np.diff(p) <= 0) and np.all((p >= 0) & (p <= 1))


def test_study_small_runs_show_the_expected_pattern():
    aa = aa_false_positives(300, 2000, seed=1)
    assert 0.02 <= aa["false_positive_rate_welch"] <= 0.09
    pk = peeking(150, 3000, seed=2)
    assert pk["reject_rate_naive_daily_peeking"] > pk["reject_rate_msprt"]
    assert pk["reject_rate_msprt"] <= 0.05


# ------------------------------------------------------------------ readout decisions
@pytest.mark.parametrize("kw,decision", [
    (dict(lift_hours=0.05), "SHIP"),
    (dict(lift_hours=0.05, retention_change=-0.02), "NO_SHIP"),
    (dict(lift_hours=-0.05), "NO_SHIP"),
    (dict(lift_hours=0.05, treatment_share=0.515), "INVALID"),
    (dict(), "INCONCLUSIVE"),
])
def test_decisions(kw, decision):
    assert analyze(experiment(40000, seed=7, **kw), "x", P, G).decision == decision


def test_bad_input_is_rejected():
    df = experiment(1000, seed=1)
    with pytest.raises(ValueError):
        analyze(df.drop(columns="hours"), "x", P, G)
    df.loc[0, "variant"] = "treatment_b"
    with pytest.raises(ValueError):
        analyze(df, "x", P, G)


# ------------------------------------------------------------------ AI summary validation
@pytest.fixture(scope="module")
def win():
    return analyze(experiment(40000, seed=7, lift_hours=0.05), "Row ordering test", P, G)


def test_correct_summary_passes(win):
    lift = win.metrics[0]["lift"] * 100
    text = f"Recommendation: ship. Viewing hours rose {lift:.1f}%, and 30-day retention held steady."
    assert validate_summary(text, win) == []


@pytest.mark.parametrize("text,problem", [
    ("We recommend shipping: viewing hours rose 12.4%.", "12.4%"),
    ("Viewing hours rose 4.5%, but we should not ship yet.", "recommends not shipping"),
])
def test_wrong_numbers_and_verdicts_are_caught(win, text, problem):
    assert any(problem in p for p in validate_summary(text.replace("4.5", f"{win.metrics[0]['lift']*100:.1f}"), win))


def test_srm_must_be_mentioned():
    bad = analyze(experiment(40000, seed=7, lift_hours=0.05, treatment_share=0.515), "x", P, G)
    assert any("ratio mismatch" in p for p in validate_summary("Viewing hours went up; we should ship.", bad))


def test_rejected_or_failing_ai_falls_back_to_template(win):
    r = analyze(experiment(40000, seed=7, lift_hours=0.05), "x", P, G)
    ai_summary(r, lambda prompt: "We recommend shipping: hours rose 99%.")
    assert r.summary_source == "template" and r.summary_rejections
    r2 = analyze(experiment(40000, seed=7, lift_hours=0.05), "x", P, G)

    def boom(prompt):
        raise TimeoutError

    ai_summary(r2, boom)
    assert r2.summary_source == "template" and "unavailable" in r2.summary_rejections[0]


def test_prompt_contains_only_aggregates(win):
    seen = []
    ai_summary(analyze(experiment(5000, seed=3, lift_hours=0.05), "x", P, G), lambda p: seen.append(p) or "x")
    assert "member_id" not in seen[0] and "hours_pre" not in seen[0]
