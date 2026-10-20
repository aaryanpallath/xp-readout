"""Experiment readout: per-metric results, guardrails, a decision, and a summary for business partners.

The AI summary (optional) receives only aggregate results, never member-level data. Every number it writes is
checked against the computed results, and its recommendation must match the decision; otherwise it is rejected and a
template summary is used instead.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import pandas as pd

from .stats import cuped, srm, welch

DECISIONS = {"SHIP": "Ship", "NO_SHIP": "Don't ship", "INCONCLUSIVE": "Inconclusive", "INVALID": "Do not trust"}


@dataclass
class MetricSpec:
    name: str
    column: str
    higher_is_better: bool = True
    pre_column: str | None = None      # pre-period covariate for CUPED
    unit: str = ""


@dataclass
class Readout:
    name: str
    srm: dict
    metrics: list[dict]
    decision: str
    reasons: list[str]
    summary: str = ""
    summary_source: str = "template"
    summary_rejections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def analyze(df: pd.DataFrame, name: str, primary: MetricSpec, guardrails: list[MetricSpec],
            treatment_share: float = 0.5) -> Readout:
    for col in ("variant",):
        if col not in df:
            raise ValueError(f"missing column {col!r}")
    variants = set(df.variant.unique())
    if variants != {"control", "treatment"}:
        raise ValueError(f"variant must be exactly 'control' and 'treatment', got {sorted(variants)}")
    c, t = df[df.variant == "control"], df[df.variant == "treatment"]
    check = srm(len(c), len(t), treatment_share)
    rows = []
    for spec, role in [(primary, "primary"), *[(g, "guardrail") for g in guardrails]]:
        for col in (spec.column, spec.pre_column):
            if col and col not in df:
                raise ValueError(f"missing column {col!r}")
        if spec.pre_column:
            eff, reduction = cuped(c[spec.column].to_numpy(float), t[spec.column].to_numpy(float),
                                   c[spec.pre_column].to_numpy(float), t[spec.pre_column].to_numpy(float))
            method = "CUPED"
        else:
            eff, reduction, method = welch(c[spec.column], t[spec.column]), 0.0, "Welch t-test"
        good = eff.lift > 0 if spec.higher_is_better else eff.lift < 0
        rows.append({"metric": spec.name, "role": role, "method": method, "variance_reduction": reduction,
                     "higher_is_better": spec.higher_is_better, "unit": spec.unit,
                     "better": eff.significant and good, "worse": eff.significant and not good, **eff.to_dict()})
    p = rows[0]
    worse_guardrails = [r["metric"] for r in rows[1:] if r["worse"]]
    if check["mismatch"]:
        decision = "INVALID"
        reasons = [f"Sample ratio mismatch: {check['observed_treatment_share']:.2%} of members are in treatment "
                   f"vs {check['expected_treatment_share']:.0%} intended (p = {check['p_value']:.2g}). "
                   "Randomization or logging is broken; fix it before reading any metric."]
    elif p["worse"] or worse_guardrails:
        decision = "NO_SHIP"
        reasons = ([f"{p['metric']} got worse ({p['lift']:+.2%})."] if p["worse"] else []) + \
                  [f"Guardrail {m} regressed." for m in worse_guardrails]
    elif p["better"]:
        decision = "SHIP"
        reasons = [f"{p['metric']} improved {p['lift']:+.2%} (95% CI {p['lift_ci'][0]:+.2%} to "
                   f"{p['lift_ci'][1]:+.2%}), and no guardrail regressed."]
    else:
        decision = "INCONCLUSIVE"
        reasons = [f"No significant change in {p['metric']} ({p['lift']:+.2%}, p = {p['p_value']:.2f}). "
                   f"This test could only detect changes of about {p['mde_80']:.1%} or larger."]
    r = Readout(name, check, rows, decision, reasons)
    r.summary = template_summary(r)
    return r


def template_summary(r: Readout) -> str:
    p = r.metrics[0]
    lines = [f"Recommendation: {DECISIONS[r.decision]}.", *r.reasons]
    if r.decision != "INVALID":
        lines.append(f"{p['metric']}: control {p['control_mean']:.2f}{p['unit']}, treatment "
                     f"{p['treatment_mean']:.2f}{p['unit']}, {p['n_control']:,} vs {p['n_treatment']:,} members.")
    return " ".join(lines)


# ---------------------------------------------------------------------------------------------- AI summary

NUM = re.compile(r"[-+]?\d[\d,]*\.?\d*%?")
VERDICT_PATTERNS = {
    "SHIP": re.compile(r"\b(recommend|should|ready to|safe to)\s+(ship|launch|roll(ing)? out)", re.I),
    "NO_SHIP": re.compile(r"\b(do not|don't|should not|shouldn't|not)\s+(ship|launch|roll out)", re.I),
}


def allowed_numbers(r: Readout) -> list[float]:
    """Every value a correct summary may cite, in the units people write them in."""
    vals = [r.srm["observed_treatment_share"] * 100, r.srm["expected_treatment_share"] * 100, r.srm["p_value"],
            r.srm["n_control"], r.srm["n_treatment"], r.srm["n_control"] + r.srm["n_treatment"], 95, 80]
    # numbers that are part of a name ("30-day retention") are labels, not claims
    for label in [r.name, *[m["metric"] for m in r.metrics]]:
        vals += [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", label)]
    for m in r.metrics:
        for k in ("control_mean", "treatment_mean", "diff", "p_value", "n_control", "n_treatment", "se"):
            vals.append(m[k])
        for k in ("lift", "mde_80", "variance_reduction"):
            vals += [m[k] * 100]
        vals += [x * 100 for x in m["lift_ci"]] + list(m["diff_ci"])
    return [abs(float(v)) for v in vals]


def validate_summary(text: str, r: Readout) -> list[str]:
    """Problems with an AI-written summary; empty means every number and the verdict check out."""
    problems = []
    allowed = allowed_numbers(r)
    for tok in NUM.findall(text):
        raw = tok.rstrip("%").replace(",", "").lstrip("+-")
        if not raw or raw == ".":
            continue
        val = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10 ** -decimals + 1e-9
        if not any(abs(a - val) <= tol for a in allowed):
            problems.append(f"number {tok!r} does not match any computed result")
    says_ship = bool(VERDICT_PATTERNS["SHIP"].search(text)) and not VERDICT_PATTERNS["NO_SHIP"].search(text)
    says_no = bool(VERDICT_PATTERNS["NO_SHIP"].search(text))
    if says_ship and r.decision != "SHIP":
        problems.append(f"recommends shipping, but the decision is {r.decision}")
    if says_no and r.decision == "SHIP":
        problems.append("recommends not shipping, but the decision is SHIP")
    if r.decision == "INVALID" and "mismatch" not in text.lower():
        problems.append("does not warn about the sample ratio mismatch")
    return problems


def ai_summary(r: Readout, complete=None) -> Readout:
    """Ask an LLM for a short partner-facing summary of the aggregate results, then validate it.
    `complete(prompt) -> str` is injectable for tests; by default Claude is used when ANTHROPIC_API_KEY is set."""
    if complete is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return r
        complete = _claude
    facts = {"experiment": r.name, "decision": r.decision, "reasons": r.reasons, "srm": r.srm,
             "metrics": [{k: m[k] for k in ("metric", "role", "lift", "lift_ci", "p_value", "significant",
                                              "control_mean", "treatment_mean", "unit")} for m in r.metrics]}
    prompt = ("Write a 3-4 sentence summary of this A/B test for non-technical business partners. Use only numbers "
              "from the data, written as percentages with at most one decimal for lifts. State the recommendation "
              f"exactly as the decision field says. Data:\n{json.dumps(facts, default=float)}")
    try:
        text = complete(prompt).strip()
    except Exception as e:  # noqa: BLE001 - the template summary remains
        r.summary_rejections = [f"AI summary unavailable: {type(e).__name__}"]
        return r
    problems = validate_summary(text, r)
    if problems:
        r.summary_rejections = problems
    else:
        r.summary, r.summary_source = text, "AI (validated)"
    return r


def _claude(prompt: str) -> str:
    import anthropic

    msg = anthropic.Anthropic().messages.create(model="claude-sonnet-5-5", max_tokens=300,
                                                messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text
