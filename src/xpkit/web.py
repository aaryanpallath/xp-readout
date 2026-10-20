"""Readout dashboard and API for business partners."""

from __future__ import annotations

import io
import itertools
from pathlib import Path
from urllib.parse import urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from starlette.middleware.trustedhost import TrustedHostMiddleware  # noqa: E402

from .readout import DECISIONS, MetricSpec, Readout, ai_summary, analyze  # noqa: E402
from .simulate import experiment  # noqa: E402

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
TEMPLATES.env.filters["pct"] = lambda v, d=1: f"{v * 100:+.{d}f}%"
TEMPLATES.env.filters["num"] = lambda v: f"{v:,.2f}"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "testserver"}

DEMOS = [
    ("New row ordering on the home page", dict(lift_hours=0.05)),
    ("Autoplay previews on by default", dict(lift_hours=0.05, retention_change=-0.02)),
    ("Redesigned title page (logging bug)", dict(lift_hours=0.05, treatment_share=0.515)),
    ("Larger artwork in search results", dict()),
]


def lift_chart(r: Readout) -> str:
    """Inline SVG: each metric's relative lift with its 95% interval around zero."""
    fig, ax = plt.subplots(figsize=(6.2, 0.7 + 0.55 * len(r.metrics)))
    for i, m in enumerate(r.metrics):
        color = "#1a7f37" if m["better"] else "#b42318" if m["worse"] else "#6e6e73"
        lo, hi = (m["lift"] - m["lift_ci"][0]) * 100, (m["lift_ci"][1] - m["lift"]) * 100
        ax.errorbar(m["lift"] * 100, i, xerr=[[lo], [hi]], fmt="o", color=color, capsize=4)
    ax.axvline(0, color="#999", lw=1)
    ax.set_yticks(range(len(r.metrics)), [m["metric"] for m in r.metrics])
    ax.invert_yaxis()
    ax.set_xlabel("relative change vs control (%), 95% CI")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.index("<svg"):]


def check_same_origin(request: Request) -> None:
    host = request.headers.get("host", "")
    if host.rsplit(":", 1)[0] not in LOCAL_HOSTS:
        raise HTTPException(403, "unexpected Host header")
    source = request.headers.get("origin") or request.headers.get("referer")
    if source is not None and urlsplit(source).netloc != host:
        raise HTTPException(403, "cross-site request rejected")
    if source is None and request.headers.get("sec-fetch-site", "same-origin") not in ("same-origin", "none"):
        raise HTTPException(403, "cross-site request rejected")


def create_app(demo_members: int = 40000) -> FastAPI:
    app = FastAPI(title="Experiment readouts")
    # Every request, reads included, must be addressed to a local host name: blocks DNS rebinding, where another
    # site points its domain at 127.0.0.1 to read this dashboard's data from the viewer's browser.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=sorted(LOCAL_HOSTS))
    store: dict[int, Readout] = {}
    ids = itertools.count(1)
    primary = MetricSpec("Viewing hours per member", "hours", True, "hours_pre", " h")
    guard = [MetricSpec("30-day retention", "retained", True)]
    for i, (name, kw) in enumerate(DEMOS):
        store[next(ids)] = ai_summary(analyze(experiment(demo_members, seed=100 + i, **kw), name, primary, guard))

    def get(eid: int) -> Readout:
        if eid not in store:
            raise HTTPException(404, "experiment not found")
        return store[eid]

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return TEMPLATES.TemplateResponse(request, "index.html", {"items": store, "labels": DECISIONS})

    @app.get("/experiments/{eid}", response_class=HTMLResponse)
    def page(request: Request, eid: int):
        r = get(eid)
        return TEMPLATES.TemplateResponse(request, "experiment.html", {"r": r, "label": DECISIONS[r.decision],
                                                                       "chart": lift_chart(r)})

    @app.get("/api/experiments/{eid}")
    def api(eid: int):
        return get(eid).to_dict()

    @app.post("/experiments")
    async def upload(request: Request, file: UploadFile = File(...), name: str = Form(...),
                     primary_column: str = Form(...), pre_column: str = Form(""), guardrails: str = Form("")):
        check_same_origin(request)
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file larger than 20 MB")
        try:
            df = pd.read_csv(io.BytesIO(data))
            spec = MetricSpec(primary_column, primary_column, True, pre_column or None)
            gs = [MetricSpec(g.strip(), g.strip(), True) for g in guardrails.split(",") if g.strip()]
            r = ai_summary(analyze(df, name[:120], spec, gs))
        except (ValueError, KeyError, pd.errors.ParserError) as e:
            raise HTTPException(400, f"could not analyze file: {e}") from None
        eid = next(ids)
        store[eid] = r
        return RedirectResponse(f"/experiments/{eid}", status_code=303)

    return app
