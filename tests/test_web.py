import io

from fastapi.testclient import TestClient

import xpkit.web as web
from xpkit.simulate import experiment

client = TestClient(web.create_app(demo_members=6000))


def csv_bytes(**kw) -> bytes:
    return experiment(4000, seed=5, **kw).to_csv(index=False).encode()


def form(**extra):
    return {"name": "Upload test", "primary_column": "hours", "pre_column": "hours_pre", "guardrails": "retained",
            **extra}


def test_pages_render():
    assert client.get("/").status_code == 200
    page = client.get("/experiments/1")
    assert page.status_code == 200 and "<svg" in page.text and "Randomization check" in page.text
    assert client.get("/api/experiments/3").json()["decision"] == "INVALID"
    assert client.get("/experiments/999").status_code == 404


def test_upload_creates_a_readout():
    r = client.post("/experiments", data=form(), files={"file": ("x.csv", csv_bytes(lift_hours=0.1), "text/csv")},
                    follow_redirects=False)
    assert r.status_code == 303
    assert client.get(r.headers["location"]).status_code == 200


def test_bad_files_are_rejected_cleanly():
    r = client.post("/experiments", data=form(primary_column="nope"),
                    files={"file": ("x.csv", csv_bytes(), "text/csv")})
    assert r.status_code == 400
    r = client.post("/experiments", data=form(), files={"file": ("x.csv", b"\x00\x01garbage", "text/csv")})
    assert r.status_code == 400


def test_oversized_upload_is_refused(monkeypatch):
    monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/experiments", data=form(), files={"file": ("x.csv", csv_bytes(), "text/csv")})
    assert r.status_code == 413


def test_cross_site_upload_is_refused():
    r = client.post("/experiments", data=form(), files={"file": ("x.csv", io.BytesIO(csv_bytes()), "text/csv")},
                    headers={"origin": "https://evil.example"})
    assert r.status_code == 403
