"""xp: run the simulation study or serve the readout dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="xp", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("study", help="run the simulation study and write results/")
    st.add_argument("--scale", type=float, default=1.0, help="fraction of the full number of simulations")
    st.add_argument("--out", type=Path, default=Path("results"))
    sv = sub.add_parser("serve", help="run the readout dashboard on 127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    if a.cmd == "study":
        from .plots import study_charts
        from .study import run_all

        a.out.mkdir(parents=True, exist_ok=True)
        r = run_all(a.scale)
        (a.out / "study.json").write_text(json.dumps(r, indent=2) + "\n")
        study_charts(r, a.out / "study.png")
        print(json.dumps(r, indent=1))
    else:
        import uvicorn

        from .web import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=a.port)
