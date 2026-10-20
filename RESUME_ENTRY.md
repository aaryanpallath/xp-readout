# XP Readout — Trustworthy A/B Test Readouts
Tech: Python, NumPy, SciPy, pandas, FastAPI, Claude API, pytest, GitHub Actions
- Built an A/B test readout service and dashboard with CUPED variance reduction, sample-ratio checks, always-valid sequential p-values and ship/no-ship/inconclusive decisions with guardrail metrics.
- Showed by simulation that checking an ordinary test daily raised false positives from 5% to 20.1%, while always-valid p-values held 1.0%; CUPED cut variance 27% and raised detection from 17.9% to 25.3%.
- Added LLM-written summaries that are shown only after every cited number and the recommendation are verified against the computed results, with aggregate-only prompts for privacy.
