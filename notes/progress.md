# Progress notes (working file)

## Status (26 Oct 2026)
- No QPU time used. Ledger: 0 / 1,200 s.
- Account `qdrift-main` works; operational: ibm_fez, ibm_kingston, ibm_marrakesh.
- Tests: 18 passed (unit + fake-backend integration, including a SamplerV2 run of packed PUBs).
- Simulator test accuracy: two_curves M1 0.927, M2 0.873, M3 0.847; breast_cancer M1 0.967,
  M2 0.933, M3 0.913. Best classical: two_curves RBF-SVM 0.987, breast_cancer MLP 0.960.
- Full dry run (2 fake devices x 5 sessions + pilot) passes validation; 6 figures and all
  tables produced in results/dry-run (about 9 minutes).
- Real layouts frozen from live calibration (no QPU): data/layouts/ibm_fez.json and
  ibm_marrakesh.json, 14 four-qubit patches each, mean readout error 0.005-0.016.
- Estimates (IBM formula x 1.25 safety): session 32.7 s, pilot 17.4 s per device.
  Campaign 10 sessions <= 327 s reserved (cap 400 s); pilot on both devices <= 35 s (cap 90 s).

## Design decisions made before pre-registration (no hardware data used)
- M2 depth 1 layer (validation accuracy; scripts/select_fixed_maps.py).
- R2 penalty 0.25 k toward the simulator head (dry-run data; scripts/select_refit_lambda.py).
- H2 statistic: mean variance ratio (the median is biased low under pure shot noise).

## Pilot (26 Oct 2026)
- ibm_fez pilot done: 11 s used; 0.000265 s per execution (now in config); no detectable crosstalk.
- ibm_marrakesh pilot job dbfn78gm94ac738tu26g stuck in queue (device in maintenance, 2,020 jobs).
  Recommended: cancel it from the dashboard; the old waiting process then settles it at 0 s.
- Campaign devices switched to ibm_fez + ibm_kingston (Heron r2, 8 queued jobs); marrakesh is
  the fallback. ibm_kingston layout frozen.
- Code: Ctrl+C during a hardware wait is now safe (pending record + `qdrift fetch`);
  `qdrift budget audit` re-checks every settled job against IBM usage. 19 tests pass.

- ibm_kingston pilot job dbfo7vom94ac738u0lkg queued at 16:46 UTC (kingston queue grew 8 -> 50).
- Finished jobs now also save `snapshot_at_run.json` (calibration in force when the job ran,
  from historical properties) and job timestamps / queue wait; analysis uses it for R3r and
  the regression. The emulator prediction still uses the calibration at submission.

- Marrakesh pilot cancelled while queued: settled at 0 s.
- Kingston pilot done (queued ~4.8 h): 11 s, 0.000265 s/execution, no detectable crosstalk.
  Ledger after pilots: 22.0 s used, 0 pending. Pilot section of preregistration.md filled.

- Pre-registration frozen: commit 154984c, annotated tag `prereg-v1` (local only, not pushed).
  config_sha256 3a56fede...95f2. Any later config change is a reported deviation.

## Campaign log
| Day | Device | Session | Job | Queue wait | Usage | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 27 Oct | ibm_fez | 1 | dbg1nvdrd2mc73ctp640 | 2.2 min | 20 s | saved, valid |
| 27 Oct | ibm_kingston | 1 | dbg1oudrd2mc73ctp8qg | 1.9 h | 21 s | saved 30 Oct, valid |

Ledger after day 1: 63.0 s used (pilots 22 s, campaign 41 s), 0 pending.
Planned next sessions (mornings IST, both devices): 31 Oct, 2 Nov, 4 Nov, 6 Nov; deadline 10 Nov.

First look (fez s1, descriptive only, no design changes): hardware R0 accuracy is 4-19 points
below simulator; measured readout mitigation (R3m) recovers 2-5 points; split conformal
calibrated on the simulator under-covers (0.69-0.91), ACI stays near 0.90.

## Next (needs the user)
1. Optional: push the repo and tag to GitHub for a public, verifiable pre-registration date.
2. `uv run qdrift noise-aware --device ibm_fez` and `--device ibm_kingston` (no QPU).
3. Campaign: `uv run qdrift session --device ibm_fez` and `--device ibm_kingston`, one per
   device per day, 5 each over 10-14 days; `uv run qdrift fetch` later the same day.
3. Commit and tag the pre-registration (ask the user first).
4. `uv run qdrift noise-aware --device <name>` right before session 1, then 5 sessions per
   device over 10-14 days.
5. Reading log: 42 entries in literature/reading-log.md; `scripts/check_reading_log.py`
   lists open items. Bib entries marked VERIFY need author lists checked.
