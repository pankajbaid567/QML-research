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
| 30 Oct | ibm_fez | 2 | dbidpilep8hs73anl640 | 0.3 h | 20 s | ran 18:24 UTC, valid |
| 30 Oct | ibm_kingston | 2 | dbidqso1h92s73838kc0 | 0.2 h | 21 s | ran 18:23 UTC, valid |
| 4 Nov | ibm_fez | 3 | dblbhfe74qds73dq6bsg | 0.3 h | 20 s | ran 05:07 UTC (10:37 IST), valid |
| 4 Nov | ibm_kingston | 3 | dblbht674qds73dq6cd0 | 0.0 h | 21 s | ran 04:47 UTC (10:17 IST), valid |

Ledger after day 2: 104.0 s used (pilots 22 s, campaign 82 s), 0 pending. All four sessions
match the tagged config hash; run-path code is unchanged between prereg-v1 and 8ebe0ed; the
tag (27 Oct 03:25 UTC) precedes session 1 (03:35 UTC).
Session 2 ran in the evening (deviation logged). Session 3 ran on 4 Nov in the morning slot.
Ledger after day 3: 145.0 s used (campaign 123 s), 0 pending. Sessions 4 and 5: mornings of
7 and 9 Nov (10 Nov backup); the window ends 10 Nov.

First look (fez s1, descriptive only, no design changes): hardware R0 accuracy is 4-19 points
below simulator; measured readout mitigation (R3m) recovers 2-5 points; split conformal
calibrated on the simulator under-covers (0.69-0.91), ACI stays near 0.90.

## Paper (Task 11), 30 Oct 2026
- paper/ skeleton (IEEEtran conference) with guidance comments per section; numbers via
  \val{key}; `uv run qdrift paper --build` regenerates 7 tables, 6 figures, 134 numbers and the
  IEEE-formatted bibliography, then builds paper/build/main.pdf with tectonic (5 pages, no prose).
- scripts/check_paper.py: citations, labels, \val keys, TODOs, page count; `--overlap` marks
  8-word runs shared with the 38 downloaded papers (verified on a copied QuantumNAT sentence).
- Observation for Limitations: IBM refreshes reported properties often, so "calibration age"
  at job start is only 0.2-0.4 h on all runs so far.
- Preliminary (2 sessions): emulator over hardware +7.4 points [4.4, 10.5]; R3m best repair
  (+1.5 points); R1 normalisation hurts (-0.9 points). Not final; no design change.

## Paper drafts, 31 Oct 2026
- At the user's request the assistant drafted Introduction, Related Work, Methods and the
  design-based Discussion paragraphs (limitations, deviations). Each file is marked as an AI
  draft; an AI-use sentence is in the Acknowledgment (IEEE requires disclosure).
- Claims about cited papers were checked against their abstracts or text.
- Preliminary after 4 sessions (not final): H1 supported (EMU - R0 +7.4 points), H2 supported
  (variance ratio 3.3), H3 not supported, H4 largest term is CZ error, H5 not supported
  (CP_s1 over-covers at 0.93 instead of under-covering).

## Next
1. User: sessions 3-5 on both devices at about 09:00 IST on 2, 5 and 8 Nov; `uv run qdrift
   fetch` later the same day.
2. User: revise the AI drafts in their own words; reading log.
3. Assistant, after session 5: final analysis, then Results, the interpretation part of the
   Discussion, Abstract and Conclusion; cut to 8 pages (drop Fig. 1, Fig. 4 to a sentence,
   pilot table to text).
4. Optional: push the repo and tag to GitHub for a public, verifiable pre-registration date.
3. Commit and tag the pre-registration (ask the user first).
4. `uv run qdrift noise-aware --device <name>` right before session 1, then 5 sessions per
   device over 10-14 days.
5. Reading log: 42 entries in literature/reading-log.md; `scripts/check_reading_log.py`
   lists open items. Bib entries marked VERIFY need author lists checked.

## 12 Nov 2026, 23:30 IST: campaign complete
- 10 of 10 sessions (ibm_fez 5, ibm_kingston 5), all valid, config hash = prereg-v1; 227 s of 1,200 s used.
- Final numbers (macros in paper/generated/numbers.tex): gap 9.2 points, emulator +5.5, variance ratio 3.8,
  R3m +1.0, R4 +0.8, R2_16 +0.3 (interval starts at 0), ACI coverage 0.901.
- PDFs: paper/Interim_Report_Intro_LitReview_ProblemStatement.pdf (5 pages) and
  paper/Final_Research_Paper.pdf (11 pages incl. references). Group paper: Pankaj Baid, Kartikey Gupta,
  Shreshtha Gupta. Open: repository URL and DOI, authors revise the AI drafts, confirm affiliation/e-mails.
