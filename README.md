# qdrift: noise drift of quantum classifiers on IBM Heron

Research code for a class paper (IEEE two-column). Quantum classifiers are trained in
simulation, then run on IBM Heron processors across several days and two devices. The study
measures how accuracy, calibration and conformal coverage change between simulator, emulator
and hardware, which calibration data explain the change, and which low-cost repairs recover it
per QPU-second. No quantum advantage is claimed; tuned classical baselines are reported.

## Setup

```bash
uv sync                      # Python 3.13, pinned dependencies
uv run pytest -q             # unit and fake-backend integration tests (about 2 minutes)
```

IBM credentials never enter this repository. Save them once to `~/.qiskit/qiskit-ibm.json`:

```bash
uv run python -c "from getpass import getpass; from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token=getpass('API key: '), instance=input('Instance CRN: '), name='qdrift-main', overwrite=True)"
cp config/accounts.example.toml config/accounts.toml   # git-ignored list of approved accounts
```

## Commands

| Command | QPU time |
| --- | --- |
| `uv run qdrift prepare` | none: datasets, classical baselines, simulator training |
| `uv run qdrift dryrun-campaign` | none: full rehearsal on fake backends with injected drift |
| `uv run qdrift backends` | none |
| `uv run qdrift layout --device ibm_fez` | none: freezes 14 qubit patches from calibration data |
| `uv run qdrift noise-aware --device ibm_fez` | none: emulated training data for repair R4 |
| `uv run qdrift estimate --device ibm_fez` | none: predicted usage of a pilot and a session |
| `uv run qdrift pilot --device ibm_fez` | yes, about 20 s; asks you to type RUN |
| `uv run qdrift session --device ibm_fez` | yes, about 30 s; asks you to type RUN |
| `uv run qdrift fetch` | none: saves jobs that finished after you stopped waiting |
| `uv run qdrift budget status` / `reconcile` / `audit` | none |
| `uv run qdrift analyze` / `validate` | none |
| `uv run qdrift paper --build` | none: tables, numbers and figures into `paper/generated/`, PDF via tectonic |

## Paper and literature

- `paper/main.tex` and `paper/sections/*.tex`: IEEE two-column skeleton. Comments in each
  section list what to cover and which papers to cite; you write the prose. Results are
  referenced with `\val{key}` from `paper/generated/numbers.tex`, never typed by hand.
- `uv run python scripts/check_paper.py --overlap`: citation, label, TODO and page checks, plus
  8-word overlap with the downloaded papers. `--draft file.txt` checks a single paragraph.
- `uv run python scripts/fetch_papers.py`: downloads open copies of `literature/refs.bib`
  into `literature/pdfs/` (git-ignored) and writes `literature/papers.csv` with links.
- `uv run python scripts/check_reading_log.py`: open items in `literature/reading-log.md`.

Add `--dry-run` and a `fake_*` device (for example `fake_fez`) to rehearse any command.

## QPU budget safety

Every hardware job is reserved in `data/budget/ledger.jsonl` before submission and settled with
the usage IBM reports. A job is refused if it would exceed 1,200 s across all approved accounts,
its phase cap (pilot 90 s, campaign 400 s, rerun 110 s, add-on 600 s), or 90% of the account's
rolling 28-day limit. Accounts missing from `config/accounts.toml` are always refused. IBM is
also asked to cancel any job that runs longer than 120 s.

## Layout

```
config/            experiment.toml (frozen settings), accounts.example.toml
src/qdrift/        circuits, data, training, device, session, repairs, metrics, analysis, CLI
scripts/           design checks and the reading-log checker
literature/        refs.bib and reading-log.md
data/processed/    datasets, trained parameters, simulator features
data/raw/          one folder per hardware session (counts, calibration snapshot, metadata)
results/           tables, hypotheses.json and figures
preregistration.md hypotheses and decision rules (tagged before the campaign)
```

## Status

Pre-registered on 26-27 Oct 2026 (tag `prereg-v1`). Pilots done on ibm_fez and ibm_kingston.
Campaign: session 1 done on both devices (27 Oct); four more per device to go. Changes after
the tag are listed in `notes/deviations.md`. The paper skeleton builds (`paper/build/main.pdf`).
