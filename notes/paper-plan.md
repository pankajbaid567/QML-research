# Paper drafting plan (31 Oct 2026)

User asked the assistant to write the prose ("tum karo").

## Now (design is frozen, so these do not depend on results)
1. Introduction, Related Work, Methods: full drafts with \val{} numbers and \cite keys.
2. Add design macros (two_curves settings, pilot size, PUBs per session) to paper.py.
3. Remove \nocite{*}; add an AI-use disclosure to the Acknowledgment (IEEE policy requires it).
4. Build, then `scripts/check_paper.py --overlap` (target < 10% against the 38 PDFs).

## After the last session (about 6 Nov)
Results, Discussion, Abstract, Conclusion. They interpret the final data and the
pre-registered decisions, so drafting them now would mean writing conclusions before the data.

## Claims about cited papers
Only claims checked against abstracts or text read earlier (QuantumNAT, AQCP, QAOA drift,
QBioBench, cross-device transfer, Bowles et al.). Keep other citations generic.
