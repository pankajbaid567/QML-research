# Paper drafting plan

## Done
- 31 Oct: Introduction, Related Work, Methods, and the limitations/deviations parts of the
  Discussion drafted (AI-assisted, marked in each file; AI-use sentence in the Acknowledgment).
- 4 Nov: publication-style figures (IEEE widths, 8 pt serif, embedded fonts); 95% intervals for
  repair gains; overdispersion table; about 160 new result macros for the Results text
  (see the comment block at the top of paper/sections/results.tex); pilot numbers moved from a
  table into the Methods text; the simulator/emulator/hardware bar chart dropped from the paper.

## After the last session (planned 9 Nov)
1. `uv run qdrift fetch`, `uv run qdrift validate`, `uv run qdrift analyze`, `uv run qdrift paper --build`.
2. Results: one subsection per RQ, numbers only through \val{}; then the pre-registered
   decisions (Table "hypotheses").
3. Discussion interpretation, Abstract, Conclusion; confirm the session times in the deviations
   paragraph.
4. Page budget (now 7 pages before Results, Abstract and Conclusion; expect about 8.3):
   - keep one of the coverage table and the coverage figure;
   - shorten the hypotheses table to one column, or move its statistics into the text;
   - trim background references (QOC, QuantumNAS, ZNE, CDR, learned mitigation, kernel
     concentration group) if still over 8 pages.
5. `scripts/check_paper.py --overlap`, then the user revises the AI drafts in their own words.

## Claims about cited papers
Only claims checked against abstracts or text read earlier. Keep other citations generic.

## 10 Nov 2026
- All sections drafted (AI-assisted): Results, Discussion implications, Abstract and Conclusion
  added. Every number is a \val{} macro; 21 directional claims are tagged `% claim:<id>` and
  re-checked by `uv run qdrift paper` (all hold on 6 sessions). PDF: 8 pages incl. references.
- QCE allows 8-10 pages plus 2 pages of references, so no cuts are needed.
- To finish after the last session: fetch, validate, analyze, `qdrift paper --build`; rewrite any
  sentence whose claim fails; fill the repository URL and DOI at release.

## 12 Nov 2026: course structure and interim report
- The course requires these sections: Abstract, Introduction, Literature Review, Problem
  Statement, Methodology, Results and Discussion, Conclusion. The teacher reviews Introduction,
  Literature Review and Problem Statement first (16 Nov).
- paper/interim.tex = only those three sections (+ references, acknowledgment); paper/main.tex =
  the complete paper. Both input the same files, so one edit updates both.
- New: sections/literature-review.tex (7 themed subsections, a comparison table of the closest
  studies, and the research gap), sections/problem-statement.tex (setting, quantities, RQ and
  hypothesis table, scope). Related Work was merged into the Literature Review; Methods became
  Methodology; Results and Discussion are one section.
- Every statement about another paper was checked against its abstract or text (literature/pdfs).
- Fixed: a sentence in the old Discussion lost its first words to a stray % comment.
