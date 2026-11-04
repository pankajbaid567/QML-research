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
