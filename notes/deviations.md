# Changes after the pre-registration (tag `prereg-v1`, commit 154984c)

Every change after the tag that touches the design, the data, the analysis or the references
is listed here and reported in the paper. "Design" changes alter what was pre-registered;
"fix" and "admin" entries do not. New tooling that leaves results unchanged (paper assets,
checks, PDF copies of figures) is in the git history only.

| Date | Change | Type | Effect on the pre-registered design |
| --- | --- | --- | --- |
| 27 Oct 2026 | References: removed Moradi et al. (arXiv:2206.01573), withdrawn by its authors in July 2022; author lists of 15 entries verified against arXiv/OpenAlex. | admin | None |
| 30 Oct 2026 | Analysis: calibration age (`age_h`) is now measured at job start for run-time snapshots. Before, the fetch time was used by mistake, which inflated the age for jobs fetched late. | fix | None; only the `age_h` covariate of the H4 regression changes |

| 30 Oct 2026 | Schedule: session 2 on both devices ran at 18:23-18:24 UTC (about 23:50 IST), while session 1 ran at 03:37-05:31 UTC. The design asked for a similar time of day. | design (timing) | Sessions 1 and 2 differ in time of day as well as date; reported in the paper. Sessions 3-5 run in the session-1 slot (morning IST); dates are in the schedule notes. |

## Campaign schedule notes
- Session 1 on both devices: 27 Oct 2026 (ibm_fez ran 03:37 UTC, ibm_kingston 05:31 UTC after
  a 1.9 h queue). No sessions on 28-30 Oct; the pre-registered window (5 sessions per device
  within 10-14 days, so by 10 Nov) still holds.
- Session 3 on both devices: 4 Nov 2026, morning slot (ibm_kingston ran 10:17 IST, ibm_fez
  10:37 IST). Planned for 2 Nov; moved by two days, still inside the window.
- Sessions 4 and 5: planned for the mornings of 7 and 9 Nov (10 Nov as backup), so the
  campaign spans 13-14 days.
- 10 Nov 2026, 20:42 IST (last day of the 14-day window): sessions 4 and 5 not run yet on either
  device (3 of 5 done). Whatever happens next is reported in the paper through generated
  sentences (window and session counts in paper/sections/discussion.tex).
- 10-11 Nov 2026: ibm_fez s4 and ibm_kingston s4 ran 10 Nov 22:39 IST (last day of the window).
  ibm_kingston s5 ran 11 Nov 01:34 IST, only 2.9 h after s4, so that pair measures variation within
  one night. ibm_fez s5 has not run. The reason is not confirmed; the pending-job guard would have blocked a
  second fez job while s4 was unfetched, but no output of that attempt was seen.
  Campaign span 14.7 days (registered maximum 14). All reported in the paper by generated sentences.
- 12 Nov 2026: two exploratory checks added (not registered; the registered statistics stay the
  primary results): H2 without sessions that start < 12 h after the previous one on the same device
  (ratio 3.53 vs 3.93 registered), and the H4 regression without the chains whose couplers the
  run-time calibration marked non-operational (2 couplers, ibm_kingston s3; readout error becomes
  the largest term, so the registered H4 decision rests on those inputs).
