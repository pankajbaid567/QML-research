# Changes after the pre-registration (tag `prereg-v1`, commit 154984c)

Every change after the tag that touches the design, the data, the analysis or the references
is listed here and reported in the paper. "Design" changes alter what was pre-registered;
"fix" and "admin" entries do not. New tooling that leaves results unchanged (paper assets,
checks, PDF copies of figures) is in the git history only.

| Date | Change | Type | Effect on the pre-registered design |
| --- | --- | --- | --- |
| 27 Oct 2026 | References: removed Moradi et al. (arXiv:2206.01573), withdrawn by its authors in July 2022; author lists of 15 entries verified against arXiv/OpenAlex. | admin | None |
| 30 Oct 2026 | Analysis: calibration age (`age_h`) is now measured at job start for run-time snapshots. Before, the fetch time was used by mistake, which inflated the age for jobs fetched late. | fix | None; only the `age_h` covariate of the H4 regression changes |

| 30 Oct 2026 | Schedule: session 2 on both devices ran at 18:23-18:24 UTC (about 23:50 IST), while session 1 ran at 03:37-05:31 UTC. The design asked for a similar time of day. | design (timing) | Sessions 1 and 2 differ in time of day as well as date; reported in the paper. Sessions 3-5 are planned in the session-1 slot, about 09:00 IST (03:30 UTC), on 2, 5 and 8 Nov. |

## Campaign schedule notes
- Session 1 on both devices: 27 Oct 2026 (ibm_fez ran 03:37 UTC, ibm_kingston 05:31 UTC after
  a 1.9 h queue). No sessions on 28-30 Oct; the pre-registered window (5 sessions per device
  within 10-14 days, so by 10 Nov) still holds.
