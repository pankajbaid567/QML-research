# Changes after the pre-registration (tag `prereg-v1`, commit 154984c)

Every change made after the tag is listed here and reported in the paper. "Design" changes
alter what was pre-registered; "fix" and "admin" entries do not.

| Date | Change | Type | Effect on the pre-registered design |
| --- | --- | --- | --- |
| 27 Oct 2026 | References: removed Moradi et al. (arXiv:2206.01573), withdrawn by its authors in July 2022; author lists of 15 entries verified against arXiv/OpenAlex. | admin | None |
| 30 Oct 2026 | Analysis: calibration age (`age_h`) is now measured at job start for run-time snapshots. Before, the fetch time was used by mistake, which inflated the age for jobs fetched late. | fix | None; only the `age_h` covariate of the H4 regression changes |

## Campaign schedule notes
- Session 1 on both devices: 27 Oct 2026 (ibm_fez ran 03:37 UTC, ibm_kingston 05:31 UTC after
  a 1.9 h queue). No sessions on 28-30 Oct; the pre-registered window (5 sessions per device
  within 10-14 days, so by 10 Nov) still holds.
