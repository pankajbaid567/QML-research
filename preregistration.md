# Pre-registration (draft; freeze with a git tag after the pilot, before the campaign)

Study: noise drift of simulator-trained quantum classifiers on IBM Heron processors, and
low-cost repairs. Frozen settings live in `config/experiment.toml`; its SHA-256 is stored in
every session's `meta.json`.

## Design
- Devices: ibm_fez and ibm_kingston (Heron r2; fallback ibm_marrakesh, which was in maintenance
  with 2,020 queued jobs when the pilots ran), Open Plan, one job per session.
- Five sessions per device over 10 to 14 days, one job per session, similar time of day.
- Layout: 14 disjoint 4-qubit linear patches per device, frozen in `data/layouts/` before the
  first campaign session and never changed afterwards.
- Models: M1 data re-uploading classifier (Z), M2 projected single-qubit X/Y/Z features of a
  fixed map, M3 quantum extreme learning machine (Z and ZZ); each with a logistic head.
- Data: two_curves (Bowles, Ahmed and Schuld generator) and breast_cancer (PCA to 4 features).
  Per session and dataset: 32 labelled calibration samples and the same 150 test samples.
- Shots: 500 per circuit; 2,000 per readout-calibration circuit. Sampler twirling and
  dynamical decoupling are off, so raw device noise is measured.
- Budget: at most 400 QPU seconds for the campaign; hard stop at 1,200 s across all accounts.

## Primary outcomes
Test accuracy of each (device, session, dataset, model, method), plus ECE, Brier score,
ROC-AUC, conformal coverage and set size at alpha = 0.1.

## Hypotheses and decision rules (95% bootstrap intervals, 2,000 resamples)
- H1: the calibration emulator overestimates hardware accuracy. Supported if the mean of
  (emulator - hardware R0 accuracy) over all points has a lower bound above 0.
- H2: session-to-session variation exceeds shot noise. For each test sample, the ratio of
  across-session feature variance to expected shot variance (1 - F^2)/shots is averaged over
  features. Supported if the mean ratio over samples has a lower bound above 1 (its expected
  value is 1 under pure shot noise).
- H3: refitting the head on 16 labelled hardware samples (R2_16, penalty 0.25 k toward the
  simulator head) beats batch normalisation (R1). Supported if the mean paired accuracy
  difference has a lower bound above 0.
- H4: patch readout error is the strongest calibration predictor of feature error.
  Supported if readout has the largest absolute standardised coefficient in the HC3 OLS.
- H5: split conformal calibrated in session 1 under-covers later sessions while adaptive
  conformal inference holds. Supported if the CP_s1 coverage upper bound in sessions 2-5 is
  below 0.9 and the mean ACI coverage is within 0.03 of 0.9.

## Exclusions and deviations
- A session is excluded only if its job fails or its data fail `qdrift validate`; it is then
  rerun from the `rerun` budget and the exclusion is reported.
- If a device is unavailable for more than 3 days, the fallback device replaces it for the
  remaining sessions, reported as a deviation.
- Any change to the config after the tag is a deviation and is listed in the paper.

## Pilot results
- ibm_fez (job dbfn4u0m94ac738ttrlg, 26 Oct 2026): 11 s for 34,000 executions, so
  0.000265 s per execution after the 2 s job overhead (IBM quick estimate 0.00035 s).
- ibm_fez packed vs isolated, 28 two_curves samples: excess total-variation distance over the
  shot-noise null M1 [-0.002, 0.019], M3 [-0.009, 0.006]; accuracy difference M1
  [-0.07, 0.14], M3 [-0.21, 0.14]. No crosstalk detectable at this size, so packing is kept.
- ibm_kingston (job dbfo7vom94ac738u0lkg, 26 Oct 2026, about 4.8 h in the queue): 11 s for
  34,000 executions, the same 0.000265 s per execution. Packed vs isolated: excess total-variation
  distance M1 [-0.013, 0.004], M3 [-0.012, 0.004]; accuracy difference M1 [-0.18, 0.07],
  M3 [-0.18, 0.18]. No crosstalk detectable at this size.
- Pilot accuracy on these 28 samples (simulator -> hardware, R0 head): ibm_fez M1 0.857 -> 0.821,
  M3 0.786 -> 0.643; ibm_kingston M1 0.857 -> 0.786, M3 0.786 -> 0.714. Small sample; indicative
  only, and pilot data are not used in the campaign analysis.
- Budget levers: none needed. Pilot total 22 s. A session is reserved at 25.4 s (1.25 safety
  factor); ten sessions are expected to use about 200-230 s of the 400 s campaign cap.
- The ibm_marrakesh pilot was cancelled while queued (device in maintenance); 0 s used.
