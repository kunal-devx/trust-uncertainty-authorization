# Project Status

Status date: 2026-09-14

## Current research question

Under controlled evidence sparsity, does explicitly incorporating evidence-based trust-estimation uncertainty into graduated runtime authorization change the safety–unnecessary-intervention frontier relative to point-estimate trust gating, and through what mechanism?

## Current hypotheses

Primary hypothesis:

> At matched aggregate harmful-action execution rate, uncertainty-aware mechanism B2 achieves significantly lower aggregate unnecessary-intervention rate than point-estimate mechanism B1.

Current frozen result: **H1 is not supported. The primary result is INCONCLUSIVE.**

B3 is not part of the primary hypothesis. It is a descriptive synthetic exogenous uncertainty-penalty control.

## Frozen methodology

- B1 score: posterior trust mean `T`.
- B2 score: clipped `T - z * sqrt(U)`.
- `U`: exact Beta-posterior variance.
- Primary prior: Beta(1,1).
- Primary `z`: `1.0`.
- Primary `p_catch`: `0.80`.
- Profiles: reliable `.05`, unreliable `.40`, malicious `.80`.
- Evidence counts: `0, 1, 2, 3, 5, 8, 12, 20, 40, 100`.
- `K = 30` agents per profile/evidence cell; 900 agents per seed.
- `M = 50` measurement actions per agent.
- Calibration seeds: `10001`–`10010`.
- Evaluation seeds: `20001`–`20020`.
- Target HAER grid: `0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15`.
- Threshold grid: `0.05`–`0.95` by `0.05`, with strict reject/verify ordering.
- Statistical unit: seed.
- Trust is a fixed pre-measurement snapshot; there are no measurement-time updates.

## Complete milestones

### COMPLETE

- Core Beta-Bernoulli trust model.
- Deterministic graduated authorization.
- Controlled population generation.
- VERIFY simulation.
- HAER/UI metrics.
- Paired calibration and primary runner.
- Pilot pipeline.
- Primary calibration and 20-seed evaluation.
- Primary seed-level statistical and curve analysis.
- Primary decision decomposition.
- B3 production diagnostic and descriptive analysis.
- Combined B1/B2/B3 mechanism analysis.
- One-factor sensitivity batch and descriptive analysis.
- Publication figure-generation script.
- Primary integrity, B3 integrity, and sensitivity tests.

### FROZEN

- `configs/primary.yaml`.
- Primary calibration and evaluation artifacts.
- Primary analysis and conclusion.
- Primary thresholds and seed protocol.
- B3 interpretation as synthetic exogenous uncertainty only.
- Sensitivity outputs as robustness context, not replacement primary results.
- Seed-level statistical unit.

### OPTIONAL FUTURE WORK

- Manuscript writing and venue-specific formatting.
- External archival of ignored raw results with recorded hashes.
- A separately approved B3 profile/evidence diagnostic if cell-level B3 mechanism analysis is needed.
- Additional controlled environments or action-risk variants, clearly labeled as new experiments.
- Empirical model-confidence measurements, if a future study has a defensible data source.

## Major findings

1. B2’s UI effect is mixed across the operating curve: it is lower than B1 at targets `0.01`, `0.02`, `0.03`, and `0.10`, and higher at `0.05`, `0.08`, and `0.15`.
2. Decision decomposition attributes the mixed pattern primarily to changes in benign VERIFY versus benign REJECT composition after independent threshold recalibration.
3. Reliable agents with sparse evidence are the strongest B1/B2 descriptive mechanism region.
4. B3 is useful only as an illustrative synthetic exogenous uncertainty control; it does not establish anything about real model confidence.
5. Sensitivity conditions remain mixed/sensitivity-dependent and do not change the primary INCONCLUSIVE conclusion.

## Known limitations

- Synthetic behavioral environment rather than real LLM agents.
- Fixed trust snapshot during measurement.
- Homogeneous primary action risk.
- Discrete threshold grid and closely matched rather than exactly equal achieved HAER.
- B3 uses prespecified synthetic Beta(2,5) proxy values, not empirical confidence.
- Inference uses seed-level pairing; actions and cells are not independent statistical units.
- Cold-start sensitivity boundaries are derived from retained primary cell results, not separate full reruns.
- The current contribution is empirical benchmark characterization and mechanism analysis, not a claim of algorithmic novelty in Bayesian trust, trust-based authorization, uncertainty-aware authorization, or graduated enforcement.

## What must not change casually

- Primary seeds, thresholds, target grid, prior, `z`, `p_catch`, profiles, evidence counts, `K`, or `M`.
- Primary calibration/evaluation outputs.
- The primary INCONCLUSIVE conclusion.
- The paired realization design.
- The seed-level statistical unit.
- B3’s synthetic/non-empirical interpretation.
- The separation between ignored generated data and tracked source/configuration/tests.

Any change to these requires a new explicitly labeled experiment and new provenance artifacts, not an edit to the frozen primary pipeline.

## Exact next steps for a future researcher

1. Clone the repository and read [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
2. Create the tested Python environment and run the full test suite.
3. Restore archived ignored results or regenerate them from the frozen configs.
4. Verify the three primary SHA-256 hashes before interpreting downstream analysis.
5. Review the existing primary, B3, sensitivity, and combined analysis artifacts.
6. Draft the manuscript around the controlled benchmark, mixed recalibrated frontier behavior, and mechanism evidence.
7. Keep B3 in a secondary/appendix role unless a new empirical confidence design is approved.
8. Record any new experiment in a separate configuration and output directory; never overwrite `results/primary/`.

## Manuscript status

No manuscript or venue submission is included. This repository is at a reproducibility and research-handoff milestone, ready for review and manuscript preparation but not claiming publication or venue acceptance.
