# Reproducibility Handoff

This document describes the audited pipeline for `trust-uncertainty-authorization`. It records the commands and boundaries used by the frozen research implementation. Generated result data are intentionally not part of the Git checkout.

## Tested environment

The audited environment used:

```text
Python 3.12.14
numpy==2.5.3
scipy==1.18.1
PyYAML==6.0.3
matplotlib==3.11.2
pytest==9.1.1
```

Create a clean environment with:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy PyYAML matplotlib pytest
```

The repository uses no `requirements.txt` or packaging metadata at this milestone. The dependency list above is the tested environment and follows the imports used by the source and experiment scripts.

## Pipeline

The implementation flows as follows:

```text
trust_model.py
  -> population.py
  -> calibration.py
  -> authorization.py / verify_sim.py
  -> metrics.py
  -> stats.py / curves.py
  -> downstream analysis and figures
```

### Core modules

- `src/trust_model.py`: Beta-Bernoulli posterior mean, exact posterior variance, and clipped LCB.
- `src/population.py`: controlled profiles, evidence cells, posterior parameters, and fixed measurement-action realizations.
- `src/authorization.py`: deterministic ALLOW/VERIFY/REJECT thresholds.
- `src/verify_sim.py`: benign actions complete; harmful VERIFY actions are caught with `p_catch`.
- `src/calibration.py`: threshold-pair evaluation and deterministic HAER-based selection.
- `src/metrics.py`: HAER, UI, false VERIFY, false REJECT, and legitimate completion.
- `src/stats.py`: paired seed-level UI bootstrap and secondary Wilcoxon utility.
- `src/curves.py`: descriptive operating-point matching.
- `src/sensitivity.py`: validated sensitivity configuration containers.

### Configurations

- `configs/pilot.yaml`: small pipeline-validation configuration.
- `configs/primary.yaml`: frozen primary configuration.
- `configs/sensitivity.yaml`: one-factor robustness conditions; it does not alter `primary.yaml`.

## Commands

Use the repository root as the working directory.

### Tests

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest -q
```

The final audit passed 291 tests.

### Pilot

```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_pilot.py
```

Pilot output is written under `results/` and is ignored.

### Primary experiment

Review `configs/primary.yaml` before production execution. Calibration and evaluation are separate commands:

```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_calibration.py
PYTHONPATH=. ./.venv/bin/python experiments/run_primary.py
```

Calibration writes `results/primary/calibration.json`; evaluation writes `results/primary/evaluation.json`. The primary runner validates calibration metadata and its deterministic fingerprint before using a cached calibration artifact. It refuses stale or mismatched cached calibration.

Primary RNG construction uses explicit `numpy.random.SeedSequence` children. Calibration and evaluation seeds are disjoint. B1 and B2 share population, action, ground-truth, and VERIFY realizations for each seed.

### Primary analysis

```bash
PYTHONPATH=. ./.venv/bin/python experiments/analyze_primary.py
PYTHONPATH=. ./.venv/bin/python experiments/diagnose_decision_decomposition.py
PYTHONPATH=. ./.venv/bin/python experiments/analyze_combined_mechanism.py
PYTHONPATH=. ./.venv/bin/python experiments/make_paper_figures.py
```

These commands consume existing raw outputs. They do not perform new primary evaluation. Primary analysis uses seed-level paired UI bootstrap with an explicit analysis seed; cell and profile/evidence summaries are descriptive.

### B3 diagnostic

```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_b3_diagnostic.py
PYTHONPATH=. ./.venv/bin/python experiments/analyze_b3.py
```

B3 is a secondary synthetic exogenous uncertainty-penalty control:

```text
v ~ Beta(2,5)
score_B3 = clip(T - z*sqrt(v), 0, 1)
```

It is not real model confidence, empirical confidence calibration, or a replacement for B2. B3 outputs belong under `results/diagnostic/b3/`.

### Sensitivity

```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_sensitivity.py
PYTHONPATH=. ./.venv/bin/python experiments/analyze_sensitivity.py
```

The current sensitivity protocol is one-factor-at-a-time around the primary design: Beta(2,2), `z=0.5`, `z=1.5`, `p_catch=0.6`, `p_catch=0.95`, and threshold-grid step `0.025`. The analyzer validates the seven condition files and does not include its own `analysis.json` as a condition.

Sensitivity output is descriptive and declares `inference_performed = false`.

## Output locations

Generated output locations are:

```text
results/pilot_results.json
results/primary/calibration.json
results/primary/evaluation.json
results/primary/analysis.json
results/primary/analysis_summary.txt
results/diagnostic/decision_decomposition.json
results/diagnostic/b3/b3_diagnostic.json
results/diagnostic/b3/b3_analysis.json
results/diagnostic/combined_mechanism_analysis.json
results/sensitivity/*.json
results/sensitivity/analysis.json
results/sensitivity/analysis_summary.txt
figures/primary/
figures/diagnostic/b3/
figures/paper/
```

Raw results and most generated analyses are ignored by `.gitignore` because they are reproducible, potentially large artifacts. Tracked figures are compact presentation artifacts; regenerate them only when the source analysis inputs are available and unchanged.

## Seed and pairing protocol

Primary calibration seeds are `10001` through `10010`; evaluation seeds are `20001` through `20020`. They must remain disjoint.

For each seed, the experiment path creates:

```python
ss = np.random.SeedSequence(seed)
population_ss, verify_ss = ss.spawn(2)
population_rng = np.random.default_rng(population_ss)
```

Explicit child sequences are used for VERIFY streams. B1/B2 use the same population, evidence history, measurement actions, and VERIFY outcomes. The independent experimental unit is the seed, not an action, agent, or cell.

## Frozen hashes

The audited frozen primary artifacts were:

```text
results/primary/calibration.json
ba6f1e1977fd00008663c44e55663ebdd1b55be2269e3a863e5ba4b100849d10

results/primary/evaluation.json
af57e7427d637192ea44fb998c6cdb02f29261e3e4e92cb8422218524841f5ac

results/primary/analysis.json
b81ab5ef4a3e840192f59e6e5a6bd6442eb15c1af12c251b2db2d606a5d66ad3
```

Verify an artifact with:

```bash
sha256sum results/primary/calibration.json
sha256sum results/primary/evaluation.json
sha256sum results/primary/analysis.json
```

These hashes are provenance references for the audited local artifacts. Because `results/` is ignored, a fresh clone does not contain them unless they are restored separately or regenerated.

## Reproduction cautions

- Do not alter `configs/primary.yaml` to run a sensitivity condition.
- Do not call a target exactly achieved when the threshold grid only provides a closest grid point.
- Do not interpret B3 as real model confidence.
- Do not treat actions or cells as independent observations.
- Do not use the primary result to claim B2 superiority: the frozen primary conclusion is INCONCLUSIVE.
- Do not commit ignored raw results merely to make a checkout appear complete; preserve hashes and provenance when archiving artifacts externally.
