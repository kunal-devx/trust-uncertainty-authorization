# Trust Uncertainty Authorization

Controlled benchmark of evidence-based trust-estimation uncertainty for graduated LLM-agent authorization.

## Research scope

Research question:

> Under controlled evidence sparsity, does explicitly incorporating evidence-based trust-estimation uncertainty into graduated runtime authorization change the safety–unnecessary-intervention frontier relative to point-estimate trust gating?

The primary comparison is:

- **B1:** posterior trust mean, `T`.
- **B2:** clipped lower confidence bound, `T - z * sqrt(U)`, where `U` is exact Beta-posterior variance.
- **B3:** secondary illustrative synthetic exogenous uncertainty-penalty control; it is not real or empirically calibrated model confidence.

The frozen primary conclusion is **INCONCLUSIVE**: the B2 UI effect changes direction across operating points after independent threshold recalibration. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the handoff state and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact commands.

## Repository layout

```text
src/          Frozen trust, authorization, population, calibration, metrics, and analysis utilities
experiments/  Pilot, primary, B3, sensitivity, and downstream analysis entrypoints
configs/      Explicit pilot, primary, and sensitivity configurations
tests/        Unit, integrity, reproducibility, and analysis tests
figures/      Tracked paper and diagnostic figures
results/      Generated raw and analysis outputs; ignored by Git
```

## Environment

The validated environment uses Python `3.12.14` with:

```text
numpy==2.5.3
scipy==1.18.1
PyYAML==6.0.3
matplotlib==3.11.2
pytest==9.1.1
```

Create an environment and install the runtime/test dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy PyYAML matplotlib pytest
```

All commands below use the repository virtualenv explicitly and set `PYTHONPATH=.`.

## Tests

Run the complete suite:

```bash
PYTHONPATH=. ./.venv/bin/python -m pytest -q
```

The manuscript-freeze audit passed **291 tests**.

## Pipeline commands

The commands below regenerate ignored outputs. Review the configuration and preserve the frozen seed protocol before running production work.

Pilot:

```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_pilot.py
```

Primary calibration, then primary evaluation:

```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_calibration.py
PYTHONPATH=. ./.venv/bin/python experiments/run_primary.py
```

Primary downstream analysis and figures:

```bash
PYTHONPATH=. ./.venv/bin/python experiments/analyze_primary.py
PYTHONPATH=. ./.venv/bin/python experiments/diagnose_decision_decomposition.py
PYTHONPATH=. ./.venv/bin/python experiments/analyze_combined_mechanism.py
PYTHONPATH=. ./.venv/bin/python experiments/make_paper_figures.py
```

B3 diagnostic and descriptive analysis:

```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_b3_diagnostic.py
PYTHONPATH=. ./.venv/bin/python experiments/analyze_b3.py
```

Sensitivity conditions and analysis:

```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_sensitivity.py
PYTHONPATH=. ./.venv/bin/python experiments/analyze_sensitivity.py
```

Do not use B3 to revise the primary conclusion. Do not run sensitivity by changing `configs/primary.yaml`.

## Frozen primary design

- Beta prior: `Beta(1,1)`
- `z = 1.0`
- `p_catch = 0.80`
- Profiles: reliable `.05`, unreliable `.40`, malicious `.80`
- Evidence counts: `0, 1, 2, 3, 5, 8, 12, 20, 40, 100`
- 30 agents per profile/evidence cell, 900 agents per seed
- 50 measurement actions per agent
- Calibration seeds: `10001`–`10010`
- Evaluation seeds: `20001`–`20020`
- Targets: `0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15`
- Threshold grid: `0.05`–`0.95` by `0.05`, with `theta_reject < theta_verify`
- Statistical unit: seed; B1/B2 evaluation is paired

## Results and verification

Generated raw results are intentionally ignored. Known frozen artifact hashes are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). A fresh clone must regenerate or separately restore ignored result artifacts before running downstream analyses.

Current tracked figures are reproducibility outputs, not raw data. The B3 analysis remains descriptive and does not claim real model-confidence behavior.

## Manuscript status

No manuscript or venue submission is included in this repository. The repository is at a reproducibility and research-handoff milestone; see [PROJECT_STATUS.md](PROJECT_STATUS.md).
