# Ames Housing Pipeline

I built this project to demonstrate how I stand up an end-to-end modeling workflow on top of the classic Ames housing dataset. The repo balances exploratory flexibility with the guardrails I expect in production: reproducible configs, isolated feature engineering, and a serving endpoint that mirrors what I would deploy behind an internal service.

## Repository Structure

- `src/ames/` – modular code for data access, validation, engineered features, and model orchestration.
- `app/` – FastAPI app exposing `/predict` and `/healthz` routes for lightweight scoring smoke tests.
- `configs/` – YAML-driven configuration for data locations, model hyperparameters, and evaluation targets.
- `docker/` – container definition for shipping the API and model artifact together.
- `notebooks/`, `reports/` – reserved for exploratory analysis and generated diagnostics.

## Getting Started

1. Create a virtual environment and install dependencies: `make install`.
2. Review `configs/pipeline.yaml` and update the data paths to match your local layout.
3. Select a baseline model (`random_forest` or `elastic_net`) and adjust hyperparameters or tuning grids as needed.
4. Run `make train` to execute the training pipeline and persist intermediate outputs under `artifacts/`.
5. Launch the scoring API with `make serve` and POST sample payloads to `/predict` for a quick sanity check.

## Configuration Notes

- `model.name` maps to the corresponding scikit-learn estimator and can be swapped without code changes.
- `model.params` is passed straight into the estimator constructor, so any native keyword argument is supported.
- Optional `model.tuning` hooks into `GridSearchCV` or `RandomizedSearchCV` for quick hyperparameter sweeps. Example:

```yaml
model:
  name: random_forest
  params:
    n_estimators: 200
    max_depth: 8
  tuning:
    strategy: grid
    cv: 3
    scoring: neg_root_mean_squared_error
    param_grid:
      n_estimators: [200, 400]
      max_depth: [6, 8, null]
```

## Current Focus

- Expanding `DataIngestor` to manage raw data downloads and versioned snapshots.
- Filling out `FeatureBuilder` and `GeoFeatureEngineer` with the engineered features I use in notebooks.
- Swapping the placeholder training hooks with the real estimators and model persistence.
- Wiring `ModelEvaluator` to log out-of-sample metrics once full predictions are available.
