"""
FastAPI application for serving the Ames housing model with a lightweight UI.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.ames.data.ingestion import DataIngestor
from src.ames.data.validation import DataValidator
from src.ames.features.feature_builder import FeatureBuilder
from src.ames.features.geo_features import GeoFeatureEngineer
from src.ames.utils.config import load_config


class ModelService:
    """
    Handles model loading, feature defaults, and prediction helpers.
    """

    def __init__(self, config_path: str, top_n_features: int = 10):
        self.config_path = config_path
        self.top_n_features = top_n_features

        self.config: Dict[str, Any] = {}
        self.model = None
        self.feature_frame: pd.DataFrame = pd.DataFrame()
        self.feature_columns: List[str] = []
        self.feature_baseline: pd.Series = pd.Series(dtype=float)
        self.binary_features: set[str] = set()
        self.integer_features: set[str] = set()
        self.top_features: List[str] = []
        self.field_metadata: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load_resources(self):
        """
        Load configuration, model artifacts, and feature statistics.
        """
        self.config = load_config(self.config_path)
        self._build_feature_matrix()
        self._load_model_artifact()
        self._prepare_feature_defaults()
        self._select_top_features()

    def predict(self, overrides: Dict[str, Any] | None = None) -> float:
        """
        Run a prediction using the trained model, applying overrides on top
        of the baseline feature vector.
        """
        normalized = self._normalize_overrides(overrides)
        vector = self._build_feature_vector(normalized)
        frame = pd.DataFrame([vector], columns=self.feature_columns)
        prediction = float(self.model.predict(frame)[0])
        return prediction

    def form_fields(self, overrides: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """
        Prepare metadata for rendering the HTML form.
        """
        overrides = overrides or {}
        fields: List[Dict[str, Any]] = []
        for feature in self.top_features:
            meta = self.field_metadata[feature].copy()
            value = overrides.get(feature, meta["default"])
            meta["value"] = self._format_for_input(feature, value)
            if meta["input_type"] == "select":
                options = []
                for option in meta["options"]:
                    opt_copy = option.copy()
                    opt_copy["value"] = str(opt_copy["value"])
                    opt_copy["selected"] = opt_copy["value"] == meta["value"]
                    options.append(opt_copy)
                meta["options"] = options
            fields.append(meta)
        return fields

    def parse_form_inputs(
        self, form_data: Iterable[Tuple[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Convert submitted form data into numeric overrides.
        """
        overrides: Dict[str, Any] = {}
        errors: List[str] = []
        incoming = dict(form_data)
        for feature in self.top_features:
            raw_value = incoming.get(feature)
            try:
                overrides[feature] = self._coerce_value(feature, raw_value)
            except ValueError as exc:
                label = self.field_metadata.get(feature, {}).get("label", feature)
                errors.append(f"{label}: {exc}")
        return overrides, errors

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_feature_matrix(self):
        """
        Recreate the feature matrix used during training for defaults/statistics.
        """
        ingestor = DataIngestor(self.config)
        validator = DataValidator(self.config)
        feature_builder = FeatureBuilder(self.config)
        geo_engineer = GeoFeatureEngineer(self.config)

        raw = ingestor.download_source_data()
        validated = validator.basic_checks(raw)
        validated = validator.process_missing_values(validated)
        validated = validator.process_outliers(validated)
        basic_features = feature_builder.make_basic_features(validated)

        try:
            geo_features = geo_engineer.compute_ndvi()
            combined = geo_engineer.join_to_features(basic_features, geo_features)
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[ModelService] Skipping geo features: {exc}")
            combined = basic_features

        drop_targets = [col for col in ("SalePrice",) if col in combined.columns]
        features = combined.drop(columns=drop_targets)
        features = features.fillna(0)

        self.feature_frame = features
        self.feature_columns = features.columns.tolist()

    def _load_model_artifact(self):
        """
        Load the trained estimator from disk.
        """
        artifact_path = (
            self.config.get("paths", {}).get("model_artifact")
            or "models/trained_model.pkl"
        )
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(f"Model artifact not found at {artifact!s}")

        with artifact.open("rb") as handle:
            self.model = pickle.load(handle)

    def _prepare_feature_defaults(self):
        """
        Compute baseline feature values using median/mode statistics.
        """
        medians = self.feature_frame.median(numeric_only=True)
        modes = self.feature_frame.mode(dropna=True)
        defaults: Dict[str, float] = {}
        for column in self.feature_columns:
            if column in medians.index and pd.notna(medians[column]):
                defaults[column] = float(medians[column])
            else:
                fallback = modes[column].iloc[0] if column in modes else 0.0
                defaults[column] = float(fallback)
        self.feature_baseline = pd.Series(defaults, index=self.feature_columns)

        self.binary_features = {
            col
            for col in self.feature_columns
            if set(self.feature_frame[col].dropna().unique()).issubset({0, 1})
        }
        self.integer_features = {
            col
            for col in self.feature_columns
            if pd.api.types.is_integer_dtype(self.feature_frame[col])
        } - self.binary_features

    def _select_top_features(self):
        """
        Determine the top-N features based on model importance or variance.
        """
        importances = getattr(self.model, "feature_importances_", None)
        if importances is not None:
            series = pd.Series(importances, index=self.feature_columns)
            top = series.sort_values(ascending=False).head(self.top_n_features).index
        else:
            # fall back to variance in case the model does not expose importances
            variances = self.feature_frame.var().sort_values(ascending=False)
            top = variances.head(self.top_n_features).index

        self.top_features = list(top)
        self.field_metadata = {
            feature: self._build_field_metadata(feature) for feature in self.top_features
        }

    def _build_field_metadata(self, feature: str) -> Dict[str, Any]:
        """
        Assemble metadata used for rendering form controls.
        """
        series = self.feature_frame[feature]
        default = float(self.feature_baseline[feature])
        is_binary = feature in self.binary_features
        is_integer = feature in self.integer_features

        min_value = float(series.min())
        max_value = float(series.max())
        median = float(series.median())
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))

        metadata: Dict[str, Any] = {
            "name": feature,
            "label": self._format_label(feature),
            "default": default,
            "input_type": "select" if is_binary else "number",
            "step": "1" if (is_binary or is_integer) else "0.1",
            "min": "0" if is_binary else self._format_for_input(feature, min_value),
            "max": "1" if is_binary else self._format_for_input(feature, max_value),
            "help": f"Median {median:,.0f} | Typical range {q1:,.0f} – {q3:,.0f}",
            "is_integer": is_integer,
        }

        if is_binary:
            metadata["options"] = [
                {"label": "Yes", "value": 1},
                {"label": "No", "value": 0},
            ]

        return metadata

    def _normalize_overrides(
        self, overrides: Dict[str, Any] | None
    ) -> Dict[str, float]:
        """
        Keep only features recognized by the model and coerce into floats.
        """
        overrides = overrides or {}
        normalized: Dict[str, float] = {}
        for feature, raw_value in overrides.items():
            if feature not in self.feature_columns:
                continue
            normalized[feature] = self._coerce_value(feature, raw_value)
        return normalized

    def _build_feature_vector(self, overrides: Dict[str, float]) -> pd.Series:
        """
        Merge overrides into the baseline feature vector.
        """
        vector = self.feature_baseline.copy()
        for feature, value in overrides.items():
            vector.loc[feature] = value
        return vector.reindex(self.feature_columns, fill_value=0.0)

    def _coerce_value(self, feature: str, raw_value: Any) -> float:
        """
        Convert incoming values (string/number) into floats compatible with the model.
        """
        default = float(self.feature_baseline.loc[feature])
        if raw_value is None:
            return default

        if isinstance(raw_value, str):
            stripped = raw_value.strip()
            if stripped == "":
                return default
            lowered = stripped.lower()
            if lowered in {"yes", "true", "on"}:
                raw_value = "1"
            elif lowered in {"no", "false", "off"}:
                raw_value = "0"
            else:
                raw_value = stripped

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            label = self.field_metadata.get(feature, {}).get("label", feature)
            raise ValueError(f"enter a numeric value for {label}") from exc

        if feature in self.binary_features:
            return float(int(round(value)))
        if feature in self.integer_features:
            return float(int(round(value)))
        return value

    def _format_label(self, feature: str) -> str:
        """
        Create a human-friendly label for the form input.
        """
        return feature.replace("_", " ").strip()

    def _format_for_input(self, feature: str, value: float) -> str:
        """
        Format numeric values for HTML input fields.
        """
        if feature in self.binary_features:
            return "1" if value >= 0.5 else "0"
        if feature in self.integer_features:
            return str(int(round(value)))
        if abs(value) >= 1000:
            return f"{value:.0f}"
        return f"{value:.2f}"


# ---------------------------------------------------------------------- #
# FastAPI application
# ---------------------------------------------------------------------- #

app = FastAPI(title="Ames Housing Model API")
templates = Jinja2Templates(directory="app/templates")
service = ModelService(config_path="configs/pipeline.yaml")


@app.on_event("startup")
def startup_event():
    service.load_resources()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    fields = service.form_fields()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "fields": fields,
            "prediction": None,
            "errors": [],
        },
    )


@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request):
    form_data = await request.form()
    overrides, errors = service.parse_form_inputs(form_data.multi_items())
    prediction = None
    if not errors:
        prediction = service.predict(overrides)

    fields = service.form_fields(overrides if not errors else None)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "fields": fields,
            "prediction": prediction,
            "errors": errors,
        },
    )


@app.post("/predict")
def predict(payload: Dict[str, Any] | None = None):
    prediction = service.predict(payload or {})
    return {"prediction": prediction}


@app.get("/healthz")
def health_check():
    return {"status": "ok"}
