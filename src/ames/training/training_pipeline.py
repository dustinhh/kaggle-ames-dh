"""
Training pipeline that ties together data ingestion, validation, feature engineering, and model training.
"""

import os
import pickle
from numbers import Number
from pathlib import Path

MPLCONFIG_PATH = Path("reports/mplconfig")
MPLCONFIG_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_PATH.resolve()))

import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split

from ..data.ingestion import DataIngestor
from ..data.validation import DataValidator
from ..features.feature_builder import FeatureBuilder
from ..features.geo_features import GeoFeatureEngineer
from ..models.model_factory import HyperparameterTuner, ModelFactory
from ..utils.config import load_config
from .evaluation import ModelEvaluator


class TrainingPipeline:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.ingestor = DataIngestor(self.config)
        self.validator = DataValidator(self.config)
        self.feature_builder = FeatureBuilder(self.config)
        self.geo_feature_engineer = GeoFeatureEngineer(self.config)
        self.evaluator = ModelEvaluator(self.config.get("evaluation"))
        self.model_config = self.config.get("model") or {}
        self.model_factory = ModelFactory(self.model_config)
        self.tuner = HyperparameterTuner(self.model_config.get("tuning"))
        self.model = None
        self.tuning_report = {}

    def run(self):
        """
        Runs the end-to-end training pipeline.
        """
        raw_data = self.ingestor.download_source_data()
        staged_data = self.ingestor.save_locally(raw_data)
        validated_data = self.validator.basic_checks(staged_data)
        validated_data = self.validator.process_missing_values(validated_data)
        snapshot_frame = validated_data.copy()
        self._print_feature_summary(snapshot_frame)
        self._plot_saleprice_vs_sqft(snapshot_frame)
        validated_data = self.validator.process_outliers(validated_data)
        basic_features = self.feature_builder.make_basic_features(validated_data)

        geo_features = self.geo_feature_engineer.compute_ndvi()
        combined = self.geo_feature_engineer.join_to_features(
            basic_features, geo_features
        )
        # save locally for inspection
        combined.to_parquet("src/ames/data/features_combined.parquet", index=False)

        features = combined.drop(columns=["SalePrice"])
        features = features.fillna(0)
        targets = combined["SalePrice"]

        eval_cfg = self.config.get("evaluation") or {}
        test_size = eval_cfg.get("test_size", 0.2)
        random_state = eval_cfg.get("random_state", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            targets,
            test_size=test_size,
            random_state=random_state,
        )

        candidate = self.model_factory.create()
        trained_model, tuning_report = self.tuner.tune(candidate, X_train, y_train)
        if not tuning_report:
            trained_model.fit(X_train, y_train)
        self.model = trained_model
        self.tuning_report = tuning_report
        predictions = self.model.predict(X_test)
        metrics = self.evaluator.evaluate(predictions, targets=y_test)
        if tuning_report:
            print("Tuning summary:", tuning_report)
        print("Evaluation metrics:", metrics)
        self._print_presentation_metrics(metrics)
        self._plot_feature_importance(feature_names=features.columns)
        artifact_path = self.config["paths"].get(
            "model_artifact", "models/trained_model.pkl"
        )
        with open(artifact_path, "wb") as f:
            pickle.dump(self.model, f)
        return metrics

    def _print_presentation_metrics(self, metrics):
        """
        Emit a compact block that can be dropped into slides or notes.
        """
        if not metrics:
            return

        print("\nPerformance snapshot (copy and paste):")
        for name, value in metrics.items():
            if isinstance(value, Number):
                formatted = f"{value:,.2f}"
            else:
                formatted = str(value)
            print(f"- {name}: {formatted}")
        print()

    def _print_feature_summary(self, dataframe):
        """
        Print a compact overview of a few slide-friendly variables.
        """
        key_columns = [
            "SalePrice",
            "Overall Qual",
            "Gr Liv Area",
            "Garage Cars",
            "Year Built",
            "Lot Area",
        ]
        available = [col for col in key_columns if col in dataframe.columns]
        if not available:
            return

        summary = dataframe[available].agg(["mean", "median", "min", "max"]).T
        print("\nKey feature snapshot (mean | median | min | max):")
        for column in summary.index:
            stats = summary.loc[column]
            print(
                f"- {column}: "
                f"{stats['mean']:.0f} | "
                f"{stats['median']:.0f} | "
                f"{stats['min']:.0f} | "
                f"{stats['max']:.0f}"
            )
        print()

    def _plot_saleprice_vs_sqft(self, dataframe):
        """
        Export scatter plot of SalePrice vs. above-grade square footage.
        """
        required = {"SalePrice", "Gr Liv Area"}
        if not required.issubset(dataframe.columns):
            return

        fig_dir = Path("reports/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_path = fig_dir / "saleprice_vs_grlivarea.png"

        mask_large = dataframe["Gr Liv Area"] > 4000
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            dataframe.loc[~mask_large, "Gr Liv Area"],
            dataframe.loc[~mask_large, "SalePrice"],
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5,
            label="≤ 4,000 sq ft",
            color="#1f77b4",
        )
        if mask_large.any():
            ax.scatter(
                dataframe.loc[mask_large, "Gr Liv Area"],
                dataframe.loc[mask_large, "SalePrice"],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.5,
                label="> 4,000 sq ft",
                color="#d62728",
            )

        ax.set_title("Sale Price vs. Above-Grade Living Area")
        ax.set_xlabel("Above-Grade Living Area (sq ft)")
        ax.set_ylabel("Sale Price ($)")
        ax.ticklabel_format(style="plain", axis="both")
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved scatter plot to {plot_path}")

    def _plot_feature_importance(self, feature_names, top_n=15):
        """
        Save a bar chart with the top-N feature importances.
        """
        if not hasattr(self.model, "feature_importances_"):
            print("Model does not provide feature importances; skipping plot.")
            return

        importances = pd.Series(self.model.feature_importances_, index=feature_names)
        top_features = importances.sort_values(ascending=False).head(top_n)[::-1]

        fig_dir = Path("reports/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_path = fig_dir / "feature_importances_top.png"

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_features.index, top_features.values, color="#2ca02c")
        ax.set_title(f"Top {len(top_features)} Feature Importances")
        ax.set_xlabel("Mean Decrease in Squared Error")
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved feature importance plot to {plot_path}")


def main():
    import sys

    config_path = "configs/pipeline.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    pipeline = TrainingPipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
