"""
Training pipeline that ties together data ingestion, validation, feature engineering, and model training."""

from ..data.ingestion import DataIngestor
from ..data.validation import DataValidator
from ..features.feature_builder import FeatureBuilder
from ..features.geo_features import GeoFeatureEngineer
from ..utils.config import load_config
from .evaluation import ModelEvaluator
from ..models.model_factory import ModelFactory


class TrainingPipeline:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.ingestor = DataIngestor(self.config)
        self.validator = DataValidator(self.config)  # idation step
        self.feature_builder = FeatureBuilder(self.config)
        self.geo_feature_engineer = GeoFeatureEngineer(self.config)
        self.evaluator = ModelEvaluator(self.config.get("evaluation"))
        self.model = ModelFactory(self.config.get("model")).create()

    def run(self):
        """
        Runs the end-to-end training pipeline.
        """
        raw_data = self.ingestor.download_source_data()
        staged_data = self.ingestor.save_locally(raw_data)
        validated_data = self.validator.basic_checks(staged_data)
        validated_data = self.validator.process_missing_values(validated_data)
        validated_data = self.validator.process_outliers(validated_data)
        basic_features = self.feature_builder.make_basic_features(validated_data)

        geo_features = self.geo_feature_engineer.compute_ndvi()
        combined = self.geo_feature_engineer.join_to_features(
            basic_features, geo_features
        )
        
        self.model.fit(combined.drop(columns=["SalePrice"]), combined["SalePrice"])
        predictions = self.model.predict(combined.drop(columns=["SalePrice"]))
        metrics = self.evaluator.evaluate(predictions, targets=combined["SalePrice"])
        print(metrics)
        return metrics


def main():
    import sys

    config_path = "configs/pipeline.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    pipeline = TrainingPipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
