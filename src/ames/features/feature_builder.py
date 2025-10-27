"""
Quick-and-dirty feature helpers for the Ames dataset.
"""

import pandas as pd


class FeatureBuilder:
    def __init__(self, config=None):
        self.config = config or {}
        geo_cfg = (
            (self.config.get("data_processing") or {}).get("geo_features") or {}
        )
        join_col = geo_cfg.get("features_join_column")
        self.id_columns = {join_col} if join_col else set()

    def make_basic_features(self, dataframe):
        """
        Create basic tabular features from the raw Ames data.
        Includes simple dummy encoding for categorical columns.
        """
        if dataframe is None:
            return []

        frame = dataframe.copy()
        frame = self._encode_categoricals(frame)
        return frame

    def add_geo_features(self, dataframe, geo_frame=None):
        """
        Merge in pre-computed spatial features when available.
        """
        if dataframe is None:
            return []
        return dataframe

    def _encode_categoricals(self, dataframe):
        """
        Applies one-hot encoding to categorical columns.
        """
        if dataframe is None:
            return []

        categorical_columns = self._categorical_columns(dataframe)
        if not categorical_columns:
            return dataframe

        encoded = pd.get_dummies(dataframe, columns=categorical_columns, drop_first=True)
        return encoded

    def _categorical_columns(self, dataframe):
        """
        Returns the list of categorical columns to encode.
        Uses config overrides when provided.
        """
        feature_cfg = (
            self.config.get("data_processing", {})
            .get("feature_builder", {})
            .get("categorical_columns")
        )
        protected = self.id_columns
        if feature_cfg:
            return [
                col for col in feature_cfg if col in dataframe.columns and col not in protected
            ]

        candidates = dataframe.select_dtypes(include=["object", "category"]).columns.tolist()
        return [col for col in candidates if col not in protected]
