"""
Evaluation utilities for trained models.
"""


class ModelEvaluator:
    def __init__(self, config=None):
        self.config = config or {}

    def evaluate(self, predictions, targets):
        """
        Compute metrics like RMSE once targets are available.
        """
        from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error, mean_absolute_percentage_error
        import numpy as np

        metrics = {}
        if targets is not None:
            rmse = root_mean_squared_error(targets, predictions)
            mape = mean_absolute_percentage_error(targets, predictions)
            metrics
            rmsle = root_mean_squared_log_error(targets, predictions)

            metrics["RMSE"] = rmse
            metrics["MAPE"] = mape
            metrics["RMSLE"] = rmsle
        return metrics
