"""
Helpers for building regression models from config.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


MODEL_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "elastic_net": ElasticNet,}


class ModelFactory:
    """
    Builds models based on the `model` section of the config file.
    """

    def __init__(self, model_config=None):
        self.model_config = model_config or {}

    def create(self):
        """
        Returns an unfitted sklearn regressor.
        """
        model_name = self.model_config.get("name", "random_forest")
        params = self.model_config.get("params", {})
        model_cls = MODEL_REGISTRY.get(model_name)
        if model_cls is None:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model `{model_name}`. Pick from: {available}.")
        return model_cls(**params)


class HyperparameterTuner:
    """
    Wraps simple grid or random search using sklearn utilities.
    """

    def __init__(self, tuning_config=None):
        self.tuning_config = tuning_config or {}

    def tune(self, estimator, features, targets):
        """
        Runs the configured search if both data and tuning settings exist.
        Returns (best_estimator, tuning_summary_dict).
        """
        if (
            not self.tuning_config
            or not self.tuning_config.get("param_grid")
            or features is None
            or targets is None
        ):
            return estimator, {}

        strategy = self.tuning_config.get("strategy", "grid")
        param_grid = self.tuning_config.get("param_grid", {})
        cv = self.tuning_config.get("cv", 3)
        scoring = self.tuning_config.get("scoring")
        n_jobs = self.tuning_config.get("n_jobs", -1)
        verbose = self.tuning_config.get("verbose", 0)

        if strategy == "random":
            n_iter = self.tuning_config.get("n_iter", 10)
            random_state = self.tuning_config.get("random_state", 42)
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
            )
        else:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose,
            )

        search.fit(features, targets)
        summary = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
        }
        return search.best_estimator_, summary
