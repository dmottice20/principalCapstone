from xgboost import XGBRegressor, XGBClassifier
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from typing import Union, Dict, Any


class XGBoostModel(BaseEstimator):
    def __init__(self, model_type: str = 'regressor', **kwargs):
        self.model_type = model_type
        self.model_params = kwargs
        self.model = None

    def _initialize_model(self):
        if self.model_type == 'regressor':
            self.model = XGBRegressor(**self.model_params)
        elif self.model_type == 'classifier':
            self.model = XGBClassifier(**self.model_params)
        else:
            raise ValueError("model_type must be either 'regressor' or 'classifier'")

    def fit(self, X, y):
        if self.model is None:
            self._initialize_model()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() before predict().")
        return self.model.predict(X)

    def set_params(self, **params):
        self.model_params.update(params)
        if self.model is not None:
            self.model.set_params(**params)
        return self

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if self.model is not None:
            params.update(self.model.get_params())
        return params


class XGBoostRegressor(XGBoostModel, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(model_type='regressor', **kwargs)


class XGBoostClassifier(XGBoostModel, ClassifierMixin):
    def __init__(self, **kwargs):
        super().__init__(model_type='classifier', **kwargs)
