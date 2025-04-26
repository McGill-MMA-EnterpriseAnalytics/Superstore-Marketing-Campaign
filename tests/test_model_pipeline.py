import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')   # Use non-interactive backend for plotting
import pytest
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Replace 'model_pipeline' with the actual filename (without .py) of your module
import src.train_model as mp

class DummyModel:
    """A dummy classifier to capture fit/predict calls."""
    def __init__(self):
        self.fit_called_with = None
        self.predict_called_with = None
        self.proba_called_with = None

    def fit(self, X, y, **kwargs):
        self.fit_called_with = X.copy()

    def predict(self, X):
        self.predict_called_with = X.copy()
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        self.proba_called_with = X.copy()
        # Return two-class probabilities: all zeros for class 0, ones for class 1
        return np.vstack([np.zeros(len(X)), np.ones(len(X))]).T
    

def test_xgboost_wrapper_fit_and_predict():
    # Create DataFrame with object and special-character column names
    df = pd.DataFrame({
        'cat': ['a', 'b'],
        'num': [1, 2],
        'weird<col>': [3, 4]
    })
    y = np.array([0, 1])
    dummy = DummyModel()
    wrapper = mp.XGBoostWrapper(dummy)

    # Fit wrapper
    wrapper.fit(df, y)
    # classes_ should be unique labels
    assert np.array_equal(wrapper.classes_, np.array([0, 1]))
    # The underlying dummy.fit should have been called with cleaned DataFrame
    fit_df = dummy.fit_called_with
    # 'cat' encoded to integers
    assert fit_df['cat'].tolist() == [0, 1]
    # special characters replaced in column names
    assert 'weird_col_' in fit_df.columns

    # Predict
    preds = wrapper.predict(df)
    assert isinstance(preds, np.ndarray)
    assert preds.tolist() == [0, 0]

    # Predict_proba
    probas = wrapper.predict_proba(df)
    assert probas.shape == (2, 2)
    assert np.all(probas[:, 1] == 1)