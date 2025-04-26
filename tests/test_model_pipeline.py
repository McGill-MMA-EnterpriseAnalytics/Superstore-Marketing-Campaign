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

def test_lgbm_wrapper_fit_and_predict():
    # Create DataFrame with an object column
    df = pd.DataFrame({
        'cat': ['x', 'y'],
        'val': [10, 20]
    })
    y = np.array([1, 0])
    dummy = DummyModel()
    wrapper = mp.LGBMWrapper(dummy)

    # Fit
    wrapper.fit(df, y)
    # classes_ should be unique labels
    assert np.array_equal(wrapper.classes_, np.array([0, 1]))
    # Underlying dummy.fit called with encoded object column
    fit_df = dummy.fit_called_with
    assert sorted(fit_df['cat'].unique().tolist()) == [0, 1]

    # Predict
    preds = wrapper.predict(df)
    assert isinstance(preds, np.ndarray)
    assert preds.tolist() == [0, 0]

    # Predict_proba
    probas = wrapper.predict_proba(df)
    assert probas.shape == (2, 2)

def test_load_data(tmp_path, monkeypatch):
    # Create three small DataFrames and write as parquet
    df = pd.DataFrame({'a': [1, 2, 3], 'target': [0, 1, 0]})
    train = tmp_path / 'train.parquet'
    val = tmp_path / 'validation.parquet'
    test = tmp_path / 'test.parquet'
    df.to_parquet(train)
    df.to_parquet(val)
    df.to_parquet(test)

    # Mock get_data_paths()
    monkeypatch.setattr(mp, 'get_data_paths', lambda: {
        'processed': {
            'train': str(train),
            'validation': str(val),
            'test': str(test)
        }
    })

    X_train, y_train, X_val, y_val, X_test, y_test = mp.load_data()

    # Verify shapes and types
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert y_train.tolist() == [0, 1, 0]

def test_get_model_variants(monkeypatch):
    # Mock get_model_config to return a simple param dict
    monkeypatch.setattr(mp, 'get_model_config', lambda name, tuned: {'n_estimators': 5})

    # xgboost
    model_xgb = mp.get_model('xgboost')
    assert isinstance(model_xgb, XGBClassifier)

    # lightgbm
    model_lgb = mp.get_model('lightgbm')
    assert isinstance(model_lgb, LGBMClassifier)

    # randomforest
    model_rf = mp.get_model('randomforest')
    assert isinstance(model_rf, RandomForestClassifier)

    # catboost returns tuple (model, cat_features)
    result = mp.get_model('catboost')
    assert isinstance(result, tuple)
    model_cb, cat_features = result
    assert isinstance(model_cb, CatBoostClassifier)

    # unsupported model raises ValueError
    with pytest.raises(ValueError):
        mp.get_model('unknown_model')

def test_save_and_load_model(tmp_path, monkeypatch):
    # Monkeypatch get_paths() to return tmp 'models' directory
    monkeypatch.setattr(mp, 'get_paths', lambda: {'models': str(tmp_path)})

    dummy = DummyModel()
    filepath = mp.save_model(dummy, 'testmodel', tuned=False)

    # File should exist
    assert os.path.exists(filepath)

    # Loading should return the same DummyModel
    loaded = joblib.load(filepath)
    assert isinstance(loaded, DummyModel)


def test_plot_feature_importance_and_confusion(tmp_path, monkeypatch):
    # Monkeypatch get_paths() to return tmp 'figures' directory
    monkeypatch.setattr(mp, 'get_paths', lambda: {'figures': str(tmp_path)})

    # Feature importance: use a dummy model with feature_importances_
    class FIModel:
        feature_importances_ = np.array([0.1, 0.2, 0.3])

    model = FIModel()
    feature_names = ['f1', 'f2', 'f3']
    path1 = mp.plot_feature_importance(model, feature_names, 'mod')
    assert os.path.exists(path1)

    # Confusion matrix: use a real DecisionTreeClassifier on simple data
    X = pd.DataFrame({'x': [0, 1, 0, 1]})
    y = np.array([0, 1, 0, 1])
    dt = DecisionTreeClassifier().fit(X, y)
    path2 = mp.plot_confusion_matrix(dt, X, y, 'dtmod')
    assert os.path.exists(path2)


def test_evaluate_model_on_test():
    # Use a simple DecisionTreeClassifier
    X = pd.DataFrame({'x': [0, 1, 1, 0]})
    y = np.array([0, 1, 1, 0])
    dt = DecisionTreeClassifier().fit(X, y)
    metrics = mp.evaluate_model_on_test(dt, X, y, 'dtmod')
    # Check that all expected keys are present
    for key in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']:
        assert key in metrics

def test_save_results(tmp_path):
    # Monkeypatch get_paths() to return tmp 'results' directory
    import importlib
    importlib.reload(mp)
    mp.get_paths = lambda: {'results': str(tmp_path)}

    val_metrics = {'accuracy': 0.8}
    test_metrics = {'test_accuracy': 0.75}

    # First save
    file1 = mp.save_results('mod', val_metrics, test_metrics, tuned=False)
    assert os.path.exists(file1)
    df1 = pd.read_csv(file1)
    assert 'val_accuracy' in df1.columns and 'test_accuracy' in df1.columns

    # Append second time
    file2 = mp.save_results('mod', val_metrics, test_metrics, tuned=False)
    df2 = pd.read_csv(file2)
    assert len(df2) == 2