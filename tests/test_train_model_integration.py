# tests/test_train_model_integration.py ..

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

# Import the module under test (adjust the name if different)
import src.train_model as tm

def test_train_evaluate_save_pipeline(tmp_path, monkeypatch):
    """
    Full integration flow across:
      - load_data()
      - train_model()
      - evaluate_model_on_test()
      - save_results()
    """

    # --- 1) Prepare dummy preprocessed parquet files ---
    df = pd.DataFrame({
        'feature': [0, 0, 1, 1],
        'target':  [0, 0, 1, 1]
    })
    # Write train/validation/test to temporary directory
    train_path = tmp_path / "train.parquet"
    val_path   = tmp_path / "validation.parquet"
    test_path  = tmp_path / "test.parquet"
    df.to_parquet(train_path)
    df.to_parquet(val_path)
    df.to_parquet(test_path)

    # --- 2) Monkey-patch external I/O and logging ---
    # Patch get_data_paths() to point to our dummy files
    monkeypatch.setattr(tm, 'get_data_paths', lambda: {
        "processed": {
            "train": str(train_path),
            "validation": str(val_path),
            "test": str(test_path),
        }
    })
    # Patch log_model_metrics so no MLflow calls occur
    monkeypatch.setattr(tm, 'log_model_metrics', lambda *args, **kwargs: None)
    # Patch get_paths() so save_results writes into tmp_path/results/
    monkeypatch.setattr(tm, 'get_paths', lambda: {"results": str(tmp_path / "results")})

    # --- 3) Load data via load_data() ---
    X_train, y_train, X_val, y_val, X_test, y_test = tm.load_data()
    assert isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.Series)
    assert len(X_train) == 4 and len(y_train) == 4

    # --- 4) Train a simple classifier via train_model() ---
    model = DecisionTreeClassifier(random_state=0)
    fitted_model, val_metrics, feature_importance = tm.train_model(
        model,
        X_train, y_train,
        X_val,   y_val,
        cat_features=None,
        run_id=None,
        model_name='dt_integration',
        tuned=False
    )
    # All validation metrics should be perfect on this toy data
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert pytest.approx(1.0, rel=1e-6) == val_metrics[metric]

    # Feature importance must match the trained model’s attribute
    np.testing.assert_allclose(feature_importance, fitted_model.feature_importances_)

    # --- 5) Evaluate on test set via evaluate_model_on_test() ---
    test_metrics = tm.evaluate_model_on_test(
        fitted_model,
        X_test, y_test,
        model_name="dt_integration_test"
    )
    # All test metrics should also be 1.0
    for value in test_metrics.values():
        assert pytest.approx(1.0, rel=1e-6) == value

    # --- 6) Save results via save_results() and validate CSV ---
    result_file = tm.save_results(
        model_name="dt_integration",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        tuned=False
    )
    assert os.path.exists(result_file)

    results_df = pd.read_csv(result_file)
    # Must contain both validation and test metric columns
    assert "val_accuracy" in results_df.columns
    assert "test_accuracy" in results_df.columns
    # First call writes one row
    assert len(results_df) == 1

    # A second save should append a row
    result_file_2 = tm.save_results(
        model_name="dt_integration",
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        tuned=False
    )
    df2 = pd.read_csv(result_file_2)
    assert len(df2) == 2
