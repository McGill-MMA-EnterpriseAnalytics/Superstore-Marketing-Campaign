# import pandas as pd
# import numpy as np
# from src.preprocess import handle_missing_values

# def test_handle_missing_values_drops_and_imputes():
#     df = pd.DataFrame({
#         "A": [np.nan, np.nan, np.nan, 2],  # 3/4 = 75% missing (above 70% threshold)
#         "B": [1, np.nan, 3, 4],            # 1/4 = 25% missing (below threshold)
#         "C": [10, 20, 30, 40],             # 0% missing
#     })
#     cleaned = handle_missing_values(df)
#     # 'A' > 70% NA → dropped
#     assert "A" not in cleaned.columns
#     # 'B' < 70% NA → imputed
#     assert "B" in cleaned.columns
#     assert cleaned["B"].isna().sum() == 0  # No more NaN values

import os
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import src.preprocess as preprocess

def test_clean_column_names():
    df = pd.DataFrame({' Column One ': [1], 'SecondCol': [2]})
    cleaned = preprocess.clean_column_names(df.copy())
    assert list(cleaned.columns) == ['column_one', 'secondcol']

def test_handle_missing_values_drop_and_impute(monkeypatch):
    config = {"missing_values": {"drop_threshold": 0.5, "imputation_method": "median"}}
    monkeypatch.setattr(preprocess, 'get_preprocessing_config', lambda: config)
    df = pd.DataFrame({'A': [1, None, 3], 'B': [None, None, 1], 'C': [10, 20, 30]})
    result = preprocess.handle_missing_values(df.copy())
    assert 'B' not in result.columns
    assert result['A'].iloc[1] == 2.0

def test_apply_recategorization(monkeypatch):
    config = {
        "feature_engineering": {
            "recategorization": {
                "col": {"a": "A", "b": "B"}
            }
        }
    }
    monkeypatch.setattr(preprocess, 'get_preprocessing_config', lambda: config)
    df = pd.DataFrame({'col': ['a', 'b', 'c']})
    result = preprocess.apply_recategorization(df.copy())
    # a→A, b→B, c stays 'c'
    assert result['col'].tolist() == ['A', 'B', 'c']

def test_load_data(tmp_path, monkeypatch):
    df = pd.DataFrame({'x': [1, 2, 3]})
    file = tmp_path / 'data.csv'
    df.to_csv(file, index=False)
    monkeypatch.setattr(preprocess, 'get_data_paths', lambda: {"raw": str(file)})
    loaded = preprocess.load_data()
    # Should load exactly what we wrote
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df)

def test_identify_and_transform_skewed_features(monkeypatch):
    # need >2 unique values so the column is considered
    config = {"feature_engineering": {"skewness_threshold": 0.5}}
    monkeypatch.setattr(preprocess, 'get_preprocessing_config', lambda: config)
    df = pd.DataFrame({
        'flat': [1, 1, 1, 1, 1, 1],
        'skewed': [0, 0, 0, 0, 50, 100]
    })
    result = preprocess.identify_and_transform_skewed_features(df.copy())
    # 'flat' unchanged
    assert all(result['flat'] == 1)
    # 'skewed' last value should be log1p(100)
    assert np.isclose(result['skewed'].iloc[-1], np.log1p(100))

def test_implement_clustering(monkeypatch):
    config = {"feature_engineering": {"clustering": {"n_clusters": 2, "random_state": 0}}}
    monkeypatch.setattr(preprocess, 'get_preprocessing_config', lambda: config)
    df = pd.DataFrame({
        'income': [1,1,100,100],
        'kidhome': [0,0,0,0],
        'teenhome': [0,0,0,0],
        'recency': [1,1,100,100],
        'mntwines': [0,0,0,0],
        'mntfruits': [0,0,0,0],
        'mntmeatproducts': [0,0,0,0],
        'mntfishproducts': [0,0,0,0],
        'mntsweetproducts': [0,0,0,0],
        'mntgoldprods': [0,0,0,0],
        'numwebpurchases': [0,0,0,0],
        'numcatalogpurchases': [0,0,0,0],
        'numstorepurchases': [0,0,0,0],
        'numwebvisitsmonth': [0,0,0,0]
    })
    result = preprocess.implement_clustering(df.copy())
    assert 'customer_segment' in result.columns
    assert set(result['customer_segment']) == {0, 1}

def test_create_engineered_features():
    current_year = dt.datetime.now().year
    today = dt.datetime.now().date()
    df = pd.DataFrame([{
        'mntwines': 1,
        'mntfruits': 1,
        'mntmeatproducts': 1,
        'mntfishproducts': 1,
        'mntsweetproducts': 1,
        'mntgoldprods': 1,
        'year_birth': current_year - 30,
        'kidhome': 1,
        'teenhome': 0,
        'marital_status': 'Married',
        'income': 100,
        'recency': 10,
        'numwebpurchases': 2,
        'numcatalogpurchases': 0,
        'numstorepurchases': 1,
        'numwebvisitsmonth': 5,
        'dt_customer': today.isoformat()
    }])
    result = preprocess.create_engineered_features(df.copy())
    assert result['total_spend'].iloc[0] == 6
    assert result['age'].iloc[0] == 30
    assert result['has_children'].iloc[0] == 1
    # after one-hot, check the dummy column for 'web'
    assert result['preferred_channel_web'].iloc[0] == 1
    assert result['channel_diversity'].iloc[0] == 2
    assert np.isclose(result['recency_score'].iloc[0], np.exp(-10/30))
    assert result['is_active'].iloc[0] == 1