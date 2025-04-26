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