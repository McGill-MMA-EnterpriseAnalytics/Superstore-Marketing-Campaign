import warnings
import datetime as dt
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import src.preprocess as preprocess
from sklearn.exceptions import ConvergenceWarning

# Ignore any leftover KMeans ConvergenceWarnings in this file
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def test_preprocess_data_integration(tmp_path, monkeypatch):
    # 1) Create dummy raw CSV with 4 rows—but two different “segments”:
    today = dt.datetime.now().date().isoformat()
    base = {
        'Id': 0,
        'mntwines': 1, 'mntfruits': 2, 'mntmeatproducts': 3,
        'mntfishproducts': 4, 'mntsweetproducts': 5, 'mntgoldprods': 6,
        'year_birth': 1990, 'kidhome': 0, 'teenhome': 1,
        'marital_status': 'Married', 'income': None,
        'recency': None,
        'numwebpurchases': 1, 'numcatalogpurchases': 0, 'numstorepurchases': 1,
        'numwebvisitsmonth': 2,
        'dt_customer': today,
        'response': 1
    }

    # Build two low‐value rows and two high‐value rows
    rows = []
    for income, recency in [(10, 5), (10, 5), (1000, 50), (1000, 50)]:
        r = base.copy()
        r['income'] = income
        r['recency'] = recency
        rows.append(r)

    df = pd.DataFrame(rows)
    raw_path = tmp_path / "raw.csv"
    df.to_csv(raw_path, index=False)

    # 2) Mock config & paths
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    monkeypatch.setattr(preprocess, 'get_data_paths', lambda: {
        "raw": str(raw_path),
        "processed": {
            "train": str(processed_dir / "train.parquet"),
            "validation": str(processed_dir / "validation.parquet"),
            "test": str(processed_dir / "test.parquet"),
        }
    })
    monkeypatch.setattr(preprocess, 'get_preprocessing_config', lambda: {
        "missing_values": {"drop_threshold": 1.0, "imputation_method": "median"},
        "feature_engineering": {
            "recategorization": {},
            "clustering": {"n_clusters": 2, "random_state": 0},
            "skewness_threshold": 10.0
        }
    })
    monkeypatch.setattr(preprocess, 'create_directories', lambda: None)

    # 3) Run the full preprocess
    train_df, val_df, test_df = preprocess.preprocess_data()

    # 4) Check splits: 4 rows → train 2, val 1, test 1
    assert len(train_df) == 2
    assert len(val_df) == 1
    assert len(test_df) == 1

    # 5) Check that parquet files were written
    assert (processed_dir / "train.parquet").exists()
    assert (processed_dir / "validation.parquet").exists()
    assert (processed_dir / "test.parquet").exists()

    # 6) All targets should be 1 (our dummy response)
    for df_out in (train_df, val_df, test_df):
        assert 'target' in df_out.columns
        assert (df_out['target'] == 1).all()

    # 7) Clustering sanity: we should see exactly two segments
    segments = set(train_df['customer_segment']).union(
        set(val_df['customer_segment']),
        set(test_df['customer_segment'])
    )
    assert len(segments) == 2
