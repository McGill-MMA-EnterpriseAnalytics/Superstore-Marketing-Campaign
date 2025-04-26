import logging
import glob
import joblib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score
from src.utils.config import load_config

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, cfg_path="config.yaml"):
        cfg = load_config(cfg_path)
        mon = cfg["monitoring"]

        # load reference & production paths
        self.ref = pd.read_parquet(mon["reference_data_path"])
        self.prod_paths = glob.glob(mon["production_data_glob"])

        # thresholds dict
        self.thresh = mon["thresholds"]

        # feature list (exclude target)
        self.features = [c for c in self.ref.columns if c != "target"]
        # split numeric vs categorical
        self.numeric_features = [
            c for c in self.features
            if pd.api.types.is_numeric_dtype(self.ref[c])
        ]
        self.cat_features = [
            c for c in self.features
            if not pd.api.types.is_numeric_dtype(self.ref[c])
        ]

        # load model and grab its exact feature ordering
        self.model = joblib.load(mon["model_path"])
        self.feature_names = self.model.get_booster().feature_names
        logger.info(f"Loaded model; expecting features: {self.feature_names}")

    def _predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode cats, reindex to model.feature_names (fill missing=0), then predict.
        """
        X = df.copy()
        # encode categorical columns as integer codes
        for f in self.cat_features:
            X[f] = X[f].astype("category").cat.codes
        # reindex to exactly the training features
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict(X_aligned)

    def _check_feature_drift(self, df: pd.DataFrame) -> float:
        ws = []
        # numeric features: Wasserstein on raw values
        for f in self.numeric_features:
            ws.append(wasserstein_distance(self.ref[f], df[f]))
        # categorical features: Wasserstein on category‐codes
        for f in self.cat_features:
            ref_codes = self.ref[f].astype("category").cat.codes
            cur_codes = df[f].astype("category").cat.codes
            ws.append(wasserstein_distance(ref_codes, cur_codes))
        avg_ws = float(np.mean(ws))
        logger.info(f"Avg feature drift (W1): {avg_ws:.4f}")
        return avg_ws

    def _check_target_drift(self, df: pd.DataFrame) -> float:
        p_ref = self.ref["target"].value_counts(normalize=True)
        p_cur = df["target"].value_counts(normalize=True)
        drift = float((p_ref - p_cur).abs().sum() / 2)
        logger.info(f"Target drift (L1): {drift:.4f}")
        return drift

    def _check_prediction_drift(self, df: pd.DataFrame) -> float:
        pred_ref = self._predict(self.ref[self.features])
        pred_cur = self._predict(df[self.features])
        p_ref = pd.Series(pred_ref).value_counts(normalize=True)
        p_cur = pd.Series(pred_cur).value_counts(normalize=True)
        drift = float((p_ref - p_cur).abs().sum() / 2)
        logger.info(f"Prediction drift (L1): {drift:.4f}")
        return drift

    def _check_concept_drift(self, df: pd.DataFrame) -> float:
        # compute F1 on reference vs current
        y_ref = self.ref["target"]
        y_cur = df["target"]
        pred_ref = self._predict(self.ref[self.features])
        pred_cur = self._predict(df[self.features])

        f1_ref = f1_score(y_ref, pred_ref)
        f1_cur = f1_score(y_cur, pred_cur)
        drop = float(f1_ref - f1_cur)
        logger.info(f"Concept drift (F1 drop): {drop:.4f}")
        return drop

    def run_all_checks(self):
        for path in self.prod_paths:
            df = pd.read_parquet(path)
            logger.info(f"\n--- Running drift checks on {path} ---")

            alerts = []
            if self._check_feature_drift(df) > self.thresh["feature_drift"]:
                alerts.append("Feature drift exceeded")
            if self._check_target_drift(df) > self.thresh["target_drift"]:
                alerts.append("Target drift exceeded")
            if self._check_prediction_drift(df) > self.thresh["prediction_drift"]:
                alerts.append("Prediction drift exceeded")
            if self._check_concept_drift(df) > self.thresh["performance_drop"]:
                alerts.append("Concept drift exceeded")

            if alerts:
                for a in alerts:
                    logger.warning(a)
            else:
                logger.info("No drift detected.")

if __name__ == "__main__":
    dd = DriftDetector("config.yaml")
    dd.run_all_checks()
