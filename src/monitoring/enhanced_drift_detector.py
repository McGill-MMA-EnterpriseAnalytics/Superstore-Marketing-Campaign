import datetime
import glob
import json
import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from src.utils.config import load_config
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

class EnhancedDriftDetector:
    def __init__(self, cfg_path="config.yaml"):
        """
        Initialize the enhanced drift detector.
        
        Parameters:
            cfg_path (str): Path to configuration file
        """
        self.cfg_path = cfg_path
        cfg = load_config(cfg_path)
        mon = cfg["monitoring"]
        paths = cfg.get("paths", {})
        
        # Set up paths
        self.results_path = paths.get("results", "results/")
        self.figures_path = paths.get("figures", "figures/")
        self.history_path = os.path.join(self.results_path, "drift_history.csv")
        
        # Create directories if they don't exist
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
        # Initialize history DataFrame if not exists
        if not os.path.exists(self.history_path):
            self._initialize_history_file()
        
        # Load reference & production paths
        self.ref_path = mon["reference_data_path"]
        self.ref = pd.read_parquet(self.ref_path)
        self.prod_paths = glob.glob(mon["production_data_glob"])
        
        # Load thresholds with defaults
        self.thresh = mon.get("thresholds", {
            "feature_drift": 0.1,
            "target_drift": 0.05,
            "prediction_drift": 0.05,
            "performance_drop": 0.05
        })
        
        # Feature lists (exclude target)
        self.features = [c for c in self.ref.columns if c != "target"]
        # Split numeric vs categorical
        self.numeric_features = [
            c for c in self.features
            if pd.api.types.is_numeric_dtype(self.ref[c])
        ]
        self.cat_features = [
            c for c in self.features
            if not pd.api.types.is_numeric_dtype(self.ref[c])
        ]
        
        # Load model
        model_path = mon["model_path"]
        try:
            self.model = joblib.load(model_path)
            if hasattr(self.model, 'get_booster'):
                self.feature_names = self.model.get_booster().feature_names
            else:
                self.feature_names = self.features
            logger.info(f"Loaded model from {model_path}")
            
            # Extract model metadata
            self.model_name = Path(model_path).stem
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _initialize_history_file(self):
        """Initialize empty history CSV file with headers."""
        history_df = pd.DataFrame(columns=[
            'timestamp', 'dataset', 'feature_drift', 'target_drift', 
            'prediction_drift', 'concept_drift', 'accuracy', 'precision', 
            'recall', 'f1_score', 'roc_auc', 'alerts'
        ])
        history_df.to_csv(self.history_path, index=False)
        logger.info(f"Initialized drift history file at {self.history_path}")
    
    def _predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode categorical features, align columns to model's expected features, and predict.
        
        Parameters:
            df (DataFrame): Input data
            
        Returns:
            ndarray: Predictions
        """
        X = df.copy()
        # Encode categorical columns as integer codes
        for f in self.cat_features:
            X[f] = X[f].astype("category").cat.codes
        
        # Reindex to exactly the training features
        if hasattr(self.model, 'get_booster'):
            X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
            return self.model.predict(X_aligned)
        else:
            # For models without get_booster method
            return self.model.predict(X[self.features])
    
    def _predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions if the model supports it.
        
        Parameters:
            df (DataFrame): Input data
            
        Returns:
            ndarray: Probability predictions or None if not supported
        """
        X = df.copy()
        # Encode categorical columns as integer codes
        for f in self.cat_features:
            X[f] = X[f].astype("category").cat.codes
        
        # Reindex to exactly the training features
        if hasattr(self.model, 'get_booster'):
            X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        else:
            X_aligned = X[self.features]
        
        # Check if predict_proba is available
        if hasattr(self.model, 'predict_proba'):
            try:
                return self.model.predict_proba(X_aligned)
            except:
                logger.warning("predict_proba failed, falling back to predict")
                return None
        return None
    
    def _check_feature_drift(self, df: pd.DataFrame) -> dict:
        """
        Check for feature drift using Wasserstein distance and KS test.
        
        Parameters:
            df (DataFrame): Current data
            
        Returns:
            dict: Dictionary with overall and per-feature drift metrics
        """
        ws_values = {}
        ks_values = {}
        
        # Numeric features: Wasserstein on raw values + KS test
        for f in self.numeric_features:
            ws_values[f] = float(wasserstein_distance(self.ref[f], df[f]))
            ks_stat, ks_pval = ks_2samp(self.ref[f], df[f])
            ks_values[f] = float(ks_pval)
        
        # Categorical features: Wasserstein on category-codes
        for f in self.cat_features:
            ref_codes = self.ref[f].astype("category").cat.codes
            cur_codes = df[f].astype("category").cat.codes
            ws_values[f] = float(wasserstein_distance(ref_codes, cur_codes))
            
            # For categorical features, compare value distributions
            p_ref = self.ref[f].value_counts(normalize=True)
            p_cur = df[f].value_counts(normalize=True)
            
            # Align indices
            all_cats = set(p_ref.index) | set(p_cur.index)
            p_ref = p_ref.reindex(all_cats, fill_value=0)
            p_cur = p_cur.reindex(all_cats, fill_value=0)
            
            # L1 distance between distributions
            ks_values[f] = float((p_ref - p_cur).abs().sum() / 2)
        
        # Calculate average drift
        avg_ws = float(np.mean(list(ws_values.values())))
        
        # Find top drifting features
        top_drifting = sorted(ws_values.items(), key=lambda x: x[1], reverse=True)[:5]
        
        result = {
            'avg_wasserstein': avg_ws,
            'per_feature_wasserstein': ws_values,
            'per_feature_ks_pvalue': ks_values,
            'top_drifting_features': dict(top_drifting)
        }
        
        logger.info(f"Avg feature drift (W1): {avg_ws:.4f}")
        logger.info(f"Top 5 drifting features: {dict(top_drifting)}")
        
        return result
    
    def _check_target_drift(self, df: pd.DataFrame) -> dict:
        """
        Check for target distribution drift.
        
        Parameters:
            df (DataFrame): Current data
            
        Returns:
            dict: Target drift metrics
        """
        p_ref = self.ref["target"].value_counts(normalize=True)
        p_cur = df["target"].value_counts(normalize=True)
        
        # Align indices to ensure comparison works
        all_classes = set(p_ref.index) | set(p_cur.index)
        p_ref = p_ref.reindex(all_classes, fill_value=0)
        p_cur = p_cur.reindex(all_classes, fill_value=0)
        
        # L1 distance (Total Variation Distance)
        l1_drift = float((p_ref - p_cur).abs().sum() / 2)
        
        # Target proportions
        ref_pos_rate = float(self.ref["target"].mean())
        cur_pos_rate = float(df["target"].mean())
        
        result = {
            'l1_distance': l1_drift,
            'reference_positive_rate': ref_pos_rate,
            'current_positive_rate': cur_pos_rate,
            'absolute_change': abs(ref_pos_rate - cur_pos_rate)
        }
        
        logger.info(f"Target drift (L1): {l1_drift:.4f}")
        logger.info(f"Positive rate: reference={ref_pos_rate:.4f}, current={cur_pos_rate:.4f}")
        
        return result
    
    def _check_prediction_drift(self, df: pd.DataFrame) -> dict:
        """
        Check for drift in model predictions.
        
        Parameters:
            df (DataFrame): Current data
            
        Returns:
            dict: Prediction drift metrics
        """
        # Get predictions
        pred_ref = self._predict(self.ref)
        pred_cur = self._predict(df)
        
        # Calculate prediction distributions
        p_ref = pd.Series(pred_ref).value_counts(normalize=True)
        p_cur = pd.Series(pred_cur).value_counts(normalize=True)
        
        # Align indices
        all_classes = set(p_ref.index) | set(p_cur.index)
        p_ref = p_ref.reindex(all_classes, fill_value=0)
        p_cur = p_cur.reindex(all_classes, fill_value=0)
        
        # L1 distance between prediction distributions
        l1_drift = float((p_ref - p_cur).abs().sum() / 2)
        
        # Prediction proportions
        ref_pos_rate = float(pd.Series(pred_ref).mean())
        cur_pos_rate = float(pd.Series(pred_cur).mean())
        
        # Try to get probability distributions if available
        prob_drift = None
        prob_ref = self._predict_proba(self.ref)
        prob_cur = self._predict_proba(df)
        
        if prob_ref is not None and prob_cur is not None:
            # For binary classification, use positive class probability
            if prob_ref.shape[1] == 2:
                ref_probs = prob_ref[:, 1]
                cur_probs = prob_cur[:, 1]
                # Wasserstein on probability distributions
                prob_drift = float(wasserstein_distance(ref_probs, cur_probs))
        
        result = {
            'l1_distance': l1_drift,
            'reference_positive_rate': ref_pos_rate,
            'current_positive_rate': cur_pos_rate,
            'absolute_change': abs(ref_pos_rate - cur_pos_rate)
        }
        
        if prob_drift is not None:
            result['probability_wasserstein'] = prob_drift
        
        logger.info(f"Prediction drift (L1): {l1_drift:.4f}")
        logger.info(f"Prediction positive rate: reference={ref_pos_rate:.4f}, current={cur_pos_rate:.4f}")
        
        return result
    
    def _check_concept_drift(self, df: pd.DataFrame) -> dict:
        """
        Check for concept drift by comparing model performance.
        
        Parameters:
            df (DataFrame): Current data
            
        Returns:
            dict: Performance metrics and drift indicators
        """
        # True labels
        y_ref = self.ref["target"]
        y_cur = df["target"]
        
        # Predictions
        pred_ref = self._predict(self.ref)
        pred_cur = self._predict(df)
        
        # Calculate metrics for reference data
        metrics_ref = {
            'accuracy': float(accuracy_score(y_ref, pred_ref)),
            'precision': float(precision_score(y_ref, pred_ref)),
            'recall': float(recall_score(y_ref, pred_ref)),
            'f1': float(f1_score(y_ref, pred_ref))
        }
        
        # Calculate metrics for current data
        metrics_cur = {
            'accuracy': float(accuracy_score(y_cur, pred_cur)),
            'precision': float(precision_score(y_cur, pred_cur)),
            'recall': float(recall_score(y_cur, pred_cur)),
            'f1': float(f1_score(y_cur, pred_cur))
        }
        
        # Try to get ROC-AUC if predict_proba is available
        prob_ref = self._predict_proba(self.ref)
        prob_cur = self._predict_proba(df)
        
        if prob_ref is not None and prob_cur is not None:
            # For binary classification
            if prob_ref.shape[1] == 2:
                metrics_ref['roc_auc'] = float(roc_auc_score(y_ref, prob_ref[:, 1]))
                metrics_cur['roc_auc'] = float(roc_auc_score(y_cur, prob_cur[:, 1]))
        
        # Calculate performance drops
        perf_drops = {}
        for metric in metrics_ref:
            perf_drops[metric] = float(metrics_ref[metric] - metrics_cur[metric])
        
        # F1 score drop (primary concept drift metric)
        f1_drop = perf_drops.get('f1', 0.0)
        
        result = {
            'reference_metrics': metrics_ref,
            'current_metrics': metrics_cur,
            'performance_drops': perf_drops,
            'f1_drop': f1_drop
        }
        
        logger.info(f"Concept drift (F1 drop): {f1_drop:.4f}")
        logger.info(f"Reference metrics: {metrics_ref}")
        logger.info(f"Current metrics: {metrics_cur}")
        
        return result
    
    def _save_drift_results(self, dataset_name, results):
        """
        Save drift results to history CSV.
        
        Parameters:
            dataset_name (str): Name of the dataset
            results (dict): Dictionary with drift results
        """
        # Create a row for the history DataFrame
        row = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset': dataset_name,
            'feature_drift': results['feature_drift']['avg_wasserstein'],
            'target_drift': results['target_drift']['l1_distance'],
            'prediction_drift': results['prediction_drift']['l1_distance'],
            'concept_drift': results['concept_drift']['f1_drop'],
            'accuracy': results['concept_drift']['current_metrics']['accuracy'],
            'precision': results['concept_drift']['current_metrics']['precision'],
            'recall': results['concept_drift']['current_metrics']['recall'],
            'f1_score': results['concept_drift']['current_metrics']['f1'],
            'alerts': ','.join(results['alerts']) if results['alerts'] else 'None'
        }
        
        # Add roc_auc if available
        if 'roc_auc' in results['concept_drift']['current_metrics']:
            row['roc_auc'] = results['concept_drift']['current_metrics']['roc_auc']
        else:
            row['roc_auc'] = None
        
        # Load existing history, append new row, and save
        history = pd.read_csv(self.history_path)
        history = pd.concat([history, pd.DataFrame([row])])
        history.to_csv(self.history_path, index=False)
        
        logger.info(f"Saved drift results to history file")
    
    def _plot_feature_drift(self, df, drift_results, dataset_name):
        """
        Create and save a feature drift visualization.
        
        Parameters:
            df (DataFrame): Current data
            drift_results (dict): Feature drift metrics
            dataset_name (str): Name of the dataset
        
        Returns:
            str: Path to saved plot
        """
        # Get top 10 drifting features
        top_features = list(drift_results['top_drifting_features'].items())
        top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:10]
        
        plt.figure(figsize=(12, 6))
        feature_names = [f[0] for f in top_features]
        drift_values = [f[1] for f in top_features]
        
        # Create bar chart of feature drift
        bars = plt.bar(feature_names, drift_values)
        plt.axhline(y=self.thresh['feature_drift'], color='r', linestyle='--', label='Threshold')
        
        # Color bars based on threshold
        for i, v in enumerate(drift_values):
            if v > self.thresh['feature_drift']:
                bars[i].set_color('red')
            else:
                bars[i].set_color('blue')
        
        plt.title(f'Top 10 Feature Drift - {dataset_name}')
        plt.xlabel('Features')
        plt.ylabel('Wasserstein Distance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.figures_path, f'feature_drift_{dataset_name.replace("/", "_")}.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _plot_drift_trends(self):
        """
        Create and save visualizations of drift trends over time.
        
        Returns:
            str: Path to saved plot
        """
        if not os.path.exists(self.history_path):
            logger.warning("No history file found for trend analysis")
            return None
        
        # Load history data
        history = pd.read_csv(self.history_path)
        
        if len(history) < 2:
            logger.warning("Not enough history data for trend analysis")
            return None
        
        # Convert timestamp to datetime
        history['timestamp'] = pd.to_datetime(history['timestamp'])
        
        # Create plot with 4 subplots (one for each drift type)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Drift Metrics Over Time', fontsize=16)
        
        # Feature drift
        axs[0, 0].plot(history['timestamp'], history['feature_drift'], marker='o')
        axs[0, 0].axhline(y=self.thresh['feature_drift'], color='r', linestyle='--', label='Threshold')
        axs[0, 0].set_title('Feature Drift')
        axs[0, 0].set_ylabel('Wasserstein Distance')
        axs[0, 0].tick_params(axis='x', rotation=45)
        axs[0, 0].legend()
        
        # Target drift
        axs[0, 1].plot(history['timestamp'], history['target_drift'], marker='o')
        axs[0, 1].axhline(y=self.thresh['target_drift'], color='r', linestyle='--', label='Threshold')
        axs[0, 1].set_title('Target Drift')
        axs[0, 1].set_ylabel('L1 Distance')
        axs[0, 1].tick_params(axis='x', rotation=45)
        axs[0, 1].legend()
        
        # Prediction drift
        axs[1, 0].plot(history['timestamp'], history['prediction_drift'], marker='o')
        axs[1, 0].axhline(y=self.thresh['prediction_drift'], color='r', linestyle='--', label='Threshold')
        axs[1, 0].set_title('Prediction Drift')
        axs[1, 0].set_ylabel('L1 Distance')
        axs[1, 0].tick_params(axis='x', rotation=45)
        axs[1, 0].legend()
        
        # Concept drift
        axs[1, 1].plot(history['timestamp'], history['concept_drift'], marker='o')
        axs[1, 1].axhline(y=self.thresh['performance_drop'], color='r', linestyle='--', label='Threshold')
        axs[1, 1].set_title('Concept Drift (F1 Score Drop)')
        axs[1, 1].set_ylabel('F1 Drop')
        axs[1, 1].tick_params(axis='x', rotation=45)
        axs[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_path, 'drift_trends.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _plot_performance_metrics(self):
        """
        Create and save visualizations of performance metrics over time.
        
        Returns:
            str: Path to saved plot
        """
        if not os.path.exists(self.history_path):
            logger.warning("No history file found for performance trend analysis")
            return None
        
        # Load history data
        history = pd.read_csv(self.history_path)
        
        if len(history) < 2:
            logger.warning("Not enough history data for performance trend analysis")
            return None
        
        # Convert timestamp to datetime
        history['timestamp'] = pd.to_datetime(history['timestamp'])
        
        # Create plot for performance metrics
        plt.figure(figsize=(12, 6))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            if metric in history.columns:
                plt.plot(history['timestamp'], history[metric], marker='o', label=metric)
        
        plt.title('Model Performance Metrics Over Time')
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.figures_path, 'performance_trends.png')
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _log_to_mlflow(self, dataset_name, results, plot_paths):
        """
        Log drift detection results to MLflow.
        
        Parameters:
            dataset_name (str): Name of the dataset
            results (dict): Drift detection results
            plot_paths (list): Paths to visualization plots
        
        Returns:
            str: MLflow run ID
        """
        # Setup MLflow
        setup_mlflow()
        
        # Start a run with a specific name
        with mlflow.start_run(run_name=f"drift_detection_{Path(dataset_name).stem}"):
            # Log tags
            mlflow.set_tag("model_name", self.model_name)
            mlflow.set_tag("dataset", dataset_name)
            mlflow.set_tag("reference_data", self.ref_path)
            mlflow.set_tag("timestamp", datetime.datetime.now().isoformat())
            
            # Log metrics
            # Feature drift metrics
            mlflow.log_metric("feature_drift", results['feature_drift']['avg_wasserstein'])
            
            for feature, value in results['feature_drift']['top_drifting_features'].items():
                if len(results['feature_drift']['top_drifting_features']) <= 10:  # Limit to avoid too many metrics
                    mlflow.log_metric(f"feature_drift_{feature}", value)
            
            # Target drift metrics
            mlflow.log_metric("target_drift", results['target_drift']['l1_distance'])
            mlflow.log_metric("target_positive_rate_change", 
                              abs(results['target_drift']['reference_positive_rate'] - 
                                  results['target_drift']['current_positive_rate']))
            
            # Prediction drift metrics
            mlflow.log_metric("prediction_drift", results['prediction_drift']['l1_distance'])
            mlflow.log_metric("prediction_positive_rate_change",
                              abs(results['prediction_drift']['reference_positive_rate'] - 
                                  results['prediction_drift']['current_positive_rate']))
            
            if 'probability_wasserstein' in results['prediction_drift']:
                mlflow.log_metric("prediction_probability_drift", 
                                 results['prediction_drift']['probability_wasserstein'])
            
            # Concept drift metrics
            mlflow.log_metric("concept_drift_f1_drop", results['concept_drift']['f1_drop'])
            
            # Log all performance metrics
            for metric, value in results['concept_drift']['current_metrics'].items():
                mlflow.log_metric(f"current_{metric}", value)
                mlflow.log_metric(f"reference_{metric}", results['concept_drift']['reference_metrics'][metric])
                mlflow.log_metric(f"{metric}_drop", results['concept_drift']['performance_drops'][metric])
            
            # Log number of alerts
            mlflow.log_metric("alert_count", len(results['alerts']))
            
            # Log drift summary as a parameter
            drift_summary = {
                "feature_drift": results['feature_drift']['avg_wasserstein'] > self.thresh['feature_drift'],
                "target_drift": results['target_drift']['l1_distance'] > self.thresh['target_drift'],
                "prediction_drift": results['prediction_drift']['l1_distance'] > self.thresh['prediction_drift'],
                "concept_drift": results['concept_drift']['f1_drop'] > self.thresh['performance_drop']
            }
            mlflow.log_param("drift_detected", any(drift_summary.values()))
            
            # Log full results as a JSON artifact
            results_path = os.path.join(self.results_path, f'drift_results_{Path(dataset_name).stem}.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            mlflow.log_artifact(results_path)
            
            # Log plots as artifacts
            for plot_path in plot_paths:
                if plot_path and os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path)
            
            # Get the run ID
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Logged drift detection results to MLflow run: {run_id}")
            
            return run_id
    
    def run_all_checks(self, log_to_mlflow=True):
        """
        Run all drift detection checks on available production data.
        
        Parameters:
            log_to_mlflow (bool): Whether to log results to MLflow
            
        Returns:
            dict: Dictionary of all drift detection results
        """
        results_by_dataset = {}
        
        for path in self.prod_paths:
            try:
                df = pd.read_parquet(path)
                dataset_name = Path(path).name
                logger.info(f"\n--- Running drift checks on {dataset_name} ---")
                
                # Run all checks
                feature_drift = self._check_feature_drift(df)
                target_drift = self._check_target_drift(df)
                prediction_drift = self._check_prediction_drift(df)
                concept_drift = self._check_concept_drift(df)
                
                # Collect alerts
                alerts = []
                if feature_drift['avg_wasserstein'] > self.thresh['feature_drift']:
                    alerts.append("Feature drift exceeded")
                if target_drift['l1_distance'] > self.thresh['target_drift']:
                    alerts.append("Target drift exceeded")
                if prediction_drift['l1_distance'] > self.thresh['prediction_drift']:
                    alerts.append("Prediction drift exceeded")
                if concept_drift['f1_drop'] > self.thresh['performance_drop']:
                    alerts.append("Concept drift exceeded")
                
                # Log alerts
                if alerts:
                    for a in alerts:
                        logger.warning(a)
                else:
                    logger.info("No drift detected.")
                
                # Combine all results
                results = {
                    'feature_drift': feature_drift,
                    'target_drift': target_drift,
                    'prediction_drift': prediction_drift,
                    'concept_drift': concept_drift,
                    'alerts': alerts,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Generate visualizations
                plot_paths = []
                feature_plot = self._plot_feature_drift(df, feature_drift, dataset_name)
                plot_paths.append(feature_plot)
                
                # Save results to history
                self._save_drift_results(dataset_name, results)
                
                # Create trend visualizations after updating history
                trend_plot = self._plot_drift_trends()
                if trend_plot:
                    plot_paths.append(trend_plot)
                
                performance_plot = self._plot_performance_metrics()
                if performance_plot:
                    plot_paths.append(performance_plot)
                
                # Log to MLflow if requested
                if log_to_mlflow:
                    run_id = self._log_to_mlflow(dataset_name, results, plot_paths)
                    results['mlflow_run_id'] = run_id
                
                # Store results for this dataset
                results_by_dataset[dataset_name] = results
                
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
        return results_by_dataset


def main():
    """Main entry point for the enhanced drift detector."""
    # Configure detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("drift_detector.log"),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting enhanced drift detection")
    
    # Initialize the detector
    detector = EnhancedDriftDetector("config.yaml")
    
    # Run all checks
    results = detector.run_all_checks(log_to_mlflow=True)
    
    # Summary
    dataset_count = len(results)
    alert_count = sum(len(r["alerts"]) for r in results.values())
    
    logger.info(f"Completed drift detection for {dataset_count} datasets")
    logger.info(f"Total alerts: {alert_count}")
    
    return results


if __name__ == "__main__":
    main() 