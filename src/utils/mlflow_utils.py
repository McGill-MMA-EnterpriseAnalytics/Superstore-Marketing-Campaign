import logging
import os
import platform
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

#from utils.config import get_paths
from .config import get_paths

logger = logging.getLogger(__name__)


def setup_mlflow():
    """
    Set up MLflow tracking.
    
    Returns:
        str: MLflow experiment ID
    """
    # Get paths from config
    paths = get_paths()
    
    # Configure MLflow
    tracking_uri = paths.get("mlflow_tracking", "mlruns")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    # Create directory if it doesn't exist
    os.makedirs(tracking_uri, exist_ok=True)
    
    # Set tracking URI - Windows requires special handling
    if platform.system() == "Windows":
        # On Windows, use the local directory without file:// protocol
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # On other platforms, use file:// protocol
        mlflow.set_tracking_uri(f"file://{Path(tracking_uri).absolute()}")
    
    # Get or create experiment
    experiment_name = paths.get("experiment_name", "marketing-campaign")
    
    # Get experiment if it exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        # Create experiment - handle Windows path differently
        if platform.system() == "Windows":
            artifact_path = paths.get('artifact_path', 'artifacts')
            os.makedirs(artifact_path, exist_ok=True)
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_path
            )
        else:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=f"file://{Path(paths.get('artifact_path', 'artifacts')).absolute()}"
            )
        logger.info(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
    
    return experiment_id


def log_model_metrics(run_id=None, model_name=None, metrics=None, params=None, tags=None, artifacts=None):
    """
    Log model metrics to MLflow, using nested runs if one is already active.
    """
    # Check for existing active run
    active = mlflow.active_run()

    # Decide how to start
    if active is None:
        # No run yet: either resume by run_id or start a fresh top‐level run
        if run_id:
            run = mlflow.start_run(run_id=run_id)
        else:
            run = mlflow.start_run(run_name=model_name)
    else:
        # Already inside a run (e.g. your Optuna wrapper) -> start a nested run
        run = mlflow.start_run(nested=True, run_name=model_name)

    with run:
        # Optional: tag the run with the model name
        if model_name:
            mlflow.set_tag("model_name", model_name)

        # Log any user‐passed tags
        if tags:
            mlflow.set_tags(tags)

        # Log parameters
        if params:
            mlflow.log_params(params)

        # Log metrics
        if metrics:
            mlflow.log_metrics(metrics)

        # Log artifacts
        if artifacts:
            for path in artifacts:
                mlflow.log_artifact(path)

        current_run_id = mlflow.active_run().info.run_id
        logger.info(
            f"{'Nested' if active else 'Top‐level'} run {current_run_id} "
            f"logged metrics for {model_name}"
        )
        return current_run_id


def log_best_model(model, model_name, metrics, feature_names=None, feature_importance=None):
    """
    Log the best model to MLflow model registry.
    
    Parameters:
        model: Trained model object
        model_name (str): Name of the model
        metrics (dict): Dictionary of metrics
        feature_names (list, optional): List of feature names
        feature_importance (list, optional): List of feature importances
    
    Returns:
        str: Model URI
    """
    # Ensure the registered model exists; create once if missing
    client = MlflowClient()
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)

    with mlflow.start_run(run_name=f"{model_name}_best"):
        # Log model name as a tag
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("stage", "production")
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log feature importance if available
        if feature_names and feature_importance:
            importance_dict = dict(zip(feature_names, feature_importance))
            for name, value in importance_dict.items():
                # Convert to float to ensure it can be serialized
                if isinstance(value, (int, float)):
                    mlflow.log_param(f"importance_{name}", float(value))
        
        # Determine the flavor to use based on model type
        if "XGBoost" in model.__class__.__name__:
            if hasattr(model, "xgb_model"):
                model_info = mlflow.xgboost.log_model(
                    model.xgb_model,
                    "model",
                    registered_model_name=model_name
                )
            else:
                model_info = mlflow.xgboost.log_model(
                    model,
                    "model",
                    registered_model_name=model_name
                )
        elif "LGBM" in model.__class__.__name__:
            if hasattr(model, "lgbm_model"):
                model_info = mlflow.lightgbm.log_model(
                    model.lgbm_model,
                    "model",
                    registered_model_name=model_name
                )
            else:
                model_info = mlflow.lightgbm.log_model(
                    model,
                    "model",
                    registered_model_name=model_name
                )
        elif "CatBoost" in model.__class__.__name__:
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name
            )
        else:
            model_info = mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name
            )
        
        logger.info(f"Logged best model {model_name} to MLflow model registry")
        
        return model_info.model_uri


def get_best_run(experiment_name, metric="f1", ascending=False):
    """
    Get the best run for a given metric.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        logger.error(f"Experiment {experiment_name} not found")
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        logger.warning(f"No runs found for experiment {experiment_name}")
        return None
    
    metric_col = f"metrics.{metric}"
    if metric_col in runs.columns:
        runs = runs.sort_values(metric_col, ascending=ascending)
        return runs.iloc[0]
    else:
        logger.warning(f"Metric {metric} not found in runs")
        return None


def get_model_from_registry(model_name, stage="Production"):
    """
    Get a model from the MLflow registry.
    """
    client = MlflowClient()
    
    try:
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if not latest_versions:
            logger.error(f"No {model_name} model found in {stage} stage")
            return None
        
        model_version = latest_versions[0]
        logger.info(f"Loading {model_name} version {model_version.version} from {stage} stage")
        
        model_uri = f"models:/{model_name}/{model_version.version}"
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"Error loading model with pyfunc: {e}")
            try:
                return mlflow.sklearn.load_model(model_uri)
            except Exception as e:
                logger.error(f"Error loading model with sklearn: {e}")
                raise ValueError(f"Could not load model {model_name} from registry")
    except Exception as e:
        logger.error(f"Error getting model from registry: {e}")
        return None


def compare_runs(experiment_name, metric_name="f1", n_runs=5):
    """
    Compare top runs in an experiment based on a metric.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        logger.error(f"Experiment {experiment_name} not found")
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        logger.warning(f"No runs found for experiment {experiment_name}")
        return None
    
    metric_col = f"metrics.{metric_name}"
    if metric_col in runs.columns:
        runs = runs.sort_values(metric_col, ascending=False)
        return runs.head(n_runs)
    else:
        logger.warning(f"Metric {metric_name} not found in runs")
        return None
