import logging
import os
from functools import partial

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
import mlflow

from src.utils.config import get_data_paths, get_model_config, get_paths
from src.utils.mlflow_utils import setup_mlflow, log_model_metrics
from src.train_model import load_data, get_model, evaluate_model_on_test, save_model



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Optuna logging
optuna.logging.set_verbosity(optuna.logging.INFO)


def objective(trial, model_name, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna to optimize.
    
    Parameters:
        trial: Optuna trial object
        model_name: String name of the model to optimize
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        float: F1 score on validation data
    """
    # Define parameter search space based on model type
    params = {}
    
    if model_name == "xgboost":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42
        }
    elif model_name == "lightgbm":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha',  1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda',  1e-8, 1.0, log=True),
            'random_state': 42
        }
    elif model_name == "catboost":
        params = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'random_seed': 42
        }
    elif model_name == "randomforest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42
        }
    
    # For catboost, we need to handle cat_features
    if model_name == "catboost":
        model_result = get_model(model_name, tuned=False)
        if isinstance(model_result, tuple):
            # Get the model and cat_features
            base_model, cat_features = model_result
            # Update parameters
            for key, value in params.items():
                setattr(base_model, key, value)
            model = base_model
        else:
            model = model_result
            cat_features = None
            # Update parameters
            for key, value in params.items():
                setattr(model, key, value)
    else:
        # For other models, directly get the model with the suggested parameters
        model = get_model(model_name, tuned=False)
        
        # Update model parameters (this works for sklearn-like models)
        if hasattr(model, 'set_params'):
            model.set_params(**params)
        else:
            for key, value in params.items():
                setattr(model, key, value)
    
    # Special handling for XGBoost and object columns
    if model_name == "xgboost":
        object_cols = X_train.select_dtypes(include=['object']).columns
        X_train_clean = X_train.copy()
        X_val_clean = X_val.copy()
        
        # Convert object columns
        for col in object_cols:
            X_train_clean[col] = X_train_clean[col].astype('category').cat.codes
            X_val_clean[col] = X_val_clean[col].astype('category').cat.codes
            
        # Clean column names for XGBoost
        new_columns = {}
        for col in X_train_clean.columns:
            new_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            if new_col != col:
                new_columns[col] = new_col
                
        if new_columns:
            X_train_clean = X_train_clean.rename(columns=new_columns)
            X_val_clean = X_val_clean.rename(columns=new_columns)
            
        # Fit the model
        model.fit(X_train_clean, y_train)
        y_pred = model.predict(X_val_clean)
    elif model_name == "catboost" and cat_features is not None:
        # For CatBoost with categorical features
        cat_features_indices = []
        for i, col in enumerate(X_train.columns):
            if X_train[col].dtype == 'object' or "education_" in col or "marital_status_" in col:
                cat_features_indices.append(i)
                
        # Fit the model
        model.fit(X_train, y_train, cat_features=cat_features_indices)
        y_pred = model.predict(X_val)
    else:
        # For other models, handle object columns generically
        object_cols = X_train.select_dtypes(include=['object']).columns
        X_train_clean = X_train.copy()
        X_val_clean = X_val.copy()
        
        for col in object_cols:
            X_train_clean[col] = X_train_clean[col].astype('category').cat.codes
            X_val_clean[col] = X_val_clean[col].astype('category').cat.codes
            
        # Fit the model
        model.fit(X_train_clean, y_train)
        y_pred = model.predict(X_val_clean)
    
    # Calculate F1 score
    score = f1_score(y_val, y_pred)
    
    # Log trial information
    logger.info(f"Trial {trial.number}: F1 Score = {score:.4f}")
    
    # Log parameters
    trial.set_user_attr("params", params)
    
    return score


def optimize_hyperparameters(model_name, n_trials=20):
    """
    Run hyperparameter optimization for a specific model.
    
    Parameters:
        model_name: String name of the model to optimize
        n_trials: Number of trials for optimization
        
    Returns:
        best_params: Dictionary of best parameters
        best_value: Best F1 score achieved
    """
    # Set up MLflow
    setup_mlflow()
    
    # Load data
    logger.info(f"Loading data for hyperparameter tuning of {model_name}...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Create an Optuna study
    study_name = f"{model_name}_optimization"
    logger.info(f"Creating Optuna study '{study_name}'...")
    
    try:
        # Create a SQLite storage for the study
        paths = get_paths()
        os.makedirs(paths.get("models", "models/"), exist_ok=True)
        storage_path = f"sqlite:///{paths.get('models', 'models/')}/{model_name}_optuna.db"
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage_path,
            load_if_exists=True
        )
    except Exception as e:
        logger.warning(f"Error creating study with SQLite storage: {e}")
        logger.info("Falling back to in-memory storage...")
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize"
        )
    
    # Create objective function with fixed data
    objective_func = partial(
        objective,
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_name}_optimization"):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("optimization_type", "optuna")
        
        # Optimize
        logger.info(f"Starting optimization with {n_trials} trials...")
        study.optimize(objective_func, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Log best parameters and result
        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1", best_value)
        
        # Log optimization history as a plot
        try:
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.write_image("optuna_history.png")
            mlflow.log_artifact("optuna_history.png")
        except Exception as e:
            logger.warning(f"Could not create optimization history plot: {e}")
        
        logger.info(f"Best F1 score: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Train the final model with best parameters
        logger.info("Training final model with best parameters...")
        
        # Update model configuration with best parameters
        model_config = get_model_config(model_name, tuned=True)
        model_config.update(best_params)
        
        # For catboost, we need to handle cat_features
        if model_name == "catboost":
            model_result = get_model(model_name, tuned=True)
            if isinstance(model_result, tuple):
                model, cat_features = model_result
                # Update parameters
                for key, value in best_params.items():
                    setattr(model, key, value)
            else:
                model = model_result
                cat_features = None
                # Update parameters
                for key, value in best_params.items():
                    setattr(model, key, value)
        else:
            # For other models, directly get the model
            model = get_model(model_name, tuned=True)
            # Update model parameters
            if hasattr(model, 'set_params'):
                model.set_params(**best_params)
            else:
                for key, value in best_params.items():
                    setattr(model, key, value)
        
        # Train the model with best parameters
        from src.train_model import train_model
        tuned_model, tuned_metrics, tuned_feature_importance = train_model(
        #tuned_model, tuned_metrics = train_model(
            model, X_train, y_train, X_val, y_val, 
            cat_features=cat_features if model_name == "catboost" else None,
            model_name=f"{model_name}_automl_tuned",
            tuned=True
        )
        
        # Save tuned model
        tuned_model_path = save_model(tuned_model, f"{model_name}_automl", tuned=True)
        
        # Evaluate on test set
        tuned_test_metrics = evaluate_model_on_test(
            tuned_model, X_test, y_test, f"{model_name}_automl_tuned"
        )
        
        # Save best parameters to a file
        import json
        os.makedirs(paths.get("models", "models/"), exist_ok=True)
        params_path = os.path.join(paths.get("models", "models/"), f"{model_name}_best_params.json")
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Best parameters saved to {params_path}")
        
        return best_params, best_value


def main():
    """Main function to run hyperparameter optimization for all models."""
    # List of models to optimize
    models = ["xgboost", "lightgbm", "randomforest", "catboost"]
    
    results = {}
    
    for model_name in models:
        logger.info(f"Starting hyperparameter optimization for {model_name}...")
        best_params, best_score = optimize_hyperparameters(model_name, n_trials=20)
        results[model_name] = {
            "best_params": best_params,
            "best_score": best_score
        }
        logger.info(f"Completed optimization for {model_name}")
    
    # Find best model overall
    best_model = max(results.items(), key=lambda x: x[1]["best_score"])
    logger.info(f"Best overall model: {best_model[0]} with F1 score: {best_model[1]['best_score']:.4f}")
    
    return results


if __name__ == "__main__":
    main()