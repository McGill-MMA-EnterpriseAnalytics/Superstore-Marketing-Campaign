"""
This module implements causal inference models for analyzing marketing campaign effects.

The module provides functionality for:
- Propensity score matching
- Treatment effect estimation
- Causal inference analysis
"""

import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from causalml.inference.meta import BaseRRegressor
from sklearn.linear_model import LogisticRegression
from src.utils.config import load_config, get_data_paths
from sklearn.model_selection import KFold
import optuna
import os
from datetime import datetime
import joblib
import json
from sklearn.preprocessing import StandardScaler


class CausalModel:
    """
    A class for performing causal inference analysis on marketing campaign data.
    
    This class implements various methods for estimating causal effects including:
    - Propensity score matching
    - Average Treatment Effect (ATE) estimation
    - Feature importance analysis
    """
    
    def __init__(self):
        self.propensity_model = None
        self.r_learner = None
        self.results = []
        self.feature_names = None
        
        # Define target variables
        self.target_vars = [
            'mntwines', 'mntfruits', 'mntmeatproducts', 
            'mntfishproducts', 'mntsweetproducts', 'mntgoldprods'
        ]
        
        # Define control variables
        self.controls = [
            'year_birth', 'income', 'kidhome', 'teenhome', 'recency',
            'numwebpurchases', 'numcatalogpurchases', 'numstorepurchases',
            'numwebvisitsmonth', 'complain', 'customer_segment',
            'enrollments_year', 'enrollments_month', 
            'education_Graduate', 'education_Post Graduate', 'education_Pre Graduate',
            'marital_status_Divorced', 'marital_status_Married',
            'marital_status_Single', 'marital_status_Together',
            'marital_status_Widow'
        ]
    
    def load_data(self, train_path, test_path, val_path):
        """
        Load and combine the training, test, and validation datasets.
        
        Args:
            train_path (str): Path to training data parquet file
            test_path (str): Path to test data parquet file
            val_path (str): Path to validation data parquet file
            
        Returns:
            pd.DataFrame: Combined dataset
        """
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        val_df = pd.read_parquet(val_path)
        
        # Combine datasets
        df = pd.concat([train_df, test_df, val_df])
        
        # Print unique values
        print("\nUnique education values:", df['education'].unique())
        print("\nUnique marital status values:", df['marital_status'].unique())
        
        # Create dummy variables for education and marital status
        education_dummies = pd.get_dummies(df['education'], prefix='education')
        marital_dummies = pd.get_dummies(df['marital_status'], prefix='marital_status')
        
        # Print created dummy columns
        print("\nCreated education columns:", education_dummies.columns.tolist())
        print("\nCreated marital status columns:", marital_dummies.columns.tolist())
        
        # Drop original columns and add dummy variables
        df = df.drop(['education', 'marital_status'], axis=1)
        df = pd.concat([df, education_dummies, marital_dummies], axis=1)
        
        return df
    
    def fit_propensity_score(self, X, T):
        """
        Fit a propensity score model to estimate probability of treatment assignment.
        
        Args:
            X (pd.DataFrame): Feature matrix
            T (pd.Series): Binary treatment indicator
            
        Returns:
            np.array: Propensity scores
        """
        self.propensity_model = LogisticRegression(max_iter=500)
        self.propensity_model.fit(X, T)
        return self.propensity_model.predict_proba(X)[:, 1]
    
    def compute_pehe_score(self, r_learner, X_val, T_val, Y_val, p_val):
        """
        Compute Precision in Estimation of Heterogeneous Effect (PEHE) score.
        Lower PEHE scores indicate better performance.
        
        Args:
            r_learner: Fitted R-learner model
            X_val: Validation feature matrix
            T_val: Validation treatment indicators
            Y_val: Validation outcomes
            p_val: Validation propensity scores
            
        Returns:
            float: PEHE score
        """
        # Predict treatment effects
        tau_pred = r_learner.predict(X_val)
        
        # Estimate observed treatment effects (approximate using matching)
        treated_idx = T_val == 1
        control_idx = T_val == 0
        
        # Calculate treatment effects using nearest neighbor matching
        from sklearn.neighbors import NearestNeighbors
        
        nn_treated = NearestNeighbors(n_neighbors=1)
        nn_control = NearestNeighbors(n_neighbors=1)
        
        nn_treated.fit(X_val[treated_idx])
        nn_control.fit(X_val[control_idx])
        
        # For treated units, find nearest control
        _, indices = nn_control.kneighbors(X_val[treated_idx])
        control_outcomes = Y_val[control_idx].iloc[indices.flatten()]
        treated_effects = Y_val[treated_idx].values - control_outcomes.values
        
        # For control units, find nearest treated
        _, indices = nn_treated.kneighbors(X_val[control_idx])
        treated_outcomes = Y_val[treated_idx].iloc[indices.flatten()]
        control_effects = treated_outcomes.values - Y_val[control_idx].values
        
        # Combine treatment effects
        tau_obs = np.zeros(len(Y_val))
        tau_obs[treated_idx] = treated_effects
        tau_obs[control_idx] = control_effects
        
        # Calculate PEHE
        pehe = np.sqrt(np.mean((tau_pred - tau_obs) ** 2))
        return pehe

    def tune_xgboost_hyperparameters(self, X, T, Y, p, n_trials=50):
        """
        Tune XGBoost hyperparameters for R-learner using Optuna and PEHE score.
        
        Args:
            X: Feature matrix
            T: Treatment indicators
            Y: Outcomes
            p: Propensity scores
            n_trials: Number of optimization trials
            
        Returns:
            dict: Best hyperparameters
        """
        # Convert data types to float32 to improve numerical stability
        X = X.astype('float32')
        Y = Y.astype('float32')
        T = T.astype('float32')
        p = p.astype('float32')
        
        def objective(trial):
            # Define hyperparameter space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.2),
                'objective': 'reg:squarederror',
                'random_state': 42,
                'verbosity': 0  # Suppress XGBoost messages
            }
            
            # Cross-validation
            scores = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X):
                # Split data
                X_train = X.iloc[train_idx]
                X_val = X.iloc[val_idx]
                T_train = T.iloc[train_idx]
                T_val = T.iloc[val_idx]
                Y_train = Y.iloc[train_idx]
                Y_val = Y.iloc[val_idx]
                p_train = p[train_idx]
                p_val = p[val_idx]
                
                try:
                    # Create and fit model with specific configuration to minimize warnings
                    xgb_model = xgb.XGBRegressor(**params)
                    r_learner = BaseRRegressor(
                        learner=xgb_model,
                        random_state=42
                    )
                    
                    # Fit with error handling
                    try:
                        r_learner.fit(X=X_train, treatment=T_train, y=Y_train, p=p_train)
                    except Exception as e:
                        print(f"Warning: Error during fitting: {str(e)}")
                        return float('inf')  # Return worst possible score
                    
                    # Compute PEHE score
                    pehe = self.compute_pehe_score(r_learner, X_val, T_val, Y_val, p_val)
                    scores.append(pehe)
                except Exception as e:
                    print(f"Warning: Error in CV fold: {str(e)}")
                    return float('inf')  # Return worst possible score
            
            if not scores:
                return float('inf')
            
            return np.mean(scores)  # Return mean PEHE score (lower is better)
        
        # Create and run Optuna study
        study = optuna.create_study(direction='minimize')  # Minimize PEHE
        study.optimize(objective, n_trials=n_trials)
        
        # Store best parameters
        self.best_params = study.best_params.copy()
        self.best_params['objective'] = 'reg:squarederror'
        self.best_params['random_state'] = 42
        self.best_params['verbosity'] = 0
        
        return self.best_params

    def estimate_ate(self, X, T, Y, propensity_score):
        """
        Estimate the Average Treatment Effect (ATE) using R-learner with XGBoost.
        Uses standard XGBoost parameters.
        
        Args:
            X (pd.DataFrame): Feature matrix
            T (pd.Series): Binary treatment indicator
            Y (pd.Series): Outcome variable
            propensity_score (np.array): Propensity scores
            
        Returns:
            tuple: (ATE estimate, lower bound, upper bound, tau estimates)
        """
        print("\nData sizes before processing:")
        print(f"X shape: {X.shape}")
        print(f"Treatment group size: {T.sum()}")
        print(f"Control group size: {len(T) - T.sum()}")
        print(f"Y size: {len(Y)}")
        
        # Create XGBoost model with standard parameters
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0
        )
        
        # Initialize R-learner with minimal configuration
        self.r_learner = BaseRRegressor(
            learner=xgb_model,
            random_state=42
        )
        
        # Convert data types to float32 to improve numerical stability
        X = X.astype('float32')
        Y = Y.astype('float32')
        T = T.astype('float32')
        propensity_score = propensity_score.astype('float32')
        
        
        print("\nData sizes after removing missing values:")
        print(f"X shape: {X.shape}")
        print(f"Treatment group size: {T.sum()}")
        print(f"Control group size: {len(T) - T.sum()}")
        print(f"Y size: {len(Y)}")
        
        # Print value distributions
        print("\nValue distributions:")
        print("Y values:", Y.value_counts().head())
        print("\nT values:", T.value_counts())
        print("\nPropensity score range:", [propensity_score.min(), propensity_score.max()])
        
        # Ensure we have enough samples in both treatment and control groups
        n_treatment = T.sum()
        n_control = len(T) - n_treatment
        min_samples = 20
        
        if n_treatment < min_samples or n_control < min_samples:
            print(f"\nWarning: Not enough samples in treatment ({n_treatment}) or control ({n_control}) groups")
            print("Proceeding with estimation but results may be unreliable")
        
        # Fit and estimate effects
        try:
            print("\nStarting ATE estimation...")
            te, lb, ub = self.r_learner.estimate_ate(X=X, treatment=T, y=Y, p=propensity_score)
            print("ATE estimation completed successfully")
            
            print("\nStarting tau prediction...")
            tau = self.r_learner.fit_predict(X=X, treatment=T, y=Y, p=propensity_score)
            print("Tau prediction completed successfully")
            
        except Exception as e:
            print(f"\nWarning: Error during ATE estimation: {str(e)}")
            print("Falling back to default parameters...")
            # Fallback to simpler configuration if error occurs
            self.r_learner = BaseRRegressor(
                learner=xgb_model,  
                random_state=42
            )
            te, lb, ub = self.r_learner.estimate_ate(X=X, treatment=T, y=Y, p=propensity_score)
            tau = self.r_learner.fit_predict(X=X, treatment=T, y=Y, p=propensity_score)
        
        return te[0], lb[0], ub[0], tau
    
    def analyze_target_variable(self, df, target):
        """
        Analyze a single target variable for causal effects.
        
        Args:
            df (pd.DataFrame): Combined dataset
            target (str): Target variable name
            
        Returns:
            dict: Results dictionary containing ATE and other metrics
        """
        print(f"\nProcessing {target}")
        print("\nInitial data sizes:")
        print(f"Total samples: {len(df)}")
        print(f"Treatment group size: {df['target'].sum()}")
        print(f"Control group size: {len(df) - df['target'].sum()}")
        
        # Prepare the data
        Y = df[target]
        X = df[self.controls + self.target_vars].drop(target, axis=1)
        T = df['target']
        
        print("\nTarget variable statistics before filtering:")
        print(f"Min value: {Y.min()}")
        print(f"Max value: {Y.max()}")
        print(f"Mean value: {Y.mean():.2f}")
        print(f"Number of zeros: {(Y == 0).sum()}")
        print(f"Number of missing values: {Y.isna().sum()}")
        
        # Remove rows where target variable is 0 or missing
        mask = (Y > 0) & (~Y.isna())
        Y = Y[mask]
        X = X.loc[mask]
        T = T[mask]
        
        # Print data statistics
        print(f"\nData statistics after preprocessing:")
        print(f"Total samples: {len(Y)}")
        print(f"Treatment group: {T.sum()}")
        print(f"Control group: {len(T) - T.sum()}")
        print(f"Target mean: {Y.mean():.2f}")
        print(f"Target std: {Y.std():.2f}")
        
        # Check for any remaining issues
        print("\nData quality checks:")
        print(f"Missing values in X: {X.isna().any().sum()}")
        print(f"Missing values in Y: {Y.isna().sum()}")
        print(f"Missing values in T: {T.isna().sum()}")
        
        # Get propensity scores
        propensity_score = self.fit_propensity_score(X, T)
        
        # Print propensity score statistics
        print("\nPropensity score statistics:")
        print(f"Min: {propensity_score.min():.4f}")
        print(f"Max: {propensity_score.max():.4f}")
        print(f"Mean: {propensity_score.mean():.4f}")
        
        # Estimate ATE
        te, lb, ub, tau = self.estimate_ate(X, T, Y, propensity_score)
        
        return {
            'target_var': target,
            'ate': te,
            'lb': lb,
            'ub': ub,
            'tau': tau,
            'r_learner': self.r_learner,
            'X': X,
            'T': T,
            'Y': Y
        }
    
    def run_analysis(self, df):
        """
        Run causal analysis on all target variables.
        
        Args:
            df (pd.DataFrame): Combined dataset
            
        Returns:
            pd.DataFrame: Results dataframe with ATE estimates for each target
        """
        self.results = []
        
        for target in self.target_vars:
            result = self.analyze_target_variable(df, target)
            self.results.append(result)
        
        return pd.DataFrame(self.results)
    
    def plot_feature_importance(self, X, tau, target_var, method="auto"):
        """
        Plot feature importance for a target variable.
        
        Args:
            X (pd.DataFrame): Feature matrix
            tau (np.array): Treatment effect estimates
            target_var (str): Target variable name
            method (str): Importance calculation method ("auto" or "permutation")
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Generate the feature importance plot
        self.r_learner.plot_importance(
            X=X, 
            tau=tau, 
            normalize=True, 
            method=method, 
            features=X.columns
        )
        
        # Add a title
        plt.title(f"Feature Importance by {method} with R-Learner for Tau on {target_var}")
        
        # Finalize the figure to ensure it's complete before returning
        fig = plt.gcf()  # Get current figure
        plt.tight_layout()
        
        return fig
    
    def plot_tau_distribution(self, target_var, tau_estimates):
        """
        Plot the distribution of treatment effects for a target variable.
        
        Args:
            target_var (str): Target variable name
            tau_estimates (np.array): Treatment effect estimates
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        fig = plt.figure(figsize=(10, 6))
        plt.hist(tau_estimates, bins=30, edgecolor='k', alpha=0.7)
        plt.title(f'Distribution of Estimated Treatment Effects for R-Learner on {target_var}')
        plt.xlabel('Estimated Treatment Effect (τ)')
        plt.ylabel('Frequency')
        plt.grid(True)
        return fig
    
    def plot_shap_values(self, X, tau, target_var):
        """
        Plot SHAP values for feature importance.
        
        Args:
            X (pd.DataFrame): Feature matrix
            tau (np.array): Treatment effect estimates
            target_var (str): Target variable name
            
        Returns:
            matplotlib.figure.Figure: The generated SHAP values plot
        """
        fig = plt.figure(figsize=(10, 6))
        self.r_learner.plot_shap_values(X=X, tau=tau, features=X.columns)
        plt.title(f"SHAP Values for {target_var}")
        return fig
    
    def get_top_features(self, X, tau, n_features=10):
        """
        Get the top n most important features.
        
        Args:
            X (pd.DataFrame): Feature matrix
            tau (np.array): Treatment effect estimates
            n_features (int): Number of top features to return
            
        Returns:
            dict: Top n features with their importance scores
        """
        try:
            # Get importance from r_learner
            importance = self.r_learner.get_importance(X=X, tau=tau, features=X.columns)
            
            # Handle different return types from get_importance
            if isinstance(importance, dict):
                # Ensure all values are scalar floats, not Series or other objects
                importance = {k: float(v) if hasattr(v, '__float__') else 0.0 
                             for k, v in importance.items()}
            elif isinstance(importance, pd.DataFrame):
                # Convert DataFrame to dictionary
                importance = importance.iloc[:, 0].to_dict()
            elif isinstance(importance, pd.Series):
                # Convert Series to dictionary
                importance = importance.to_dict()
            else:
                # Convert other types (like numpy arrays) to dictionary
                feature_names = X.columns.tolist()
                importance = {feature_names[i]: float(val) if i < len(feature_names) else 0.0 
                             for i, val in enumerate(importance) if hasattr(val, '__float__')}
            
            # Sort and get top features by absolute value
            sorted_importance = sorted(importance.items(), key=lambda x: abs(float(x[1])), reverse=True)
            return {k: float(v) for k, v in sorted_importance[:n_features]}
            
        except Exception as e:
            print(f"Error calculating importance from model: {str(e)}")
            print("Falling back to correlation-based importance")
            # Fallback to correlation-based importance
            correlations = self.calculate_feature_tau_correlations(X, tau)
            importance = correlations.abs()  # Use absolute correlation as importance
            # Ensure the returned dictionary has float values
            return {k: float(v) for k, v in importance.sort_values(ascending=False).head(n_features).items()}

    def calculate_feature_tau_correlations(self, X, tau):
        """
        Calculate correlations between features and treatment effects (tau).
        
        Args:
            X (pd.DataFrame): Feature matrix
            tau (np.array): Treatment effect estimates
            
        Returns:
            pd.Series: Correlations between features and tau
        """
        # Create a DataFrame with features and tau
        data = X.copy()
        data['tau'] = tau
        
        # Calculate correlations with tau
        correlations = data.corr()['tau'].drop('tau')
        
        return correlations


def main():
    """
    Main function to run the causal analysis on all target variables.
    Uses a standard XGBoost model for each target and reports ATE, bounds, and top features.
    Results are stored in an organized directory structure.
    """
    # Create base directory for all causal modeling results
    base_dir = "causal_modeling"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    # Create directory structure
    plots_dir = os.path.join(run_dir, "plots")
    models_dir = os.path.join(run_dir, "models")
    results_dir = os.path.join(run_dir, "results")
    
    # Create subdirectories for different types of plots
    feature_importance_dir = os.path.join(plots_dir, "feature_importance")
    tau_dist_dir = os.path.join(plots_dir, "tau_distributions")
    shap_dir = os.path.join(plots_dir, "shap_values")
    
    # Create all directories
    for directory in [feature_importance_dir, tau_dist_dir, shap_dir, 
                     models_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load configuration
    config = load_config()
    data_paths = get_data_paths(config)
    
    # Initialize the causal model
    model = CausalModel()
    
    # Load and combine the data
    df = model.load_data(
        data_paths.get('train', 'train.parquet'),
        data_paths.get('test', 'test.parquet'),
        data_paths.get('val', 'val.parquet')
    )
    
    # Create a results table
    results_table = []
    
    # Initialize correlation DataFrame with all features as rows and all targets as columns
    all_features = model.controls + model.target_vars
    correlation_df = pd.DataFrame(index=all_features, columns=model.target_vars)
    
    # For each target variable
    for target in model.target_vars:
        print(f"\n{'='*50}")
        print(f"Processing target: {target}")
        print(f"{'='*50}")
        
        # Prepare the data
        Y = df[target]
        X = df[model.controls + model.target_vars].drop(target, axis=1)
        T = df['target']
        propensity_score = model.fit_propensity_score(X, T)
        
        # Estimate effects
        print("\nEstimating treatment effects...")
        te, lb, ub, tau = model.estimate_ate(X, T, Y, propensity_score)
        
        # Calculate correlations between features and tau
        correlations = model.calculate_feature_tau_correlations(X, tau)
        
        # Store correlations in the DataFrame
        for feature in X.columns:
            correlation_df.loc[feature, target] = correlations[feature]
        
        # Set correlation to NA for the target variable with itself
        correlation_df.loc[target, target] = np.nan
        
        # Get top features
        top_features = model.get_top_features(X, tau, n_features=10)
        
        # Store results
        result = {
            'target': target,
            'ate': te,
            'lower_bound': lb,
            'upper_bound': ub,
            'top_features': top_features
        }
        results_table.append(result)
        
        # Print current target results
        print(f"\nResults for {target}:")
        print(f"ATE: {te:.4f} [{lb:.4f}, {ub:.4f}]")
        print("\nTop 10 features and their importance scores:")
        for feat, score in top_features.items():
            print(f"{feat}: {score:.4f}")
        
        # Generate and save plots
        print("\nGenerating and saving plots...")
        
        # Feature importance plot
        fig_importance = model.plot_feature_importance(X, tau, target, method="auto")
        
        # Ensure the figure is finalized before saving
        plt.figure(fig_importance.number)  # Make sure this figure is active
        plt.tight_layout()
        
        # Save with explicit format
        save_path = os.path.join(feature_importance_dir, f"{target}_feature_importance.png")
        fig_importance.savefig(save_path, format='png', bbox_inches='tight', dpi=300)
        print(f"Saved feature importance plot to: {save_path}")
        plt.close(fig_importance)
        
        # Tau distribution plot
        fig_tau = model.plot_tau_distribution(target, tau)
        fig_tau.savefig(os.path.join(tau_dist_dir, f"{target}_tau_distribution.png"), 
                       bbox_inches='tight', dpi=300)
        plt.close(fig_tau)
        
        # SHAP values plot
        fig_shap = model.plot_shap_values(X, tau, target)
        fig_shap.savefig(os.path.join(shap_dir, f"{target}_shap_values.png"), 
                        bbox_inches='tight', dpi=300)
        plt.close(fig_shap)
        
        # Save R-learner model
        joblib.dump(
            model.r_learner,
            os.path.join(models_dir, f"r_learner_{target}.joblib")
        )
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([
        {
            'Target': r['target'],
            'ATE': f"{r['ate']:.4f}",
            'CI': f"[{r['lower_bound']:.4f}, {r['upper_bound']:.4f}]",
            'Top 3 Features': ', '.join([str(feat) for feat in list(r['top_features'].keys())[:3]])
        }
        for r in results_table
    ])
    
    # Print summary
    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY")
    print("="*100)
    
    print("\nTreatment Effects Summary:")
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv(os.path.join(results_dir, 'summary.csv'), index=False)
    
    # Save correlation table
    correlation_df.to_csv(os.path.join(results_dir, 'feature_tau_correlations.csv'))
    
    # Print correlation table summary
    print("\nFeature-Tau Correlations Summary:")
    print("\nTop 3 positive correlations for each target:")
    for target in model.target_vars:
        correlations = correlation_df[target].dropna().sort_values(ascending=False)
        print(f"\n{target}:")
        for feature, corr in correlations.head(3).items():
            print(f"  {feature}: {corr:.4f}")
    
    print("\nTop 3 negative correlations for each target:")
    for target in model.target_vars:
        correlations = correlation_df[target].dropna().sort_values()
        print(f"\n{target}:")
        for feature, corr in correlations.head(3).items():
            print(f"  {feature}: {corr:.4f}")
    
    # Save full results
    full_results = {
        r['target']: {
            'ate': r['ate'],
            'lower_bound': r['lower_bound'],
            'upper_bound': r['upper_bound'],
            'top_features': r['top_features']
        }
        for r in results_table
    }
    
    with open(os.path.join(results_dir, 'full_results.json'), 'w') as f:
        json.dump(full_results, f, indent=4)
    
    print(f"\nResults have been saved to: {run_dir}")
    print(f"├── plots/")
    print(f"│   ├── feature_importance/")
    print(f"│   ├── tau_distributions/")
    print(f"│   └── shap_values/")
    print(f"├── models/")
    print(f"└── results/")
    print(f"    ├── summary.csv")
    print(f"    ├── feature_tau_correlations.csv")
    print(f"    └── full_results.json")


if __name__ == "__main__":
    main() 