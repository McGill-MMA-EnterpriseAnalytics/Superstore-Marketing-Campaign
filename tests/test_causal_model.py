import pytest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from src.causal_model import CausalModel


class TestCausalModel:
    """Test class for CausalModel functionality"""
    
    @pytest.fixture
    def setup_causal_model(self):
        """Create a CausalModel instance and sample data for testing"""
        model = CausalModel()
        
        # Create sample test data
        np.random.seed(42)
        n_samples = 200
        
        # Features
        features = {
            'year_birth': np.random.normal(1970, 10, n_samples).astype(int),
            'income': np.random.normal(50000, 15000, n_samples),
            'kidhome': np.random.choice([0, 1, 2], n_samples),
            'teenhome': np.random.choice([0, 1, 2], n_samples),
            'recency': np.random.randint(1, 100, n_samples),
            'numwebpurchases': np.random.randint(0, 10, n_samples),
            'numcatalogpurchases': np.random.randint(0, 10, n_samples),
            'numstorepurchases': np.random.randint(0, 10, n_samples),
            'numwebvisitsmonth': np.random.randint(0, 20, n_samples),
            'complain': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'customer_segment': np.random.choice([1, 2, 3, 4], n_samples),
            'enrollments_year': np.random.randint(2010, 2023, n_samples),
            'enrollments_month': np.random.randint(1, 13, n_samples),
            
            # Dummy variables for education
            'education_Graduate': np.random.choice([0, 1], n_samples),
            'education_Post Graduate': np.random.choice([0, 1], n_samples),
            'education_Pre Graduate': np.random.choice([0, 1], n_samples),
            
            # Dummy variables for marital status
            'marital_status_Divorced': np.random.choice([0, 1], n_samples),
            'marital_status_Married': np.random.choice([0, 1], n_samples),
            'marital_status_Single': np.random.choice([0, 1], n_samples),
            'marital_status_Together': np.random.choice([0, 1], n_samples),
            'marital_status_Widow': np.random.choice([0, 1], n_samples),
            
            # Treatment indicator
            'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            
            # Target variables
            'mntwines': np.random.exponential(300, n_samples),
            'mntfruits': np.random.exponential(50, n_samples),
            'mntmeatproducts': np.random.exponential(150, n_samples),
            'mntfishproducts': np.random.exponential(40, n_samples),
            'mntsweetproducts': np.random.exponential(30, n_samples),
            'mntgoldprods': np.random.exponential(20, n_samples),
        }
        
        # Create DataFrame
        df = pd.DataFrame(features)
        
        # Add treatment effect to target variables 
        # Treatment has positive effect on some products, negative on others
        effect_sizes = {
            'mntwines': -10,
            'mntfruits': -30,
            'mntmeatproducts': 50,
            'mntfishproducts': -100,
            'mntsweetproducts': -20,
            'mntgoldprods': 40
        }
        
        # Apply effects
        for target, effect in effect_sizes.items():
            df.loc[df['target'] == 1, target] += effect
            # Ensure no negative values
            df.loc[df[target] < 0, target] = 0
            
        return model, df
    
    @pytest.fixture
    def cleanup_output_dirs(self):
        """Clean up test output directories after tests"""
        yield
        if os.path.exists("causal_modeling"):
            shutil.rmtree("causal_modeling")
    
    def test_initialization(self):
        """Test that the model initializes correctly with expected attributes"""
        model = CausalModel()
        
        # Check attributes
        assert model.propensity_model is None
        assert model.r_learner is None
        assert isinstance(model.results, list)
        assert model.feature_names is None
        
        # Check target variables
        assert len(model.target_vars) == 6
        assert 'mntwines' in model.target_vars
        assert 'mntfruits' in model.target_vars
        
        # Check control variables
        assert len(model.controls) > 0
        assert 'income' in model.controls
        assert 'year_birth' in model.controls
    
    def test_fit_propensity_score(self, setup_causal_model):
        """Test propensity score calculation"""
        model, df = setup_causal_model
        
        # Prepare data
        X = df[model.controls]
        T = df['target']
        
        # Fit propensity model
        propensity_scores = model.fit_propensity_score(X, T)
        
        # Check output
        assert propensity_scores.shape == (len(df),)
        assert np.all(propensity_scores >= 0)
        assert np.all(propensity_scores <= 1)
        assert model.propensity_model is not None
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance(self, mock_savefig, setup_causal_model, monkeypatch):
        """Test feature importance plotting"""
        model, df = setup_causal_model
        
        # Create mock r_learner
        mock_r_learner = MagicMock()
        model.r_learner = mock_r_learner
        
        # Mock plot_importance method
        def mock_plot_importance(*args, **kwargs):
            # Just create a simple bar plot as a replacement
            fig = matplotlib.pyplot.figure()
            ax = fig.add_subplot(111)
            ax.bar(['feat1', 'feat2', 'feat3'], [0.5, 0.3, 0.2])
            return fig
            
        monkeypatch.setattr(model.r_learner, "plot_importance", mock_plot_importance)
        
        # Create test data
        X = df[model.controls]
        tau = np.random.normal(0, 1, len(df))
        
        # Call the method
        fig = model.plot_feature_importance(X, tau, "test_target", method="auto")
        
        # Check output
        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)
    
    def test_get_top_features(self, setup_causal_model, monkeypatch):
        """Test get_top_features method"""
        model, df = setup_causal_model
        
        # Create mock r_learner
        mock_r_learner = MagicMock()
        model.r_learner = mock_r_learner
        
        # Mock get_importance method to return test importance scores
        test_importance = {
            'income': 0.25,
            'year_birth': 0.15,
            'kidhome': 0.10,
            'teenhome': 0.05,
            'numwebpurchases': 0.45
        }
        
        monkeypatch.setattr(model.r_learner, "get_importance", lambda **kwargs: test_importance)
        
        # Create test data
        X = df[model.controls]
        tau = np.random.normal(0, 1, len(df))
        
        # Call the method
        top_features = model.get_top_features(X, tau, n_features=3)
        
        # Check output
        assert isinstance(top_features, dict)
        assert len(top_features) == 3
        assert 'numwebpurchases' in top_features
        assert 'income' in top_features
        assert 'year_birth' in top_features
        assert 'teenhome' not in top_features  # Should be excluded as it's not in top 3
    
    def test_calculate_feature_tau_correlations(self, setup_causal_model):
        """Test the calculation of feature-tau correlations"""
        model, df = setup_causal_model
        
        # Create test data with known correlations
        n = 100
        X = pd.DataFrame({
            'feature1': np.linspace(0, 10, n),  # Strong positive correlation
            'feature2': np.random.normal(5, 2, n),  # Random (low correlation)
            'feature3': 10 - np.linspace(0, 10, n)  # Strong negative correlation
        })
        
        # Create tau with strong correlation to feature1, negative to feature3, none to feature2
        tau = X['feature1'] + np.random.normal(0, 0.1, n) - X['feature3']
        
        # Calculate correlations
        correlations = model.calculate_feature_tau_correlations(X, tau)
        
        # Check output
        assert isinstance(correlations, pd.Series)
        assert len(correlations) == 3
        assert correlations['feature1'] > 0.9  # Strong positive correlation
        assert correlations['feature3'] < -0.9  # Strong negative correlation
        assert abs(correlations['feature2']) < abs(correlations['feature1'])  # Lower correlation than feature1
    
    @patch('src.causal_model.BaseRRegressor')
    def test_estimate_ate(self, mock_BaseRRegressor, setup_causal_model):
        """Test Average Treatment Effect estimation"""
        model, df = setup_causal_model
        
        # Setup mock for BaseRRegressor
        mock_r_learner = MagicMock()
        mock_BaseRRegressor.return_value = mock_r_learner
        
        # Set return values for estimate_ate
        mock_r_learner.estimate_ate.return_value = (np.array([0.5]), np.array([0.3]), np.array([0.7]))
        
        # Set return values for fit_predict
        mock_r_learner.fit_predict.return_value = np.random.normal(0.5, 0.2, len(df))
        
        # Prepare data
        X = df[model.controls]
        T = df['target']
        Y = df['mntwines']
        propensity_score = np.random.uniform(0.2, 0.8, len(df))
        
        # Call the method
        te, lb, ub, tau = model.estimate_ate(X, T, Y, propensity_score)
        
        # Check output
        assert te == 0.5
        assert lb == 0.3
        assert ub == 0.7
        assert len(tau) == len(df)
        assert mock_r_learner.estimate_ate.called
        assert mock_r_learner.fit_predict.called
    
    @patch('src.causal_model.load_config')
    @patch('src.causal_model.get_data_paths')
    @patch('pandas.read_parquet')
    @patch('src.causal_model.os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('src.causal_model.joblib.dump')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.figure.Figure.savefig')
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_main_function(self, mock_json_dump, mock_open, mock_fig_savefig, 
                          mock_plt_savefig, mock_dump, mock_to_csv, mock_makedirs, 
                          mock_read_parquet, mock_get_data_paths, mock_load_config,
                          setup_causal_model, cleanup_output_dirs, monkeypatch):
        """Test the main function that runs the full pipeline"""
        # Setup mocks
        model, df = setup_causal_model
        
        # Add education and marital_status columns that load_data expects
        df['education'] = np.random.choice(['Graduate', 'Post Graduate', 'Pre Graduate'], len(df))
        df['marital_status'] = np.random.choice(['Married', 'Single', 'Together', 'Divorced', 'Widow'], len(df))
        
        # Mock config loading
        mock_load_config.return_value = {'data_paths': {}}
        mock_get_data_paths.return_value = {
            'train': 'mock_train.parquet',
            'test': 'mock_test.parquet',
            'val': 'mock_val.parquet'
        }
        
        # Make sure os.makedirs doesn't raise an exception
        mock_makedirs.return_value = None
        
        # Make sure DataFrame.to_csv doesn't try to write to file
        mock_to_csv.return_value = None
        
        # Make figure.savefig not try to write to file
        mock_fig_savefig.return_value = None
        mock_plt_savefig.return_value = None
        
        # Mock parquet reading to return our test dataframe
        mock_read_parquet.return_value = df
        
        # Mock the CausalModel class methods
        import src.causal_model
        
        # Mock load_data to avoid dealing with dummy variables
        original_load_data = src.causal_model.CausalModel.load_data
        def patched_load_data(self, *args, **kwargs):
            return df
        
        # Create a simple mock for plot methods
        def mock_plot_func(*args, **kwargs):
            fig = matplotlib.pyplot.figure()
            ax = fig.add_subplot(111)
            ax.plot([1, 2, 3], [1, 2, 3])
            return fig
        
        # Mock get_top_features to return predictable results
        original_get_top_features = src.causal_model.CausalModel.get_top_features
        def patched_get_top_features(self, X, tau, n_features=10):
            # Return fixed values for testing
            return {
                'feature1': 0.5,
                'feature2': 0.3,
                'feature3': 0.2
            }
        
        # Patch the estimate_ate method to return predictable values
        original_estimate_ate = src.causal_model.CausalModel.estimate_ate
        def patched_estimate_ate(self, X, T, Y, propensity_score):
            # Return fixed values for testing
            tau = np.random.normal(0.5, 0.2, len(X))
            self.r_learner = MagicMock()  # Ensure r_learner is not None
            return 0.5, 0.3, 0.7, tau
        
        # Apply the monkey patches for testing
        monkeypatch.setattr(src.causal_model.CausalModel, 'load_data', patched_load_data)
        monkeypatch.setattr(src.causal_model.CausalModel, 'estimate_ate', patched_estimate_ate)
        monkeypatch.setattr(src.causal_model.CausalModel, 'get_top_features', patched_get_top_features)
        monkeypatch.setattr(src.causal_model.CausalModel, 'plot_feature_importance', mock_plot_func)
        monkeypatch.setattr(src.causal_model.CausalModel, 'plot_tau_distribution', mock_plot_func)
        monkeypatch.setattr(src.causal_model.CausalModel, 'plot_shap_values', mock_plot_func)
        
        # Mock functions that use file system
        mock_open.return_value.__enter__.return_value = MagicMock()
        mock_json_dump.return_value = None
        
        # Call the main function
        from src.causal_model import main
        main()
        
        # Check directory creation
        assert mock_makedirs.called
        
        # Check file saving - check either plt.savefig or fig.savefig was called
        assert mock_fig_savefig.called or mock_plt_savefig.called
        assert mock_dump.called
        assert mock_to_csv.called
        
        # Restore original methods
        monkeypatch.setattr(src.causal_model.CausalModel, 'estimate_ate', original_estimate_ate)
        monkeypatch.setattr(src.causal_model.CausalModel, 'load_data', original_load_data)
        monkeypatch.setattr(src.causal_model.CausalModel, 'get_top_features', original_get_top_features)


if __name__ == "__main__":
    pytest.main(["-xvs", "test_causal_model.py"]) 