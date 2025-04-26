import datetime as dt
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.config import (
    create_directories,
    get_data_paths,
    get_preprocessing_config,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


def load_data():
    """Load raw data from file."""
    # Get data paths from config
    data_paths = get_data_paths()
    raw_path = data_paths.get("raw")

    logger.info(f"Loading raw data from {raw_path}")
    data = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(data)} records")

    return data


def handle_missing_values(data):
    """Handle missing values according to config."""
    # Get preprocessing config
    preproc_config = get_preprocessing_config()
    missing_values_config = preproc_config.get("missing_values", {})

    # Drop columns with high percentage of missing values
    drop_threshold = missing_values_config.get("drop_threshold", 0.7)
    logger.info(f"Dropping columns with more than {drop_threshold*100}% missing values")

    # Calculate missing percentage for all columns at once
    missing_percentages = data.isnull().mean()
    cols_to_drop = missing_percentages[missing_percentages > drop_threshold].index.tolist()

    if cols_to_drop:
        logger.info(f"Dropping columns: {cols_to_drop}")
        data = data.drop(columns=cols_to_drop)

    # Impute missing values
    imputation_method = missing_values_config.get("imputation_method", "median")
    logger.info(f"Imputing missing values using {imputation_method} method")

    # Apply imputation based on method
    if imputation_method == "median":
        for col in data.select_dtypes(include=['number']).columns:
            if data[col].isnull().any():
                logger.info(f"Imputing missing values in {col} with median")
                data[col] = data[col].fillna(data[col].median())
    elif imputation_method == "mean":
        for col in data.select_dtypes(include=['number']).columns:
            if data[col].isnull().any():
                logger.info(f"Imputing missing values in {col} with mean")
                data[col] = data[col].fillna(data[col].mean())
    elif imputation_method == "mode":
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].isnull().any():
                logger.info(f"Imputing missing values in {col} with mode")
                data[col] = data[col].fillna(data[col].mode()[0])

    return data


def clean_column_names(data):
    """Clean column names."""
    logger.info("Cleaning column names")
    data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
    return data


def apply_recategorization(data):
    """Apply recategorization according to config."""
    # Get preprocessing config
    preproc_config = get_preprocessing_config()
    recategorization = preproc_config.get("feature_engineering", {}).get("recategorization", {})

    # Apply recategorization for each specified column
    for col, mapping in recategorization.items():
        if col in data.columns:
            logger.info(f"Applying recategorization to {col}")
            data[col] = data[col].map(mapping).fillna(data[col])

    return data


def implement_clustering(data):
    """Implement K-means clustering for customer segmentation."""
    # Get preprocessing config
    preproc_config = get_preprocessing_config()
    clustering_config = preproc_config.get("feature_engineering", {}).get("clustering", {})

    n_clusters = clustering_config.get("n_clusters", 6)
    random_state = clustering_config.get("random_state", 42)

    logger.info(f"Implementing K-means clustering with {n_clusters} clusters")

    # Select relevant features for clustering
    cluster_features = [
        'income', 'kidhome', 'teenhome', 'recency',
        'mntwines', 'mntfruits', 'mntmeatproducts', 'mntfishproducts',
        'mntsweetproducts', 'mntgoldprods',
        'numwebpurchases', 'numcatalogpurchases', 'numstorepurchases', 'numwebvisitsmonth'
    ]

    # Make sure all features exist in the dataframe
    valid_features = [col for col in cluster_features if col in data.columns]

    # Extract clustering data
    clustering_data = data[valid_features].copy()

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Train K-means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Add cluster labels to the data
    data['customer_segment'] = clusters

    return data


def identify_and_transform_skewed_features(data):
    """Identify skewed numerical features and apply log transformation."""
    # Get preprocessing config
    preproc_config = get_preprocessing_config()
    skewness_threshold = preproc_config.get("feature_engineering", {}).get("skewness_threshold", 1.0)

    logger.info(f"Identifying and transforming skewed features (threshold: {skewness_threshold})")

    # Get only numeric columns (exclude binary)
    numeric_cols = [col for col in data.select_dtypes(include=['number']).columns
                   if len(data[col].unique()) > 2]  # Exclude binary columns

    # Calculate skewness for each numeric column
    skewness = data[numeric_cols].skew()

    # Identify highly skewed features (above threshold)
    highly_skewed = skewness[skewness.abs() > skewness_threshold].index.tolist()

    logger.info(f"Found {len(highly_skewed)} highly skewed features")

    # Apply log transformation to highly skewed features
    for feature in highly_skewed:
        # Add small constant to handle zeros
        if (data[feature] >= 0).all():  # Only transform if all values are non-negative
            # For positive skew, use log1p transformation
            if skewness[feature] > 0:
                logger.info(f"Applying log1p transformation to {feature}")
                data[feature] = np.log1p(data[feature])

    return data


def create_engineered_features(data):
    """Create engineered features."""
    logger.info("Creating engineered features")

    # Total Spending
    data['total_spend'] = data['mntwines'] + data['mntfruits'] + data['mntmeatproducts'] + \
                          data['mntfishproducts'] + data['mntsweetproducts'] + data['mntgoldprods']

    # Spending Proportions
    spend_cols = ['mntwines', 'mntfruits', 'mntmeatproducts', 'mntfishproducts', 'mntsweetproducts', 'mntgoldprods']
    for col in spend_cols:
        data[f'{col}_ratio'] = data[col] / (data['total_spend'] + 0.1)

    # Premium Product Affinity
    data['premium_ratio'] = (data['mntwines'] + data['mntgoldprods']) / (data['total_spend'] + 0.1)

    # Age Features
    current_year = dt.datetime.now().year
    data['age'] = current_year - data['year_birth']

    # Age Group
    bins = [0, 35, 50, 65, 100, float('inf')]
    labels = ['<35', '35-50', '51-65', '66-100', '>100']
    data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)

    # Family Features
    data['has_children'] = ((data['kidhome'] + data['teenhome']) > 0).astype(int)
    data['total_children'] = data['kidhome'] + data['teenhome']
    data['family_size'] = data['total_children'] + np.where(
        data['marital_status'].isin(['Married', 'Together']), 2, 1
    )

    # Income per Family Member
    data['income_per_family_member'] = data['income'] / data['family_size']

    # Recency Score
    data['recency_score'] = np.exp(-data['recency'] / 30)

    # Is Active
    data['is_active'] = (data['recency'] < 30).astype(int)

    # Channel Features
    purchase_channels = ['numwebpurchases', 'numcatalogpurchases', 'numstorepurchases']

    # Preferred Channel
    def get_preferred_channel(row):
        channels = ['web', 'catalog', 'store']
        values = [row['numwebpurchases'], row['numcatalogpurchases'], row['numstorepurchases']]
        return channels[np.argmax(values)]

    data['preferred_channel'] = data[purchase_channels].apply(get_preferred_channel, axis=1)

    # Channel Diversity
    data['channel_diversity'] = data[purchase_channels].apply(
        lambda x: sum(x > 0), axis=1
    )

    # Web Engagement Ratio
    data['webvisit_to_order_ratio'] = data['numwebvisitsmonth'] / (data['numwebpurchases'] + 1)

    # Customer Tenure
    data['day'] = pd.to_datetime(data['dt_customer']).dt.day
    data['month'] = pd.to_datetime(data['dt_customer']).dt.month
    data['year'] = pd.to_datetime(data['dt_customer']).dt.year

    # Create date columns for enrollment
    data['enrollments_year'] = data['year']
    data['enrollments_month'] = data['month']

    # Calculate tenure in days
    reference_date = pd.to_datetime(f"{current_year}-{dt.datetime.now().month}-{dt.datetime.now().day}")
    #data['customer_date'] = pd.to_datetime(
    #    dict(year=data['year'], month=data['month'], day=data['day'])
    #)
    data['customer_date'] = pd.to_datetime({
     "year":  data["year"],
     "month": data["month"],
     "day":   data["day"], })
    data['customer_tenure_days'] = (reference_date - data['customer_date']).dt.days
    data['customer_tenure_years'] = data['customer_tenure_days'] / 365.25

    # Value Interactions
    data['income_age_interaction'] = data['income'] * data['age']
    data['spend_recency_interaction'] = data['total_spend'] * data['recency_score']

    # High-Value Customer
    spend_threshold = data['total_spend'].quantile(0.75)
    data['is_high_value'] = (data['total_spend'] > spend_threshold).astype(int)

    # Customer Lifetime Value
    data['clv'] = data['total_spend'] / (data['customer_tenure_years'] + 0.1)

    # One-hot encoding
    data = pd.get_dummies(data, columns=['age_group', 'preferred_channel'], drop_first=False)

    # Clean up temporary columns
    data.drop(['customer_date', 'day', 'month', 'year', 'dt_customer'], axis=1, inplace=True)

    return data


def preprocess_data():
    """Main preprocessing function."""
    # Create necessary directories
    create_directories()

    # Load data
    data = load_data()

    # Drop Id column if it exists
    if 'Id' in data.columns:
        data.drop(['Id'], axis=1, inplace=True)

    # Clean column names
    data = clean_column_names(data)

    # Handle missing values
    data = handle_missing_values(data)

    # Apply recategorization
    data = apply_recategorization(data)

    # Create engineered features
    data = create_engineered_features(data)

    # Transform skewed features
    data = identify_and_transform_skewed_features(data)

    # Implement clustering
    data = implement_clustering(data)

    # Split data
    logger.info("Splitting data into train, validation, and test sets")
    from sklearn.model_selection import train_test_split

    y = data['response']  # Target variable
    X = data.drop('response', axis=1)  # Features

    # First split: train and temp (val+test)
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Second split: val and test from temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=42
    )

    # Combine features and target
    train_df = X_train.copy()
    train_df['target'] = y_train.values

    val_df = X_val.copy()
    val_df['target'] = y_val.values

    test_df = X_test.copy()
    test_df['target'] = y_test.values

    # Save preprocessed data
    data_paths = get_data_paths()
    processed_paths = data_paths.get("processed", {})

    logger.info(f"Saving preprocessed data to {processed_paths}")
    train_df.to_parquet(processed_paths.get("train", "train.parquet"))
    val_df.to_parquet(processed_paths.get("validation", "val.parquet"))
    test_df.to_parquet(processed_paths.get("test", "test.parquet"))

    logger.info("Preprocessing completed successfully")
    return train_df, val_df, test_df


if __name__ == "__main__":
    preprocess_data()
