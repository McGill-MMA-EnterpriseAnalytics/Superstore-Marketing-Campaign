# Superstore Marketing Campaign Analysis

## Project Overview

The **Superstore Marketing Campaign Analysis** is a comprehensive machine learning solution designed to evaluate customer behavior, analyze spending patterns, and predict responses to marketing campaigns. This project leverages data science and MLOps best practices to create a robust, scalable system for marketing optimization.

### Objectives

- **Customer Segmentation**: Identify distinct customer groups based on purchasing behaviors
- **Response Prediction**: Build models to determine which customers are likely to respond to campaigns 
- **Marketing Optimization**: Provide data-driven strategies to enhance engagement and sales
- **MLOps Implementation**: Establish a production-ready ML pipeline with monitoring capabilities

### Dataset

The dataset consists of 2,240 customer records with information on:
- Demographics (age, marital status, education)
- Spending patterns across product categories
- Campaign response history
- Digital engagement metrics

## Architecture

The project follows a modular architecture with several key components:

### 1. Data Engineering & Preprocessing

- **Implementation**: `src/preprocess.py`
- **Configuration**: `config.yaml` (preprocessing section)
- **Features**:
  - Missing value handling with configurable thresholds
  - Feature engineering (spending ratios, age features, etc.)
  - K-means clustering for customer segmentation
  - Categorical variable recategorization
  - Skewness detection and transformation

### 2. Model Training & Evaluation

- **Implementation**: `src/train_model.py`
- **Configuration**: `config.yaml` (models section)
- **Models**:
  - XGBoost Classifier
  - LightGBM Classifier
  - CatBoost Classifier
  - Random Forest Classifier
  - Stacking Ensemble
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC

### 3. Hyperparameter Optimization

- **Implementation**: `src/hyperparameter_tuning.py`
- **Framework**: Optuna
- **Features**:
  - Bayesian optimization
  - Model-specific parameter spaces
  - F1 score optimization
  - SQLite storage for persisting optimization runs

### 4. Experiment Tracking

- **Implementation**: Integrated in training modules
- **Framework**: MLflow
- **Features**:
  - Metric logging
  - Parameter tracking
  - Model versioning
  - Artifact storage

### 5. Model Serving

- **Implementation**: `src/app.py`
- **Framework**: FastAPI
- **Endpoints**:
  - `/predict` for single predictions
  - `/predict-batch` for multiple predictions
  - `/health` for service health checks

### 6. Model Monitoring

- **Implementation**: `src/monitoring/`
- **Features**:
  - Feature drift detection (Wasserstein distance)
  - Target drift detection
  - Prediction drift detection
  - Concept drift detection (performance degradation)
  - Scheduled monitoring with configurable thresholds

### 7. Containerization

- **Implementation**: `dockerfile` and `dockerfile.api`
- **Features**:
  - Separate containers for training and serving
  - Reproducible environments

## Directory Structure

```
├── artifacts/              # Stored model artifacts and datasets
├── src/                    # Source code
│   ├── monitoring/         # Model monitoring components
│   ├── utils/              # Utility functions
│   ├── app.py              # FastAPI application
│   ├── preprocess.py       # Data preprocessing pipeline
│   ├── train_model.py      # Model training pipeline
│   └── hyperparameter_tuning.py  # Hyperparameter optimization
├── tests/                  # Test suite
├── config.yaml             # Main configuration
├── config_monitoring.yaml  # Monitoring configuration
├── dockerfile              # Docker configuration for training
├── dockerfile.api          # Docker configuration for API
├── pyproject.toml          # Poetry dependency management
├── poetry.lock             # Poetry lock file
├── .pre-commit-config.yaml # Pre-commit hooks configuration
└── README.md               # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Docker
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/superstore-marketing-campaign.git
   cd superstore-marketing-campaign
   ```

2. **Set up the environment with Poetry**
   ```bash
   # Install Poetry if you don't have it
   pip install poetry
   
   # Create virtual environment and install dependencies
   poetry install
   ```

3. **Alternative: Setup with pip**
   ```bash
   # Create a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -e .
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Docker Setup

1. **Build the training container**
   ```bash
   docker build -t superstore-training -f dockerfile .
   ```

2. **Build the API container**
   ```bash
   docker build -t superstore-api -f dockerfile.api .
   ```

## Usage Instructions

### Data Preprocessing

```bash
# Run the preprocessing pipeline
poetry run python -m src.preprocess
```

### Model Training

```bash
# Train the model with default configuration
poetry run python -m src.train_model

# Train a specific model
poetry run python -m src.train_model --model xgboost
```

### Hyperparameter Tuning

```bash
# Run hyperparameter optimization
poetry run python -m src.hyperparameter_tuning --model xgboost --trials 50
```

### Running the API

```bash
# Start the FastAPI server
poetry run uvicorn src.app:app --reload

# With Docker
docker run -p 8000:8000 superstore-api
```

### Model Monitoring

```bash
# Run drift detection
poetry run python -m src.monitoring.drift_detector

# Set up scheduled monitoring
poetry run bash setup_monitoring.sh
```

## Deployment Guide

### Current Deployment

The project currently supports local deployment via Docker containers:

1. **Training Pipeline Deployment**
   ```bash
   docker run --rm -v $(pwd)/artifacts:/app/artifacts superstore-training
   ```

2. **API Deployment**
   ```bash
   docker run -d -p 8000:8000 --name superstore-api superstore-api
   ```

3. **Testing the API**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {...}}'
   ```

### Future AWS Deployment

The project is designed for future deployment to AWS with the following components:

1. **MLflow: Integration & Versioning**:
   - Remote Tracking Server on AWS (containerized on Fargate)
   - Artifact Store in Amazon S3
   - Backend Store in Amazon RDS
   - Environment-Driven Configuration:
     - MLFLOW_TRACKING_URI, S3 bucket name, and DB credentials from environment variables
   - Local MLflow tracking with experiment lifecycle management

2. **Model Training**:
   - AWS Batch for scalable training jobs
   - S3 for artifact storage
   - ECR for container registry

3. **Model Serving**:
   - AWS Lambda functions for serverless prediction
   - API Gateway for REST API access
   - Elastic Container Service (ECS) for containerized model serving

4. **Monitoring**:
   - CloudWatch for logs and metrics
   - Custom monitoring using the existing drift detection implemented in `src/monitoring/`
   - EventBridge for scheduling monitoring tasks

5. **CI/CD Pipeline**:
   - CodePipeline for automated deployments
   - CodeBuild for container building
   - GitHub Actions for testing and quality checks

## Contribution Guidelines

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and run tests**
   ```bash
   # Run tests
   pytest tests/
   
   # Run linter
   pre-commit run --all-files
   ```

3. **Commit your changes with meaningful messages**
   ```bash
   git commit -m "Add feature: description of the change"
   ```

4. **Push your branch and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Coding Standards

- Follow PEP 8 naming conventions
- Add docstrings to all functions and classes
- Ensure code passes linting with pre-commit hooks
- Write unit tests for new functionality

### Making Code Changes

1. **Configuration Changes**:
   - Update `config.yaml` for model parameters, preprocessing settings, etc.
   - Update `config_monitoring.yaml` for monitoring thresholds

2. **Adding New Models**:
   - Add model implementation in `src/train_model.py`
   - Define hyperparameter search space in `src/hyperparameter_tuning.py`
   - Update `config.yaml` with model configuration

3. **Modifying Preprocessing**:
   - Update transformation logic in `src/preprocess.py`
   - Add new feature engineering functions as needed

4. **API Changes**:
   - Modify endpoint logic in `src/app.py`
   - Update input/output schemas with Pydantic models

5. **Monitoring Changes**:
   - Enhance drift detection in `src/monitoring/drift_detector.py`
   - Add new monitoring metrics as needed
