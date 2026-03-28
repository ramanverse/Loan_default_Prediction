# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │Overview  │  │EDA       │  │Training │  │Predict  │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Processing Pipeline                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Load Data │→ │Preprocess│→ │Feature   │                  │
│  │          │  │          │  │Engineering│                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Training & Evaluation                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Train/Test│  │Hyperparam│  │Evaluate  │                  │
│  │Split     │  │Tuning    │  │Metrics   │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Registry                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Save Model│  │Metadata  │  │Version   │                  │
│  │(.pkl)    │  │(.json)   │  │Control   │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │/predict  │  │/predict/ │  │/health   │                  │
│  │          │  │batch     │  │          │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Data Loading**: CSV files loaded with optional sampling for large datasets
2. **Preprocessing**: Missing value imputation, feature engineering, encoding
3. **Model Training**: Multiple algorithms trained and compared
4. **Prediction**: Real-time and batch prediction capabilities
5. **Explainability**: SHAP values and feature importance analysis

## Key Components

### Preprocessing Pipeline
- Missing value handling (drop >40% missing, median imputation)
- Feature engineering (AGE, EMPLOY_YEARS, ratios)
- Categorical encoding
- Outlier capping (IQR method)
- Standardization

### Model Training
- Train/test split (80/20, stratified)
- Multiple algorithms (Logistic Regression, Decision Tree, KNN, XGBoost, LightGBM)
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Cross-validation for robust evaluation

### Deployment
- Docker containerization for both Streamlit and FastAPI
- Docker Compose for orchestration
- Health checks and monitoring endpoints

## Technology Stack

- **Frontend**: Streamlit
- **Backend API**: FastAPI
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Visualization**: Plotly
- **Containerization**: Docker, Docker Compose
- **Testing**: pytest, pytest-cov
- **Logging**: Python logging module
- **Configuration**: YAML (PyYAML)
- **Validation**: Pydantic
- **CI/CD**: GitHub Actions

## Key Features

### Logging System
- Structured logging with file and console handlers
- Automatic log rotation (daily log files)
- Request/response logging in API
- Performance timing for operations

### Configuration Management
- YAML-based configuration
- Environment-specific settings
- Easy parameter tuning without code changes

### Data Validation
- Pydantic schemas for input validation
- DataFrame validation utilities
- Range and type checking

### Performance Monitoring
- Prediction time tracking
- Model metrics history
- Performance summaries

### Feature Selection
- Multiple selection methods (K-Best, RF Importance)
- Combined importance scoring
- Feature comparison tools
