# 🏦 CreditPulse AI: Intelligent Loan Risk Guard

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CreditPulse AI** is a state-of-the-art machine learning platform designed to predict loan default risks with high precision. By leveraging the Kaggle Home Credit dataset and advanced gradient boosting algorithms, this project provides financial institutions with actionable insights and robust predictive capabilities.

---

## 🌟 Key Features

### 🔍 Interactive Data Intelligence
- **End-to-End EDA**: Interactive visualization of borrower demographics and credit history using Plotly.
- **Feature Engineering**: Automated generation of financial ratios (Credit-to-Income, Annuity-to-Income, etc.).
- **Explainable AI (XAI)**: Integrated **SHAP values** to interpret model decisions at a granular level.

### 🤖 Industrial-Grade Modeling
- **Algorithm Suite**: Implementation of XGBoost, LightGBM, Random Forest, and Logistic Regression.
- **Auto-Tuning**: Built-in hyperparameter optimization using Randomized Search CV.
- **Ensemble Excellence**: Advanced stacking and blending techniques for maximum accuracy.

### 🚀 Production Ready
- **FastAPI Backend**: Low-latency REST API for real-time model serving.
- **Dockerized**: Containerized architecture for seamless deployment across any cloud provider.
- **Comprehensive Testing**: Robust unit testing suite with 80%+ code coverage.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python-based dashboard)
- **Backend API**: FastAPI
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM
- **Visualizations**: Plotly, Seaborn, Matplotlib
- **DevOps**: Docker, GitHub Actions, Pydantic (Data Validation)

---

## 🚦 Getting Started

### 1. Dataset Setup
The dataset files are large and must be downloaded from Kaggle:
- Go to: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
- Download `application_train.csv` and `application_test.csv`
- Place them in the root directory.

### 2. Local Installation
```bash
# Clone the repository
git clone https://github.com/your-username/CreditPulse-AI.git

# Install dependencies
pip install -r requirements.txt

# Run the Dashboard
streamlit run app.py
```

### 3. API Usage
```bash
# Start the FastAPI server
uvicorn api:app --reload
```
Access auto-generated documentation at `http://localhost:8000/docs`.

---

## 📊 Performance Benchmarks

| Model | ROC-AUC | Accuracy | Precision |
|-------|---------|----------|-----------|
| **LightGBM (Tuned)** | **0.782** | 0.921 | 0.654 |
| XGBoost | 0.778 | 0.919 | 0.642 |
| Logistic Regression | 0.741 | 0.918 | 0.581 |

---

## 🏗️ Project Structure

- `app.py`: Core Streamlit dashboard logic.
- `api.py`: FastAPI implementation for model serving.
- `preprocessing.py`: Modular data cleaning and feature engineering.
- `utils/`: Structured utilities for logging, config, and metrics.
- `models/`: Model registry and metadata storage.

---

## 👨‍💻 Developer

**Your Name**  
*ML Engineer & Data Scientist*  
[LinkedIn](https://linkedin.com/in/your-profile) | [Portfolio](https://your-portfolio.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
