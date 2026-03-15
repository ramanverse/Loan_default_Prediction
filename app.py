import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, confusion_matrix,
                            classification_report, precision_recall_curve)
import pickle
import json
from datetime import datetime
import os
import warnings
import time
import sys
import os
from pathlib import Path

# Add current directory to Python path (for Streamlit Cloud compatibility)
try:
    # Try to get the directory containing this file
    current_dir = Path(__file__).parent.absolute()
except NameError:
    # If __file__ is not available (some environments), use current working directory
    current_dir = Path(os.getcwd())

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

warnings.filterwarnings('ignore')

# Initialize logger first (always available)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit_app")

# Import utilities (with fallbacks if not available)
try:
    from utils.logger import setup_logger
    logger = setup_logger("streamlit_app")
except (ImportError, Exception) as e:
    logger.warning(f"Could not import utils.logger, using basic logging: {e}")

try:
    from utils.config import config
except (ImportError, Exception) as e:
    config = None
    logger.warning(f"Could not import utils.config: {e}")

try:
    from utils.metrics import calculate_comprehensive_metrics, PerformanceMonitor
    monitor = PerformanceMonitor()
except (ImportError, Exception) as e:
    # Fallback PerformanceMonitor
    class PerformanceMonitor:
        def __init__(self):
            self.metrics_history = []
            self.prediction_times = []
        def record_prediction_time(self, elapsed_time):
            self.prediction_times.append(elapsed_time)
        def get_avg_prediction_time(self):
            return sum(self.prediction_times) / len(self.prediction_times) if self.prediction_times else 0.0
        def record_model_metrics(self, metrics):
            self.metrics_history.append(metrics)
        def get_latest_metrics(self):
            return self.metrics_history[-1] if self.metrics_history else None
    
    def calculate_comprehensive_metrics(y_true, y_pred, y_proba=None):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        return metrics
    
    monitor = PerformanceMonitor()
    logger.warning(f"Could not import utils.metrics, using fallback: {e}")

# Try to import optional advanced libraries
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
SHAP_AVAILABLE = False
IMBLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except (ImportError, RuntimeError):
    SHAP_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    pass

# Page configuration - MUST be first Streamlit command
try:
    st.set_page_config(
        page_title="CreditPulse AI | Loan Default Guard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    # If set_page_config fails (e.g., already called), continue
    pass

# Enhanced styling for better UI/UX
st.markdown("""
    <style>
    h1 {
        text-align: center;
        color: #4F46E5;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    h2 {
        color: #374151;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .stButton > button {
        border-radius: 6px;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stSidebar"] {
        background-color: #fafafa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

def create_demo_data(n_samples=5000, random_state=42):
    """Create synthetic demo dataset"""
    np.random.seed(random_state)
    sample_data = {
        'SK_ID_CURR': range(n_samples),
        'TARGET': np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08]),
        'AMT_INCOME_TOTAL': np.random.lognormal(11, 0.5, n_samples),
        'AMT_CREDIT': np.random.lognormal(12, 0.5, n_samples),
        'AMT_ANNUITY': np.random.lognormal(9, 0.5, n_samples),
        'AMT_GOODS_PRICE': np.random.lognormal(11.5, 0.5, n_samples),
        'CNT_CHILDREN': np.random.poisson(0.5, n_samples),
        'CNT_FAM_MEMBERS': np.random.poisson(2, n_samples),
        'EXT_SOURCE_1': np.random.beta(2, 5, n_samples),
        'EXT_SOURCE_2': np.random.beta(2, 5, n_samples),
        'EXT_SOURCE_3': np.random.beta(2, 5, n_samples),
        'DAYS_BIRTH': -np.random.randint(8000, 20000, n_samples),
        'DAYS_EMPLOYED': -np.random.randint(0, 5000, n_samples),
        'DAYS_REGISTRATION': -np.random.randint(1000, 5000, n_samples),
        'DAYS_ID_PUBLISH': -np.random.randint(500, 3000, n_samples),
        'CODE_GENDER': np.random.choice(['M', 'F'], n_samples),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_samples),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n_samples),
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], n_samples),
        'NAME_EDUCATION_TYPE': np.random.choice(['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary'], n_samples),
        'NAME_FAMILY_STATUS': np.random.choice(['Married', 'Single / not married', 'Civil marriage', 'Separated'], n_samples),
        'NAME_HOUSING_TYPE': np.random.choice(['House / apartment', 'Rented apartment', 'With parents'], n_samples),
        'OCCUPATION_TYPE': np.random.choice(['Laborers', 'Core staff', 'Sales staff', 'Managers'], n_samples),
        'ORGANIZATION_TYPE': np.random.choice(['Business Entity Type 3', 'Self-employed', 'Other'], n_samples),
    }
    return pd.DataFrame(sample_data)

@st.cache_data
def load_data(sample_size=None, random_state=42, use_demo=False):
    """Load and cache the training data with optional sampling"""
    # If demo mode is enabled, return demo data
    if use_demo:
        n_samples = sample_size if sample_size else 5000
        df = create_demo_data(n_samples=n_samples, random_state=random_state)
        return df, len(df), False
    
    try:
        import os
        
        # Check file size
        file_path = 'application_train.csv'
        
        # Check if file exists
        if not os.path.exists(file_path):
            # Try loading from Streamlit secrets (for cloud deployment)
            dataset_url = None
            
            # Try multiple methods to access secrets (Streamlit Cloud compatibility)
            try:
                if hasattr(st, 'secrets'):
                    # Method 1: Direct access (most common)
                    if hasattr(st.secrets, 'get'):
                        dataset_url = st.secrets.get('dataset_url')
                    # Method 2: Dictionary-style access
                    elif hasattr(st.secrets, '__getitem__'):
                        try:
                            dataset_url = st.secrets['dataset_url']
                        except (KeyError, AttributeError):
                            pass
                    # Method 3: Nested secrets structure
                    if not dataset_url and hasattr(st.secrets, 'secrets'):
                        try:
                            dataset_url = st.secrets.secrets.get('dataset_url')
                        except:
                            pass
            except Exception:
                pass
            
            if dataset_url:
                try:
                    # Show loading message
                    loading_placeholder = st.empty()
                    loading_placeholder.info(f"Loading dataset from cloud storage...")
                    
                    # Try loading with timeout
                    import urllib.request
                    import io
                    
                    # For Google Drive, we might need to handle the download differently
                    if 'drive.google.com' in dataset_url:
                        # Extract file ID
                        file_id = None
                        if '/file/d/' in dataset_url:
                            file_id = dataset_url.split('/file/d/')[1].split('/')[0]
                        elif 'id=' in dataset_url:
                            file_id = dataset_url.split('id=')[1].split('&')[0].split('/')[0]
                        
                        if file_id:
                            # For large files, Google Drive requires confirmation
                            # Use the confirmed download URL
                            dataset_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                    
                    # Try to load the dataset
                    # Google Drive large files (>100MB) return HTML virus scan warning
                    # We need to handle this specially
                    import urllib.request
                    import io
                    import http.cookiejar
                    
                    try:
                        # First, try to read a small sample to check if it's CSV or HTML
                        req = urllib.request.Request(dataset_url)
                        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                        response = urllib.request.urlopen(req, timeout=30)
                        content_sample = response.read(5000)  # Read first 5KB
                        content_str = content_sample.decode('utf-8', errors='ignore')
                        
                        # Check if response is HTML (virus scan warning page)
                        is_html = (content_str.strip().startswith('<!DOCTYPE') or 
                                  '<html>' in content_str.lower() or
                                  'virus scan' in content_str.lower() or
                                  'Google Drive' in content_str)
                        
                        if is_html and file_id:
                            # This is the virus scan warning page for large files
                            # Use the confirmed download URL
                            loading_placeholder.warning(
                                "Large file detected. Google Drive requires confirmation for files >100MB. "
                                "Attempting to download with confirmation..."
                            )
                            
                            # Use the confirmed download endpoint
                            confirmed_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t&uuid="
                            
                            # Create cookie jar for session handling
                            cj = http.cookiejar.CookieJar()
                            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
                            
                            # Make request with cookies
                            req2 = urllib.request.Request(confirmed_url)
                            req2.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                            
                            # Try to download
                            try:
                                response2 = opener.open(req2, timeout=120)  # Longer timeout for large files
                                df = pd.read_csv(io.BytesIO(response2.read()))
                            except Exception as e2:
                                # If that fails, try alternative method
                                alt_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                                req3 = urllib.request.Request(alt_url)
                                req3.add_header('User-Agent', 'Mozilla/5.0')
                                response3 = opener.open(req3, timeout=120)
                                df = pd.read_csv(io.BytesIO(response3.read()))
                        else:
                            # It's CSV data, read it normally
                            # Close the sample response and read full file
                            response.close()
                            req_full = urllib.request.Request(dataset_url)
                            req_full.add_header('User-Agent', 'Mozilla/5.0')
                            response_full = urllib.request.urlopen(req_full, timeout=120)
                            df = pd.read_csv(io.BytesIO(response_full.read()))
                            
                    except Exception as e:
                        # Fallback: try direct read_csv (works for smaller files)
                        try:
                            df = pd.read_csv(dataset_url, timeout=120)
                        except Exception as e2:
                            error_msg = str(e2)
                            if 'HTML' in error_msg or '<!DOCTYPE' in error_msg:
                                raise Exception(
                                    "Google Drive is blocking direct download of large files (>100MB).\n\n"
                                    "**Solutions:**\n"
                                    "1. **Use Dropbox** (recommended): Upload to Dropbox and use a direct download link\n"
                                    "2. **Use AWS S3**: Upload to S3 and use a public URL\n"
                                    "3. **Use GitHub Releases**: Upload as a release asset\n"
                                    "4. **Split the file**: Compress and split into smaller chunks\n\n"
                                    "For Dropbox: Get a direct download link (dl.dropboxusercontent.com)"
                                )
                            else:
                                raise Exception(f"Failed to load dataset: {str(e2)}")
                    
                    # Verify TARGET column exists
                    if 'TARGET' not in df.columns:
                        loading_placeholder.error(
                            f"⚠️ Loaded dataset is missing TARGET column!\n\n"
                            f"**File loaded:** {len(df):,} rows, {len(df.columns)} columns\n"
                            f"**Columns found:** {', '.join(df.columns[:5])}...\n\n"
                            f"This appears to be `application_test.csv` instead of `application_train.csv`.\n"
                            f"Please upload the correct training dataset with TARGET column."
                        )
                        return None, 0, False
                    
                    loading_placeholder.success(f"Successfully loaded {len(df):,} rows from cloud storage!")
                    return df, len(df), False
                except Exception as e:
                    # Error will be handled in main function
                    return None, 0, False
            
            # File not found - return None (will be handled in main)
            return None, 0, False
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        # If sample_size is specified, use efficient loading
        if sample_size:
            # For large files, use chunking approach
            if file_size_mb > 100:  # Very large file
                # Read first chunk to estimate row size
                first_chunk = pd.read_csv(file_path, nrows=1000)
                bytes_per_row = file_size_mb * 1024 * 1024 / (len(first_chunk) * 10)  # Rough estimate
                total_rows_est = int((file_size_mb * 1024 * 1024) / bytes_per_row) if bytes_per_row > 0 else sample_size * 10
                
                # Use chunking to sample
                chunk_list = []
                rng = np.random.RandomState(random_state)
                target_size = sample_size
                
                for chunk in pd.read_csv(file_path, chunksize=50000):
                    if sum(len(c) for c in chunk_list) >= target_size:
                        break
                    
                    # Sample from chunk
                    remaining = target_size - sum(len(c) for c in chunk_list)
                    if remaining > 0 and len(chunk) > 0:
                        sample_n = min(remaining, len(chunk))
                        if len(chunk) > sample_n:
                            sampled = chunk.sample(n=sample_n, random_state=rng)
                        else:
                            sampled = chunk
                        chunk_list.append(sampled)
                
                if chunk_list:
                    df = pd.concat(chunk_list, ignore_index=True)
                    if len(df) > target_size:
                        df = df.sample(n=target_size, random_state=random_state).reset_index(drop=True)
                    return df, total_rows_est, True
            
            # For smaller files, load and sample
            df = pd.read_csv(file_path)
            total_rows = len(df)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            return df, total_rows, len(df) < total_rows
        else:
            # Load full dataset
            df = pd.read_csv(file_path)
            return df, len(df), False
    except FileNotFoundError:
        st.error("Please ensure 'application_train.csv' is in the same directory as this app.")
        return None, 0, False
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, 0, False

@st.cache_data
def preprocess_data(df):
    """Preprocess the data"""
    df_clean = df.copy()
    
    # Drop columns with >40% missing values
    missing = df_clean.isnull().mean()
    cols_to_drop = missing[missing > 0.40].index.tolist()
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Handle anomalous DAYS_EMPLOYED
    if 'DAYS_EMPLOYED' in df_clean.columns:
        df_clean['DAYS_EMPLOYED'] = df_clean['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # Feature Engineering
    if 'DAYS_BIRTH' in df_clean.columns:
        df_clean['AGE'] = -df_clean['DAYS_BIRTH'] / 365.0
    
    if 'DAYS_EMPLOYED' in df_clean.columns:
        df_clean['EMPLOY_YEARS'] = -(df_clean['DAYS_EMPLOYED'] / 365.0)
    
    if {'AMT_CREDIT', 'AMT_INCOME_TOTAL'}.issubset(df_clean.columns):
        df_clean['CREDIT_INCOME_RATIO'] = df_clean['AMT_CREDIT'] / df_clean['AMT_INCOME_TOTAL'].replace({0: np.nan})
    
    if {'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(df_clean.columns):
        df_clean['ANNUITY_INCOME_RATIO'] = df_clean['AMT_ANNUITY'] / df_clean['AMT_INCOME_TOTAL'].replace({0: np.nan})
    
    exts = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df_clean.columns]
    if exts:
        df_clean['EXT_SOURCES_MEAN'] = df_clean[exts].mean(axis=1)
        df_clean['EXT_SOURCES_STD'] = df_clean[exts].std(axis=1)
        df_clean['EXT_SOURCES_MAX'] = df_clean[exts].max(axis=1)
        df_clean['EXT_SOURCES_MIN'] = df_clean[exts].min(axis=1)
    
    # Advanced Feature Engineering
    # Polynomial features for key ratios
    if 'CREDIT_INCOME_RATIO' in df_clean.columns:
        df_clean['CREDIT_INCOME_RATIO_SQ'] = df_clean['CREDIT_INCOME_RATIO'] ** 2
        df_clean['CREDIT_INCOME_RATIO_SQRT'] = np.sqrt(df_clean['CREDIT_INCOME_RATIO'].clip(lower=0))
    
    if 'ANNUITY_INCOME_RATIO' in df_clean.columns:
        df_clean['ANNUITY_INCOME_RATIO_SQ'] = df_clean['ANNUITY_INCOME_RATIO'] ** 2
    
    # Interaction features
    if {'AGE', 'EMPLOY_YEARS'}.issubset(df_clean.columns):
        df_clean['AGE_EMPLOY_INTERACTION'] = df_clean['AGE'] * df_clean['EMPLOY_YEARS']
        df_clean['AGE_EMPLOY_RATIO'] = df_clean['AGE'] / (df_clean['EMPLOY_YEARS'].replace({0: np.nan}) + 1)
    
    if {'AMT_CREDIT', 'AMT_ANNUITY'}.issubset(df_clean.columns):
        df_clean['CREDIT_ANNUITY_RATIO'] = df_clean['AMT_CREDIT'] / df_clean['AMT_ANNUITY'].replace({0: np.nan})
        df_clean['CREDIT_ANNUITY_DIFF'] = df_clean['AMT_CREDIT'] - df_clean['AMT_ANNUITY']
    
    # Binning features for non-linear relationships
    if 'AGE' in df_clean.columns:
        df_clean['AGE_BINNED'] = pd.cut(df_clean['AGE'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
        df_clean['AGE_BINNED'] = df_clean['AGE_BINNED'].astype(float)
    
    if 'AMT_INCOME_TOTAL' in df_clean.columns:
        income_quantiles = df_clean['AMT_INCOME_TOTAL'].quantile([0, 0.25, 0.5, 0.75, 1.0])
        df_clean['INCOME_BINNED'] = pd.cut(df_clean['AMT_INCOME_TOTAL'], bins=income_quantiles, labels=[0, 1, 2, 3], include_lowest=True)
        df_clean['INCOME_BINNED'] = df_clean['INCOME_BINNED'].astype(float)
    
    # Aggregation features
    if {'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(df_clean.columns):
        df_clean['TOTAL_AMOUNT'] = df_clean['AMT_CREDIT'] + df_clean['AMT_ANNUITY'] + df_clean['AMT_INCOME_TOTAL']
        df_clean['WEIGHTED_CREDIT_SCORE'] = (df_clean['AMT_CREDIT'] * 0.4 + 
                                             df_clean['AMT_INCOME_TOTAL'] * 0.3 + 
                                             df_clean['AMT_ANNUITY'] * 0.3)
    
    # Time-based features
    if 'DAYS_BIRTH' in df_clean.columns:
        df_clean['DAYS_BIRTH_ABS'] = np.abs(df_clean['DAYS_BIRTH'])
        df_clean['BIRTH_YEAR'] = (df_clean['DAYS_BIRTH'] / -365.25).astype(int) + 2024
    
    # Log transformations for skewed features
    for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']:
        if col in df_clean.columns:
            df_clean[f'{col}_LOG'] = np.log1p(df_clean[col].clip(lower=0))
    
    return df_clean

@st.cache_data
def prepare_model_data(df_clean):
    """Prepare data for modeling"""
    df_model = df_clean.copy()
    
    # Impute numeric columns
    num_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ('SK_ID_CURR', 'TARGET')]
    num_imputer = SimpleImputer(strategy='median')
    df_model[num_cols] = num_imputer.fit_transform(df_model[num_cols])
    
    # Encode categorical columns
    cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        df_model[c] = df_model[c].fillna('MISSING')
        df_model[c] = pd.Categorical(df_model[c]).codes
    
    # IQR capping for financial variables
    def cap_iqr(series, k=1.5):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        return series.clip(lower=lower, upper=upper)
    
    cap_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_INCOME_RATIO']
    cap_features = [c for c in cap_features if c in df_model.columns]
    for c in cap_features:
        df_model[c] = cap_iqr(df_model[c], k=1.5)
    
    return df_model

def main():
    st.title("🏦 CreditPulse AI: Intelligent Loan Risk Guard")
    
    # Initialize logger on app start
    logger.info("Streamlit app started")
    
    st.sidebar.title("Navigation")
    
    # Group pages by category for better organization
    pages = {
        "Data": ["Overview", "Data Exploration", "Advanced Analysis"],
        "Modeling": ["Model Training", "Hyperparameter Tuning", "Model Comparison", "Ensemble Methods"],
        "Predictions": ["Predictions", "Batch Predictions"],
        "Analysis": ["Model Explainability", "Threshold Optimization", "Profit Curve Analysis", "Feature Selection"],
        "Management": ["Model Registry", "Insights & Recommendations"]
    }
    
    # Create navigation with radio buttons grouped by category
    page = st.session_state.get('selected_page', "Overview")
    
    for category, page_list in pages.items():
        st.sidebar.markdown(f"**{category}**")
        for page_name in page_list:
            # Highlight current page
            is_selected = (page == page_name)
            button_style = "primary" if is_selected else "secondary"
            
            if st.sidebar.button(
                page_name, 
                key=f"nav_{page_name}", 
                use_container_width=True,
                type=button_style
            ):
                st.session_state['selected_page'] = page_name
                st.rerun()
        
        st.sidebar.markdown("")  # Add spacing between categories
    
    # Portfolio Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 Developer")
    st.sidebar.info("""
    **Developer Name**  
    [GitHub](https://github.com/your-username) | [LinkedIn](https://linkedin.com/in/your-profile)
    
    *ML Engineer / Data Scientist*
    """)
    
    st.sidebar.markdown("### 🛠️ Tech Stack")
    st.sidebar.markdown("""
    - **Backend**: Python (FastAPI)
    - **Frontend**: Streamlit
    - **ML**: Scikit-Learn, XGBoost, LightGBM
    - **Viz**: Plotly, Seaborn
    - **Ops**: Docker, GitHub Actions
    """)
    
    st.sidebar.title("Data Settings")
    
    # Demo mode toggle
    use_demo_mode = st.sidebar.checkbox(
        "Use Demo Mode (Sample Data)", 
        value=st.session_state.get('use_demo_mode', False),
        help="Use synthetic sample data if dataset file is not available"
    )
    st.session_state['use_demo_mode'] = use_demo_mode
    
    pages_needing_full_data = ["Model Training", "Model Comparison", "Predictions", 
                               "Batch Predictions", "Model Explainability", "Threshold Optimization"]
    use_full_dataset = st.sidebar.checkbox(
        "Use Full Dataset (for Model Training)", 
        value=(page in pages_needing_full_data)
    )
    
    if use_full_dataset:
        sample_size = None
    else:
        sample_size = st.sidebar.slider(
            "Sample Size for Exploration", 
            min_value=5000, 
            max_value=100000, 
            value=20000, 
            step=5000
        )
    
    # Load data
    df, total_rows, is_sampled = load_data(sample_size=sample_size, use_demo=use_demo_mode)
    
    # Handle missing dataset
    if df is None and not use_demo_mode:
        # Check if secrets are configured but not working
        secrets_configured = False
        dataset_url = None
        secrets_info = {}
        
        try:
            if hasattr(st, 'secrets'):
                # Try to get the URL
                if hasattr(st.secrets, 'get'):
                    dataset_url = st.secrets.get('dataset_url')
                elif hasattr(st.secrets, '__getitem__'):
                    try:
                        dataset_url = st.secrets['dataset_url']
                    except (KeyError, AttributeError):
                        pass
                
                # Get secrets info for debugging
                try:
                    if hasattr(st.secrets, 'keys'):
                        secrets_info = {
                            "secrets_available": True,
                            "secrets_keys": list(st.secrets.keys()),
                            "dataset_url_found": dataset_url is not None
                        }
                    else:
                        secrets_info = {
                            "secrets_available": True,
                            "secrets_type": str(type(st.secrets)),
                            "dataset_url_found": dataset_url is not None
                        }
                except:
                    secrets_info = {"secrets_available": True, "error": "Could not inspect secrets"}
                
                if dataset_url:
                    secrets_configured = True
        except Exception as e:
            secrets_info = {"secrets_available": False, "error": str(e)}
        
        st.error("""
        **Dataset file not found!**
        
        Please choose one of the following options:
        """)
        
        # Show secrets status if configured
        if secrets_configured:
            st.warning(f"""
            **⚠️ Secrets detected but dataset not loading**
            
            **URL configured:** `{dataset_url[:80] if dataset_url else 'None'}...`
            
            **Possible issues:**
            1. Google Drive file is not set to "Anyone with the link"
            2. URL format might need adjustment
            3. File might be too large for direct download
            4. Network timeout during download
            
            **Next steps:**
            - Check Streamlit Cloud logs (Manage app → Logs)
            - Verify file sharing settings in Google Drive
            - Try the "Test URL" button below
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Test URL Connection", type="secondary"):
                    try:
                        with st.spinner("Testing URL connection..."):
                            # Handle Google Drive URL conversion if needed
                            test_url = dataset_url
                            if 'drive.google.com/file/d/' in test_url:
                                file_id = test_url.split('/file/d/')[1].split('/')[0]
                                test_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                            
                            test_df = pd.read_csv(test_url, nrows=10)
                            st.success(f"✅ **URL is accessible!**\n\nFound {len(test_df)} rows in test sample.\n\nThe full dataset should load. If it still doesn't, the file might be too large.")
                            st.dataframe(test_df.head())
                    except Exception as e:
                        st.error(f"❌ **URL not accessible**\n\nError: `{str(e)}`")
                        st.info("""
                        **Troubleshooting steps:**
                        1. Go to Google Drive → Right-click file → Share → Set to "Anyone with the link"
                        2. Try copying the file ID and using this format in secrets:
                           ```
                           [secrets]
                           dataset_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
                           ```
                        3. Check if the file size is under 1GB (Streamlit Cloud has limits)
                        """)
            
            with col2:
                if st.button("📋 Copy Correct Secrets Format", type="secondary"):
                    file_id = "YOUR_FILE_ID"
                    if dataset_url and '/file/d/' in dataset_url:
                        file_id = dataset_url.split('/file/d/')[1].split('/')[0]
                    elif dataset_url and 'id=' in dataset_url:
                        file_id = dataset_url.split('id=')[1].split('&')[0]
                    
                    secrets_format = f'''[secrets]
dataset_url = "https://drive.google.com/uc?export=download&id={file_id}"'''
                    st.code(secrets_format, language="toml")
                    st.info("Copy this and paste into Streamlit Cloud Secrets")
        else:
            st.info("""
            **No secrets configured yet.**
            
            To add the dataset URL:
            1. Go to Streamlit Cloud → Your App → Settings → Secrets
            2. Add the following:
            ```toml
            [secrets]
            dataset_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
            ```
            3. Save and wait for redeployment
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Enable Demo Mode", type="primary"):
                st.session_state['use_demo_mode'] = True
                st.rerun()
        
        with col2:
            st.info("""
            **For production:**
            1. Download dataset from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)
            2. Upload to cloud storage (S3, GCS, etc.)
            3. Add URL to Streamlit Secrets as 'dataset_url'
            """)
        
        # Show current secrets status
        with st.expander("🔧 Debug: Secrets Configuration"):
            st.json(secrets_info)
            if dataset_url:
                st.code(f"Current URL: {dataset_url}", language="text")
        
        st.markdown("""
        **Demo Mode** creates synthetic data that allows you to:
        - Test all app features
        - Explore the UI and functionality
        - Train models (with synthetic data)
        
        Note: Model performance will not reflect real-world results.
        """)
        return
    
    if is_sampled:
        st.sidebar.info(f"Loaded {len(df):,} rows (sample from {total_rows:,} total)")
    else:
        st.sidebar.info(f"Loaded {len(df):,} rows (full dataset)")
    
    df_clean = preprocess_data(df)
    pages_needing_full_data = ["Model Training", "Hyperparameter Tuning", "Model Comparison", "Ensemble Methods",
                               "Predictions", "Batch Predictions", "Model Explainability", 
                               "Threshold Optimization", "Profit Curve Analysis", "Model Registry"]
    df_model = prepare_model_data(df_clean) if use_full_dataset or page in pages_needing_full_data else None
    
    if page == "Overview":
        show_overview(df, df_clean, is_sampled, total_rows)
    elif page == "Data Exploration":
        show_data_exploration(df_clean)
    elif page == "Advanced Analysis":
        show_advanced_analysis(df_clean)
    elif page == "Model Training":
        if df_model is None:
            st.warning("Full dataset is required for model training. Please enable 'Use Full Dataset' in the sidebar.")
            if st.button("Load Full Dataset Now"):
                st.rerun()
        else:
            show_model_training(df_model)
    elif page == "Model Comparison":
        if df_model is None:
            st.warning("Models need to be trained first. Please go to 'Model Training' page and enable 'Use Full Dataset'.")
        else:
            show_model_comparison(df_model)
    elif page == "Predictions":
        if df_model is None:
            st.warning("Models need to be trained first. Please go to 'Model Training' page and enable 'Use Full Dataset'.")
        else:
            show_predictions(df_model)
    elif page == "Batch Predictions":
        if df_model is None:
            st.warning("Models need to be trained first. Please go to 'Model Training' page and enable 'Use Full Dataset'.")
        else:
            show_batch_predictions(df_model)
    elif page == "Model Explainability":
        if df_model is None or 'log_model' not in st.session_state:
            st.warning("Models need to be trained first. Please go to 'Model Training' page.")
        else:
            show_model_explainability(df_model)
    elif page == "Threshold Optimization":
        if df_model is None or 'log_model' not in st.session_state:
            st.warning("Models need to be trained first. Please go to 'Model Training' page.")
        else:
            show_threshold_optimization(df_model)
    elif page == "Hyperparameter Tuning":
        if df_model is None:
            st.warning("Full dataset is required. Please enable 'Use Full Dataset' in the sidebar.")
        else:
            show_hyperparameter_tuning(df_model)
    elif page == "Profit Curve Analysis":
        if df_model is None or 'log_model' not in st.session_state:
            st.warning("Models need to be trained first. Please go to 'Model Training' page.")
        else:
            show_profit_curve(df_model)
    elif page == "Model Registry":
        if df_model is None:
            st.warning("Please load data first.")
        else:
            show_model_registry()
    elif page == "Feature Selection":
        if df_model is None:
            st.warning("Please load data first.")
        else:
            show_feature_selection(df_model)
    elif page == "Ensemble Methods":
        if df_model is None or 'log_model' not in st.session_state:
            st.warning("Models need to be trained first. Please go to 'Model Training' page.")
        else:
            show_ensemble_methods(df_model)
    elif page == "Insights & Recommendations":
        if df_model is None:
            st.warning("Please load data first.")
        else:
            show_insights_recommendations(df_clean, df_model)

def show_overview(df, df_clean, is_sampled=False, total_rows=0):
    st.header("Dataset Overview")
    
    if is_sampled:
        st.info(f"Showing statistics for a sample of {len(df):,} rows from {total_rows:,} total records.")
    
    # Check if TARGET column exists
    has_target = 'TARGET' in df.columns
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if is_sampled:
            st.metric("Loaded Records", f"{len(df):,}", f"of {total_rows:,}")
        else:
            st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", len(df.columns))
    with col3:
        if has_target:
            default_rate = df['TARGET'].mean() * 100
            st.metric("Default Rate", f"{default_rate:.2f}%")
        else:
            st.metric("Default Rate", "N/A")
            st.warning("⚠️ TARGET column not found. This might be a test dataset.")
    with col4:
        if has_target:
            default_rate = df['TARGET'].mean() * 100
            st.metric("Non-Default Rate", f"{100-default_rate:.2f}%")
        else:
            st.metric("Non-Default Rate", "N/A")
    
    if has_target:
        st.subheader("Target Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # Interactive pie chart with Plotly
            target_counts = df['TARGET'].value_counts()
            fig = px.pie(
                values=target_counts.values, 
                names=['Non-Default (0)', 'Default (1)'],
                title="Target Variable Distribution",
                color_discrete_sequence=['#10b981', '#ef4444'],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig)
        
        with col2:
            target_df = pd.DataFrame({
                'Status': ['Non-Default', 'Default'],
                'Count': [target_counts[0], target_counts[1]]
            })
            fig = px.bar(
                target_df, 
                x='Status', 
                y='Count',
                title="Target Count Distribution",
                color='Status',
                color_discrete_map={'Non-Default': '#10b981', 'Default': '#ef4444'},
                text='Count'
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(showlegend=False, height=400, yaxis_title="Number of Applications")
            st.plotly_chart(fig)
    else:
        st.warning("""
        **⚠️ TARGET column not found in dataset**
        
        This dataset appears to be missing the target variable. Possible reasons:
        - You may have loaded the test dataset (`application_test.csv`) instead of the training dataset (`application_train.csv`)
        - The dataset file might be incomplete
        
        **To fix:**
        - Make sure you're using `application_train.csv` which contains the TARGET column
        - Check your Streamlit Secrets configuration
        - Verify the dataset file is complete
        """)
    
    st.subheader("Dataset Information")
    if is_sampled:
        st.write(f"**Loaded Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns (sample)")
        st.write(f"**Total Dataset Size:** {total_rows:,} rows × {df.shape[1]} columns")
    else:
        st.write(f"**Original Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.write(f"**After Preprocessing:** {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    
    # Missing data summary
    st.subheader("Missing Data Summary")
    missing_summary = df_clean.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    if len(missing_summary) > 0:
        st.write(f"**Columns with missing values:** {len(missing_summary)}")
        st.dataframe(missing_summary.head(20))
    else:
        st.success("No missing values after preprocessing!")
    
    # Key statistics
    st.subheader("Key Statistics")
    numeric_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_CHILDREN']
    numeric_cols = [c for c in numeric_cols if c in df_clean.columns]
    if numeric_cols:
        stats_df = df_clean[numeric_cols].describe()
        st.dataframe(stats_df)

def show_data_exploration(df_clean):
    st.header("Data Exploration")
    
    # Sample data for faster visualization (if dataset is still large)
    max_viz_size = 20000
    if len(df_clean) > max_viz_size:
        viz_sample_size = st.sidebar.slider(
            "Sample Size for Visualizations", 
            5000, 
            min(len(df_clean), 50000), 
            min(max_viz_size, len(df_clean)),
            step=5000,
            help="Use a smaller sample for faster rendering of visualizations"
        )
        df_eda = df_clean.sample(n=min(viz_sample_size, len(df_clean)), random_state=42)
        st.info(f"Using {len(df_eda):,} rows for visualizations (from {len(df_clean):,} available)")
    else:
        df_eda = df_clean
    
    tab1, tab2, tab3, tab4 = st.tabs(["Financial Features", "Demographics", "Correlations", "Distributions"])
    
    with tab1:
        st.subheader("Financial Features Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interactive histogram with Plotly
            fig = px.histogram(
                df_eda, 
                x='AMT_INCOME_TOTAL',
                nbins=50,
                title="Income Distribution",
                labels={'AMT_INCOME_TOTAL': 'Total Income', 'count': 'Frequency'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig)
        
        with col2:
            # Interactive histogram for Credit
            fig = px.histogram(
                df_eda,
                x='AMT_CREDIT',
                nbins=50,
                title="Credit Amount Distribution",
                labels={'AMT_CREDIT': 'Credit Amount', 'count': 'Frequency'},
                color_discrete_sequence=['#10b981']
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig)
        
        col1, col2 = st.columns(2)
        
        if 'TARGET' in df_eda.columns:
            with col1:
                # Interactive box plot
                fig = px.box(
                    df_eda,
                    x='TARGET',
                    y='AMT_INCOME_TOTAL',
                    color='TARGET',
                    title="Income vs Default Status",
                    labels={'TARGET': 'Default (1 = Yes)', 'AMT_INCOME_TOTAL': 'Income'},
                    color_discrete_map={0: '#10b981', 1: '#ef4444'}
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig)
            
            with col2:
                # Interactive scatter plot
                sample_scatter = df_eda.sample(n=min(5000, len(df_eda)), random_state=42)
                fig = px.scatter(
                    sample_scatter,
                    x='AMT_INCOME_TOTAL',
                    y='AMT_CREDIT',
                    color='TARGET',
                    title="Income vs Credit by Default Status",
                    labels={'TARGET': 'Default Status', 'AMT_INCOME_TOTAL': 'Income', 'AMT_CREDIT': 'Credit'},
                    color_discrete_map={0: '#10b981', 1: '#ef4444'},
                    opacity=0.6
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig)
            
            # Credit to Income Ratio
            if 'CREDIT_INCOME_RATIO' in df_eda.columns:
                fig = px.box(
                    df_eda,
                    x='TARGET',
                    y='CREDIT_INCOME_RATIO',
                    color='TARGET',
                    title="Credit to Income Ratio vs Default",
                    labels={'TARGET': 'Default (1 = Yes)', 'CREDIT_INCOME_RATIO': 'Credit to Income Ratio'},
                    color_discrete_map={0: '#10b981', 1: '#ef4444'}
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig)
        else:
            st.warning("TARGET column not found. Some visualizations are unavailable.")
    
    with tab2:
        st.subheader("Demographic Analysis")
        
        # Gender analysis
        if 'CODE_GENDER' in df_eda.columns:
            col1, col2 = st.columns(2)
            with col1:
                gender_default = pd.crosstab(df_eda['CODE_GENDER'], df_eda['TARGET'], normalize='index') * 100
                gender_df = pd.DataFrame({
                    'Gender': gender_default.index,
                    'Non-Default': gender_default[0].values,
                    'Default': gender_default[1].values
                })
                fig = px.bar(
                    gender_df,
                    x='Gender',
                    y=['Non-Default', 'Default'],
                    title="Default Rate by Gender",
                    barmode='group',
                    color_discrete_map={'Non-Default': '#10b981', 'Default': '#ef4444'},
                    labels={'value': 'Percentage (%)', 'variable': 'Status'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig)
            
            with col2:
                if 'NAME_EDUCATION_TYPE' in df_eda.columns:
                    edu_default = pd.crosstab(df_eda['NAME_EDUCATION_TYPE'], df_eda['TARGET'], normalize='index') * 100
                    edu_df = pd.DataFrame({
                        'Education': edu_default.index,
                        'Default Rate': edu_default[1].values
                    }).sort_values('Default Rate')
                    fig = px.bar(
                        edu_df,
                        x='Default Rate',
                        y='Education',
                        orientation='h',
                        title="Default Rate by Education Level",
                        color='Default Rate',
                        color_continuous_scale='Reds',
                        labels={'Default Rate': 'Default Rate (%)'}
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig)
        
        # Age analysis
        if 'AGE' in df_eda.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df_eda,
                    x='AGE',
                    color='TARGET',
                    nbins=30,
                    title="Age Distribution by Default Status",
                    barmode='overlay',
                    opacity=0.7,
                    color_discrete_map={0: '#10b981', 1: '#ef4444'},
                    labels={'TARGET': 'Default Status', 'AGE': 'Age'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig)
            
            with col2:
                fig = px.box(
                    df_eda,
                    x='TARGET',
                    y='AGE',
                    color='TARGET',
                    title="Age vs Default Status",
                    labels={'TARGET': 'Default (1 = Yes)', 'AGE': 'Age'},
                    color_discrete_map={0: '#10b981', 1: '#ef4444'}
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig)
        
        # Family Status
        if 'NAME_FAMILY_STATUS' in df_eda.columns:
            family_default = pd.crosstab(df_eda['NAME_FAMILY_STATUS'], df_eda['TARGET'], normalize='index') * 100
            family_df = pd.DataFrame({
                'Family Status': family_default.index,
                'Default Rate': family_default[1].values
            }).sort_values('Default Rate')
            fig = px.bar(
                family_df,
                x='Default Rate',
                y='Family Status',
                orientation='h',
                title="Default Rate by Family Status",
                color='Default Rate',
                color_continuous_scale='Reds',
                labels={'Default Rate': 'Default Rate (%)'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig)
    
    with tab3:
        st.subheader("Feature Correlations")
        
        # Correlation with target
        numeric_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
        if 'TARGET' in numeric_cols:
            corr_with_target = df_eda[numeric_cols].corr()['TARGET'].sort_values(ascending=False)
            corr_with_target = corr_with_target[corr_with_target.index != 'TARGET']
            
            fig, ax = plt.subplots(figsize=(8, 10))
            top_corr = corr_with_target.head(15) if len(corr_with_target) > 15 else corr_with_target
            top_corr.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title("Top Features Correlated with Default (TARGET)")
            ax.set_xlabel("Correlation Coefficient")
            st.pyplot(fig)
        
        # Correlation heatmap
        key_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_CHILDREN', 
                       'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE', 'EMPLOY_YEARS']
        key_features = [f for f in key_features if f in df_eda.columns]
        if len(key_features) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df_eda[key_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True, ax=ax)
            ax.set_title("Correlation Heatmap - Key Features")
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Distribution Analysis")
        
        # External sources
        ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df_eda.columns]
        if ext_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in ext_cols:
                sns.kdeplot(data=df_eda, x=col, fill=True, label=col, ax=ax)
            ax.set_title("Distribution of External Credit Scores")
            ax.set_xlabel("Score (0 to 1)")
            ax.legend()
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            if 'EXT_SOURCE_2' in df_eda.columns:
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=df_eda, x='TARGET', y='EXT_SOURCE_2', ax=ax)
                    ax.set_title("EXT_SOURCE_2 vs Default")
                    ax.set_xlabel("Default (1 = Yes)")
                    st.pyplot(fig)
            
            if 'EXT_SOURCE_3' in df_eda.columns:
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=df_eda, x='TARGET', y='EXT_SOURCE_3', ax=ax)
                    ax.set_title("EXT_SOURCE_3 vs Default")
                    ax.set_xlabel("Default (1 = Yes)")
                    st.pyplot(fig)
        
        # Employment years
        if 'EMPLOY_YEARS' in df_eda.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data=df_eda, x='EMPLOY_YEARS', kde=True, color='teal', ax=ax, bins=30)
            ax.set_title("Employment Length Distribution")
            ax.set_xlabel("Years of Employment")
            st.pyplot(fig)

def show_advanced_analysis(df_clean):
    st.header("Advanced Analysis & Insights")
    
    # Check if TARGET column exists
    has_target = 'TARGET' in df_clean.columns
    if not has_target:
        st.warning("⚠️ TARGET column not found. This appears to be a test dataset. Advanced analysis requiring target variable is unavailable.")
        st.info("""
        **To enable full analysis:**
        - Make sure you're using `application_train.csv` which contains the TARGET column
        - Check your Streamlit Secrets configuration
        - Verify the dataset file is complete
        """)
    
    # Use reasonable sample size for advanced analysis
    max_analysis_size = 30000
    if len(df_clean) > max_analysis_size:
        analysis_sample_size = st.sidebar.slider(
            "Sample Size for Advanced Analysis",
            10000,
            min(len(df_clean), 50000),
            min(max_analysis_size, len(df_clean)),
            step=5000,
            key="advanced_analysis_sample"
        )
        df_eda = df_clean.sample(n=min(analysis_sample_size, len(df_clean)), random_state=42)
        st.info(f"Using {len(df_eda):,} rows for analysis (from {len(df_clean):,} available)")
    else:
        df_eda = df_clean
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Risk Factors", "🏢 Organization Analysis", "📅 Application Timing", "📋 Missing Data Patterns"])
    
    with tab1:
        st.subheader("Key Risk Factors Analysis")
        
        if not has_target:
            st.info("Risk factor analysis requires the TARGET column. Please use the training dataset.")
            return
        
        # Income type risk
        if 'NAME_INCOME_TYPE' in df_eda.columns:
            col1, col2 = st.columns(2)
            with col1:
                income_risk = df_eda.groupby('NAME_INCOME_TYPE')['TARGET'].agg(['mean', 'count'])
                income_risk = income_risk[income_risk['count'] >= 100]  # Filter small groups
                income_risk = income_risk.sort_values('mean', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                income_risk['mean'].plot(kind='barh', ax=ax, color='#e74c3c')
                ax.set_title("Default Rate by Income Type")
                ax.set_xlabel("Default Rate")
                st.pyplot(fig)
            
            with col2:
                # Housing type risk
                if 'NAME_HOUSING_TYPE' in df_eda.columns:
                    housing_risk = df_eda.groupby('NAME_HOUSING_TYPE')['TARGET'].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    housing_risk.plot(kind='barh', ax=ax, color='#c0392b')
                    ax.set_title("Default Rate by Housing Type")
                    ax.set_xlabel("Default Rate")
                    st.pyplot(fig)
        
        # Children and family members
        col1, col2 = st.columns(2)
        with col1:
            if 'CNT_CHILDREN' in df_eda.columns:
                children_risk = df_eda.groupby('CNT_CHILDREN')['TARGET'].mean()
                fig, ax = plt.subplots(figsize=(8, 5))
                children_risk.plot(kind='bar', ax=ax, color='#8e44ad')
                ax.set_title("Default Rate by Number of Children")
                ax.set_xlabel("Number of Children")
                ax.set_ylabel("Default Rate")
                st.pyplot(fig)
        
        with col2:
            if 'CNT_FAM_MEMBERS' in df_eda.columns:
                fam_risk = df_eda.groupby('CNT_FAM_MEMBERS')['TARGET'].mean()
                fam_risk = fam_risk[fam_risk.index <= 6]  # Focus on common values
                fig, ax = plt.subplots(figsize=(8, 5))
                fam_risk.plot(kind='bar', ax=ax, color='#16a085')
                ax.set_title("Default Rate by Family Size")
                ax.set_xlabel("Family Members")
                ax.set_ylabel("Default Rate")
                st.pyplot(fig)
        
        # Age groups risk
        if 'AGE' in df_eda.columns:
            df_eda['AGE_GROUP'] = pd.cut(df_eda['AGE'], bins=[0, 30, 40, 50, 60, 100], 
                                        labels=['<30', '30-40', '40-50', '50-60', '60+'])
            age_risk = df_eda.groupby('AGE_GROUP')['TARGET'].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            age_risk.plot(kind='bar', ax=ax, color='#3498db')
            ax.set_title("Default Rate by Age Group")
            ax.set_xlabel("Age Group")
            ax.set_ylabel("Default Rate")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
    
    with tab2:
        st.subheader("Organization Type Analysis")
        
        if 'ORGANIZATION_TYPE' in df_eda.columns:
            org_risk = df_eda.groupby('ORGANIZATION_TYPE')['TARGET'].agg(['mean', 'count'])
            org_risk = org_risk[org_risk['count'] >= 500]  # Filter small groups
            org_risk = org_risk.sort_values('mean', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(10, 12))
                org_risk['mean'].head(15).plot(kind='barh', ax=ax, color='#e67e22')
                ax.set_title("Top 15 Organization Types by Default Rate")
                ax.set_xlabel("Default Rate")
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 12))
                org_risk['count'].head(15).plot(kind='barh', ax=ax, color='#27ae60')
                ax.set_title("Top 15 Organization Types by Volume")
                ax.set_xlabel("Number of Applications")
                st.pyplot(fig)
            
            # Occupation type
            if 'OCCUPATION_TYPE' in df_eda.columns:
                occ_risk = df_eda.groupby('OCCUPATION_TYPE')['TARGET'].mean().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 8))
                occ_risk.plot(kind='barh', ax=ax, color='#9b59b6')
                ax.set_title("Default Rate by Occupation Type")
                ax.set_xlabel("Default Rate")
                st.pyplot(fig)
    
    with tab3:
        st.subheader("Application Timing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'WEEKDAY_APPR_PROCESS_START' in df_eda.columns:
                weekday_risk = df_eda.groupby('WEEKDAY_APPR_PROCESS_START')['TARGET'].mean()
                fig, ax = plt.subplots(figsize=(8, 5))
                weekday_risk.plot(kind='bar', ax=ax, color='#34495e')
                ax.set_title("Default Rate by Application Day")
                ax.set_xlabel("Day of Week")
                ax.set_ylabel("Default Rate")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
        
        with col2:
            if 'HOUR_APPR_PROCESS_START' in df_eda.columns:
                hour_risk = df_eda.groupby('HOUR_APPR_PROCESS_START')['TARGET'].mean()
                fig, ax = plt.subplots(figsize=(10, 5))
                hour_risk.plot(kind='line', marker='o', ax=ax, color='#e74c3c')
                ax.set_title("Default Rate by Application Hour")
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Default Rate")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # Region analysis
        if 'REGION_RATING_CLIENT' in df_eda.columns:
            region_risk = df_eda.groupby('REGION_RATING_CLIENT')['TARGET'].mean()
            fig, ax = plt.subplots(figsize=(8, 5))
            region_risk.plot(kind='bar', ax=ax, color='#1abc9c')
            ax.set_title("Default Rate by Region Rating")
            ax.set_xlabel("Region Rating")
            ax.set_ylabel("Default Rate")
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Missing Data Patterns")
        
        # Missing data heatmap
        missing_data = df_clean.isnull()
        missing_cols = missing_data.columns[missing_data.sum() > 0]
        
        if len(missing_cols) > 0:
            # Sample for visualization
            sample_missing = missing_data[missing_cols].sample(n=min(1000, len(missing_data)), random_state=42)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(sample_missing.T, cbar=True, yticklabels=True, 
                       xticklabels=False, cmap='viridis', ax=ax)
            ax.set_title("Missing Data Pattern (Sample)")
            st.pyplot(fig)
            
            # Missing by target
            if 'TARGET' in df_clean.columns:
                missing_by_target = {}
                for col in missing_cols[:10]:  # Top 10 columns
                    missing_by_target[col] = df_clean.groupby('TARGET')[col].apply(lambda x: x.isnull().mean())
                
                if missing_by_target:
                    missing_df = pd.DataFrame(missing_by_target).T
                    fig, ax = plt.subplots(figsize=(10, 6))
                    missing_df.plot(kind='barh', ax=ax, color=['#2ecc71', '#e74c3c'])
                    ax.set_title("Missing Data Rate by Target")
                    ax.set_xlabel("Missing Rate")
                    ax.legend(['Non-Default', 'Default'])
                    st.pyplot(fig)

def show_model_training(df_model):
    st.header("Model Training")
    
    st.info(f"Training on {len(df_model):,} records. This may take several minutes for large datasets.")
    
    # Training options
    col1, col2 = st.columns(2)
    with col1:
        use_smote = st.checkbox("Use SMOTE for Class Imbalance", value=True, 
                               help="Synthetic Minority Oversampling Technique to handle class imbalance")
        use_class_weights = st.checkbox("Use Class Weights", value=True,
                                        help="Automatically adjust class weights based on class frequency")
    with col2:
        use_feature_selection = st.checkbox("Use Feature Selection", value=True,
                                           help="Select top features using Random Forest importance")
        n_features = st.slider("Number of Features", 50, 200, 100, 10) if use_feature_selection else 200
    
    if st.button("Train Models", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()
        
        try:
            with st.spinner("Training models... This may take a few minutes."):
                status_text.text("Preparing data...")
                progress_bar.progress(10)
            # Prepare data
            status_text.text("Preparing features...")
            progress_bar.progress(20)
            X = df_model.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
            if 'TARGET' not in df_model.columns:
                st.error("❌ TARGET column not found in dataset. Cannot train models without target variable.")
                st.info("Please use `application_train.csv` which contains the TARGET column.")
                return
            y = df_model['TARGET']
            
            # Feature selection
            if use_feature_selection:
                status_text.text("Selecting best features...")
                progress_bar.progress(25)
                try:
                    from utils.feature_selection import select_features_by_importance
                    X_selected, selected_features = select_features_by_importance(X, y, n_features=n_features)
                    X = X_selected
                    st.info(f"Selected {len(selected_features)} features out of {df_model.shape[1]-2} total")
                except (ImportError, Exception) as e:
                    st.warning(f"Feature selection unavailable: {e}. Continuing with all features.")
            
            # Train-test split
            status_text.text("Splitting data into train/test sets...")
            progress_bar.progress(30)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Handle class imbalance with SMOTE
            if use_smote and IMBLEARN_AVAILABLE:
                status_text.text("Applying SMOTE for class imbalance...")
                progress_bar.progress(35)
                try:
                    smote = SMOTE(random_state=42, n_jobs=-1)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    st.success(f"After SMOTE: {len(X_train):,} samples (balanced)")
                except Exception as e:
                    st.warning(f"SMOTE failed: {str(e)}. Continuing without SMOTE.")
            
            # Calculate class weights
            class_weight = None
            if use_class_weights:
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_train)
                weights = compute_class_weight('balanced', classes=classes, y=y_train)
                class_weight = dict(zip(classes, weights))
            
            # Scale features
            status_text.text("Scaling features...")
            progress_bar.progress(40)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store in session state
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['X_train_scaled'] = X_train_scaled
            st.session_state['X_test_scaled'] = X_test_scaled
            st.session_state['scaler'] = scaler
            
            # Train Logistic Regression with improved settings
            st.subheader("Logistic Regression")
            status_text.text("Training Logistic Regression...")
            progress_bar.progress(50)
            log_model = LogisticRegression(
                max_iter=500, 
                random_state=42, 
                n_jobs=-1,
                class_weight=class_weight,
                C=0.1,
                solver='lbfgs'
            )
            log_model.fit(X_train_scaled, y_train)
            y_pred_log = log_model.predict(X_test_scaled)
            y_proba_log = log_model.predict_proba(X_test_scaled)[:, 1]
            
            st.session_state['log_model'] = log_model
            st.session_state['y_pred_log'] = y_pred_log
            st.session_state['y_proba_log'] = y_proba_log
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_log):.4f}")
            with col2:
                st.metric("Precision", f"{precision_score(y_test, y_pred_log):.4f}")
            with col3:
                st.metric("Recall", f"{recall_score(y_test, y_pred_log):.4f}")
            with col4:
                st.metric("F1 Score", f"{f1_score(y_test, y_pred_log):.4f}")
            with col5:
                st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba_log):.4f}")
            
            # Confusion Matrix with Plotly
            cm_log = confusion_matrix(y_test, y_pred_log)
            fig = px.imshow(
                cm_log,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Non-Default', 'Default'],
                y=['Non-Default', 'Default'],
                color_continuous_scale='Blues',
                text_auto=True,
                aspect="auto",
                title="Logistic Regression Confusion Matrix"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig)
            
            # Train Decision Tree with improved settings
            st.subheader("Decision Tree")
            status_text.text("Training Decision Tree...")
            progress_bar.progress(70)
            tree = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                class_weight=class_weight,
                random_state=42
            )
            tree.fit(X_train, y_train)
            y_pred_tree = tree.predict(X_test)
            y_proba_tree = tree.predict_proba(X_test)[:, 1]
            
            st.session_state['tree_model'] = tree
            st.session_state['y_pred_tree'] = y_pred_tree
            st.session_state['y_proba_tree'] = y_proba_tree
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_tree):.4f}")
            with col2:
                st.metric("Precision", f"{precision_score(y_test, y_pred_tree):.4f}")
            with col3:
                st.metric("Recall", f"{recall_score(y_test, y_pred_tree):.4f}")
            with col4:
                st.metric("F1 Score", f"{f1_score(y_test, y_pred_tree):.4f}")
            with col5:
                st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba_tree):.4f}")
            
            # Train Random Forest (better than KNN for this task)
            st.subheader("Random Forest")
            status_text.text("Training Random Forest...")
            progress_bar.progress(85)
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=50,
                min_samples_leaf=20,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)
            y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
            
            st.session_state['rf_model'] = rf_model
            st.session_state['y_pred_rf'] = y_pred_rf
            st.session_state['y_proba_rf'] = y_proba_rf
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_rf):.4f}")
            with col2:
                st.metric("Precision", f"{precision_score(y_test, y_pred_rf, zero_division=0):.4f}")
            with col3:
                st.metric("Recall", f"{recall_score(y_test, y_pred_rf, zero_division=0):.4f}")
            with col4:
                st.metric("F1 Score", f"{f1_score(y_test, y_pred_rf, zero_division=0):.4f}")
            with col5:
                st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba_rf):.4f}")
            
            # Train XGBoost if available
            if XGBOOST_AVAILABLE:
                st.subheader("XGBoost")
                status_text.text("Training XGBoost...")
                progress_bar.progress(90)
                
                # Calculate scale_pos_weight for XGBoost
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if use_class_weights else 1
                
                xgb_model = xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
                xgb_model.fit(X_train, y_train)
                y_pred_xgb = xgb_model.predict(X_test)
                y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
                
                st.session_state['xgb_model'] = xgb_model
                st.session_state['y_pred_xgb'] = y_pred_xgb
                st.session_state['y_proba_xgb'] = y_proba_xgb
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_xgb):.4f}")
                with col2:
                    st.metric("Precision", f"{precision_score(y_test, y_pred_xgb, zero_division=0):.4f}")
                with col3:
                    st.metric("Recall", f"{recall_score(y_test, y_pred_xgb, zero_division=0):.4f}")
                with col4:
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred_xgb, zero_division=0):.4f}")
                with col5:
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba_xgb):.4f}")
            
            # Train LightGBM if available
            if LIGHTGBM_AVAILABLE:
                st.subheader("LightGBM")
                status_text.text("Training LightGBM...")
                progress_bar.progress(95)
                
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if use_class_weights else 1
                
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                lgb_model.fit(X_train, y_train)
                y_pred_lgb = lgb_model.predict(X_test)
                y_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
                
                st.session_state['lgb_model'] = lgb_model
                st.session_state['y_pred_lgb'] = y_pred_lgb
                st.session_state['y_proba_lgb'] = y_proba_lgb
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_lgb):.4f}")
                with col2:
                    st.metric("Precision", f"{precision_score(y_test, y_pred_lgb, zero_division=0):.4f}")
                with col3:
                    st.metric("Recall", f"{recall_score(y_test, y_pred_lgb, zero_division=0):.4f}")
                with col4:
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred_lgb, zero_division=0):.4f}")
                with col5:
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba_lgb):.4f}")
            
            # Remove old KNN references
            if 'knn_model' in st.session_state:
                del st.session_state['knn_model']
                del st.session_state['y_pred_knn']
                del st.session_state['y_proba_knn']
                del st.session_state['y_test_knn']
            
                progress_bar.progress(100)
                elapsed_time = time.time() - start_time
                status_text.text("Complete!")
                logger.info(f"Model training completed in {elapsed_time:.2f} seconds")
                st.success(f"All models trained successfully! (Time: {elapsed_time:.1f}s)")
                progress_bar.empty()
                status_text.empty()
                
                # Record metrics
                metrics = calculate_comprehensive_metrics(
                    st.session_state['y_test'],
                    st.session_state['y_pred_log'],
                    st.session_state['y_proba_log']
                )
                monitor.record_model_metrics(metrics)
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            st.error(f"Error during training: {str(e)}")
            st.exception(e)
    
    else:
        st.info("Click the button above to train the models.")

def show_model_comparison(df_model):
    st.header("Model Comparison")
    
    if 'log_model' not in st.session_state:
        st.warning("Please train the models first in the 'Model Training' page.")
        return
    
    # Get predictions
    y_test = st.session_state['y_test']
    y_proba_log = st.session_state['y_proba_log']
    y_pred_log = st.session_state['y_pred_log']
    
    y_proba_tree = st.session_state['y_proba_tree']
    y_pred_tree = st.session_state['y_pred_tree']
    
    # Metrics comparison - build dynamically based on available models
    model_results = []
    
    model_results.append({
        'Model': 'Logistic Regression',
        'Accuracy': accuracy_score(y_test, y_pred_log),
        'Precision': precision_score(y_test, y_pred_log, zero_division=0),
        'Recall': recall_score(y_test, y_pred_log, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred_log, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba_log)
    })
    
    model_results.append({
        'Model': 'Decision Tree',
        'Accuracy': accuracy_score(y_test, y_pred_tree),
        'Precision': precision_score(y_test, y_pred_tree, zero_division=0),
        'Recall': recall_score(y_test, y_pred_tree, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred_tree, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_proba_tree)
    })
    
    if 'rf_model' in st.session_state:
        y_pred_rf = st.session_state['y_pred_rf']
        y_proba_rf = st.session_state['y_proba_rf']
        model_results.append({
            'Model': 'Random Forest',
            'Accuracy': accuracy_score(y_test, y_pred_rf),
            'Precision': precision_score(y_test, y_pred_rf, zero_division=0),
            'Recall': recall_score(y_test, y_pred_rf, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred_rf, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba_rf)
        })
    
    if 'xgb_model' in st.session_state:
        y_pred_xgb = st.session_state['y_pred_xgb']
        y_proba_xgb = st.session_state['y_proba_xgb']
        model_results.append({
            'Model': 'XGBoost',
            'Accuracy': accuracy_score(y_test, y_pred_xgb),
            'Precision': precision_score(y_test, y_pred_xgb, zero_division=0),
            'Recall': recall_score(y_test, y_pred_xgb, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred_xgb, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba_xgb)
        })
    
    if 'lgb_model' in st.session_state:
        y_pred_lgb = st.session_state['y_pred_lgb']
        y_proba_lgb = st.session_state['y_proba_lgb']
        model_results.append({
            'Model': 'LightGBM',
            'Accuracy': accuracy_score(y_test, y_pred_lgb),
            'Precision': precision_score(y_test, y_pred_lgb, zero_division=0),
            'Recall': recall_score(y_test, y_pred_lgb, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred_lgb, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_proba_lgb)
        })
    
    results = pd.DataFrame(model_results)
    
    st.subheader("Performance Metrics Comparison")
    st.dataframe(results.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']))
    
    # ROC Curves with Plotly
    st.subheader("ROC Curve Comparison")
    
    # Calculate ROC curves
    fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
    auc_log = roc_auc_score(y_test, y_proba_log)
    
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_tree)
    auc_tree = roc_auc_score(y_test, y_proba_tree)
    
    # Create interactive Plotly figure
    fig = go.Figure()
    
    # Add ROC curves
    fig.add_trace(go.Scatter(
        x=fpr_log, y=tpr_log,
        mode='lines',
        name=f'Logistic Regression (AUC={auc_log:.3f})',
        line=dict(color='#667eea', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=fpr_tree, y=tpr_tree,
        mode='lines',
        name=f'Decision Tree (AUC={auc_tree:.3f})',
        line=dict(color='#10b981', width=3)
    ))
    
    # Add Random Forest if available
    if 'rf_model' in st.session_state:
        y_proba_rf = st.session_state['y_proba_rf']
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
        auc_rf = roc_auc_score(y_test, y_proba_rf)
        fig.add_trace(go.Scatter(
            x=fpr_rf, y=tpr_rf,
            mode='lines',
            name=f'Random Forest (AUC={auc_rf:.3f})',
            line=dict(color='#f59e0b', width=3)
        ))
    
    # Add XGBoost if available
    if 'xgb_model' in st.session_state:
        y_proba_xgb = st.session_state['y_proba_xgb']
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
        auc_xgb = roc_auc_score(y_test, y_proba_xgb)
        fig.add_trace(go.Scatter(
            x=fpr_xgb, y=tpr_xgb,
            mode='lines',
            name=f'XGBoost (AUC={auc_xgb:.3f})',
            line=dict(color='#ef4444', width=3)
        ))
    
    # Add LightGBM if available
    if 'lgb_model' in st.session_state:
        y_proba_lgb = st.session_state['y_proba_lgb']
        fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba_lgb)
        auc_lgb = roc_auc_score(y_test, y_proba_lgb)
        fig.add_trace(go.Scatter(
            x=fpr_lgb, y=tpr_lgb,
            mode='lines',
            name=f'LightGBM (AUC={auc_lgb:.3f})',
            line=dict(color='#8b5cf6', width=3)
        ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dot')
    ))
    
    fig.update_layout(
        title='ROC Curve Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig)
    
    # Metrics comparison bar chart
    st.subheader("Metrics Comparison")
    metrics_df = results.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    fig = px.bar(
        metrics_df,
        x='Model',
        y='Score',
        color='Metric',
        barmode='group',
        title='Model Performance Metrics Comparison',
        color_discrete_sequence=['#667eea', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
        labels={'Score': 'Score', 'Model': 'Model', 'Metric': 'Metric'}
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig)
    
    # Feature Importance (for Decision Tree) with Plotly
    st.subheader("Feature Importance (Decision Tree)")
    tree_model = st.session_state['tree_model']
    X = df_model.drop(columns=['TARGET', 'SK_ID_CURR'], errors='ignore')
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': tree_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    top_features = feature_importance.head(15)
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 15 Most Important Features (Decision Tree)',
        color='Importance',
        color_continuous_scale='Blues',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature Name'}
    )
    fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig)
    
    st.dataframe(feature_importance.head(20).style.background_gradient(subset=['Importance'], cmap='Blues'))

def show_predictions(df_model):
    st.header("Make Predictions")
    
    if 'log_model' not in st.session_state:
        st.warning("Please train the models first in the 'Model Training' page.")
        return
    
    st.subheader("Enter Applicant Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amt_income = st.number_input("Total Income (AMT_INCOME_TOTAL)", min_value=0.0, value=200000.0)
        amt_credit = st.number_input("Credit Amount (AMT_CREDIT)", min_value=0.0, value=400000.0)
        amt_annuity = st.number_input("Annuity Amount (AMT_ANNUITY)", min_value=0.0, value=25000.0)
        cnt_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    
    with col2:
        ext_source_2 = st.slider("External Source 2", 0.0, 1.0, 0.5, 0.01)
        ext_source_3 = st.slider("External Source 3", 0.0, 1.0, 0.5, 0.01)
        age = st.slider("Age", 18, 100, 35)
        employ_years = st.slider("Years of Employment", 0.0, 50.0, 5.0, 0.1)
    
    with col3:
        code_gender = st.selectbox("Gender", ['M', 'F'])
        name_education = st.selectbox("Education Type", 
            ['Higher education', 'Secondary / secondary special', 'Incomplete higher', 
             'Lower secondary', 'Academic degree'])
        name_family_status = st.selectbox("Family Status",
            ['Single / not married', 'Married', 'Civil marriage', 'Separated', 
             'Widow', 'Unknown'])
        flag_own_car = st.selectbox("Own Car", ['Y', 'N'])
    
    if st.button("Predict Default Risk", type="primary"):
        try:
            # Get the trained model and scaler
            model = st.session_state['log_model']
            scaler = st.session_state['scaler']
            X_train = st.session_state['X_train']
            
            # Create a sample row with median/default values from training data
            sample_input = X_train.iloc[0:1].copy()
            
            # Update with user inputs
            feature_updates = {
                'AMT_INCOME_TOTAL': amt_income,
                'AMT_CREDIT': amt_credit,
                'AMT_ANNUITY': amt_annuity,
                'CNT_CHILDREN': cnt_children,
                'EXT_SOURCE_2': ext_source_2,
                'EXT_SOURCE_3': ext_source_3,
                'AGE': age,
                'EMPLOY_YEARS': employ_years
            }
            
            for feature, value in feature_updates.items():
                if feature in sample_input.columns:
                    sample_input[feature] = value
            
            # Calculate derived features if base features are present
            if 'AMT_CREDIT' in sample_input.columns and 'AMT_INCOME_TOTAL' in sample_input.columns:
                if 'CREDIT_INCOME_RATIO' in sample_input.columns:
                    sample_input['CREDIT_INCOME_RATIO'] = sample_input['AMT_CREDIT'] / sample_input['AMT_INCOME_TOTAL'].replace({0: np.nan})
            
            if 'AMT_ANNUITY' in sample_input.columns and 'AMT_INCOME_TOTAL' in sample_input.columns:
                if 'ANNUITY_INCOME_RATIO' in sample_input.columns:
                    sample_input['ANNUITY_INCOME_RATIO'] = sample_input['AMT_ANNUITY'] / sample_input['AMT_INCOME_TOTAL'].replace({0: np.nan})
            
            # Scale and predict
            sample_scaled = scaler.transform(sample_input)
            probability = model.predict_proba(sample_scaled)[0, 1]
            risk_level = "High Risk" if probability > 0.5 else "Low Risk"
        
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")
            st.exception(e)
            return
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Default Probability", f"{probability:.2%}")
        with col2:
            st.metric("Risk Level", risk_level)
        
        # Visualize probability with Plotly gauge
        st.subheader("Prediction Visualization")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Default Risk Probability (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#ef4444" if probability > 0.5 else "#10b981"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig)
        
        with col2:
            risk_color = "#ef4444" if probability > 0.5 else "#10b981"
            st.markdown(f"""
            <div style="padding: 1.5rem; border: 2px solid {risk_color}; border-radius: 8px; text-align: center;">
                <h3 style="color: {risk_color}; margin: 0.5rem 0;">{risk_level}</h3>
                <p style="color: #666; margin: 0;">Probability: {probability:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

def show_batch_predictions(df_model):
    st.header("Batch Predictions")
    
    if 'log_model' not in st.session_state:
        st.warning("Please train the models first in the 'Model Training' page.")
        return
    
    st.subheader("Upload CSV File for Batch Predictions")
    st.info("Upload a CSV file with applicant data. The file should contain the same features as the training dataset.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df):,} records")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(batch_df.head(10))
            
            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Processing predictions... This may take a minute for large files."):
                    model = st.session_state['log_model']
                    scaler = st.session_state['scaler']
                    X_train = st.session_state['X_train']
                    
                    # Get the imputer from training (if available)
                    # For now, we'll create a new one but use the same strategy
                    from sklearn.impute import SimpleImputer
                    
                    # Prepare batch data using same preprocessing pipeline
                    batch_processed = batch_df.copy()
                    
                    # Drop columns with >40% missing
                    missing = batch_processed.isnull().mean()
                    cols_to_drop = missing[missing > 0.40].index.tolist()
                    batch_processed = batch_processed.drop(columns=cols_to_drop)
                    
                    # Handle anomalous DAYS_EMPLOYED
                    if 'DAYS_EMPLOYED' in batch_processed.columns:
                        batch_processed['DAYS_EMPLOYED'] = batch_processed['DAYS_EMPLOYED'].replace(365243, np.nan)
                    
                    # Feature Engineering
                    if 'DAYS_BIRTH' in batch_processed.columns:
                        batch_processed['AGE'] = -batch_processed['DAYS_BIRTH'] / 365.0
                    if 'DAYS_EMPLOYED' in batch_processed.columns:
                        batch_processed['EMPLOY_YEARS'] = -(batch_processed['DAYS_EMPLOYED'] / 365.0)
                    if {'AMT_CREDIT', 'AMT_INCOME_TOTAL'}.issubset(batch_processed.columns):
                        batch_processed['CREDIT_INCOME_RATIO'] = batch_processed['AMT_CREDIT'] / batch_processed['AMT_INCOME_TOTAL'].replace({0: np.nan})
                    if {'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(batch_processed.columns):
                        batch_processed['ANNUITY_INCOME_RATIO'] = batch_processed['AMT_ANNUITY'] / batch_processed['AMT_INCOME_TOTAL'].replace({0: np.nan})
                    
                    exts = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in batch_processed.columns]
                    if exts:
                        batch_processed['EXT_SOURCES_MEAN'] = batch_processed[exts].mean(axis=1)
                    
                    # Impute numeric columns
                    num_cols = batch_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    num_cols = [c for c in num_cols if c not in ('SK_ID_CURR', 'TARGET')]
                    
                    # Use stored imputer from training if available
                    if 'num_imputer' in st.session_state and len(num_cols) > 0:
                        train_num_cols = [c for c in num_cols if c in X_train.columns]
                        if len(train_num_cols) > 0:
                            batch_processed[train_num_cols] = st.session_state['num_imputer'].transform(batch_processed[train_num_cols])
                    elif len(num_cols) > 0:
                        # Fallback: fit new imputer
                        num_imputer = SimpleImputer(strategy='median')
                        train_num_cols = [c for c in num_cols if c in X_train.columns]
                        if len(train_num_cols) > 0:
                            batch_processed[train_num_cols] = num_imputer.fit_transform(batch_processed[train_num_cols])
                    
                    # Encode categorical columns
                    cat_cols = batch_processed.select_dtypes(include=['object']).columns.tolist()
                    for c in cat_cols:
                        batch_processed[c] = batch_processed[c].fillna('MISSING')
                        batch_processed[c] = pd.Categorical(batch_processed[c]).codes
                    
                    # IQR capping
                    def cap_iqr(series, k=1.5):
                        q1 = series.quantile(0.25)
                        q3 = series.quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - k * iqr
                        upper = q3 + k * iqr
                        return series.clip(lower=lower, upper=upper)
                    
                    cap_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_INCOME_RATIO']
                    cap_features = [c for c in cap_features if c in batch_processed.columns]
                    for c in cap_features:
                        batch_processed[c] = cap_iqr(batch_processed[c], k=1.5)
                    
                    # Align with training features
                    feature_cols = list(X_train.columns)
                    missing_cols = set(feature_cols) - set(batch_processed.columns)
                    for col in missing_cols:
                        batch_processed[col] = 0
                    
                    extra_cols = set(batch_processed.columns) - set(feature_cols)
                    if 'SK_ID_CURR' in extra_cols:
                        extra_cols.remove('SK_ID_CURR')
                    batch_processed = batch_processed.drop(columns=extra_cols)
                    
                    # Reorder to match training
                    if 'SK_ID_CURR' in batch_processed.columns:
                        sk_ids = batch_processed['SK_ID_CURR']
                        batch_processed = batch_processed[feature_cols]
                    else:
                        sk_ids = pd.Series(range(len(batch_processed)), name='SK_ID_CURR')
                    
                    # Scale features
                    batch_scaled = scaler.transform(batch_processed[feature_cols])
                    
                    # Generate predictions
                    probabilities = model.predict_proba(batch_scaled)[:, 1]
                    predictions = (probabilities >= 0.5).astype(int)
                    
                    # Create results dataframe
                    results = batch_df.copy() if 'SK_ID_CURR' in batch_df.columns else pd.DataFrame()
                    if len(results) == 0:
                        results = pd.DataFrame(index=range(len(probabilities)))
                    
                    results['Default_Probability'] = probabilities
                    results['Predicted_Default'] = predictions
                    results['Risk_Level'] = results['Default_Probability'].apply(
                        lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
                    )
                    
                    st.success(f"Predictions generated for {len(results):,} records!")
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(results)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(results))
                    with col2:
                        st.metric("High Risk", len(results[results['Risk_Level'] == 'High']))
                    with col3:
                        st.metric("Medium Risk", len(results[results['Risk_Level'] == 'Medium']))
                    with col4:
                        st.metric("Low Risk", len(results[results['Risk_Level'] == 'Low']))
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    else:
        st.info("Please upload a CSV file to get started")

def show_model_explainability(df_model):
    st.header("Model Explainability")
    
    if 'log_model' not in st.session_state:
        st.warning("Please train the models first in the 'Model Training' page.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "SHAP Values", "Partial Dependence"])
    
    with tab1:
        st.subheader("Feature Importance Analysis")
        
        # Logistic Regression coefficients
        if 'log_model' in st.session_state:
            model = st.session_state['log_model']
            X_train = st.session_state['X_train']
            
            # Get feature importance from coefficients
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Coefficient': model.coef_[0],
                'Abs_Coefficient': np.abs(model.coef_[0])
            }).sort_values('Abs_Coefficient', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top positive coefficients
                top_positive = feature_importance.nlargest(10, 'Coefficient')
                fig = px.bar(
                    top_positive,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Positive Features (Logistic Regression)',
                    color='Coefficient',
                    color_continuous_scale='Greens',
                    labels={'Coefficient': 'Coefficient Value'}
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig)
            
            with col2:
                # Top negative coefficients
                top_negative = feature_importance.nsmallest(10, 'Coefficient')
                fig = px.bar(
                    top_negative,
                    x='Coefficient',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Negative Features (Logistic Regression)',
                    color='Coefficient',
                    color_continuous_scale='Reds',
                    labels={'Coefficient': 'Coefficient Value'}
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig)
            
            st.dataframe(feature_importance.head(20))
    
    with tab2:
        st.subheader("SHAP Values (SHapley Additive exPlanations)")
        
        if not SHAP_AVAILABLE:
            st.warning("SHAP library not installed. Install with: pip install shap")
            st.info("SHAP values help explain individual predictions by showing the contribution of each feature.")
        else:
            try:
                model = st.session_state['log_model']
                X_train = st.session_state['X_train']
                X_test = st.session_state['X_test']
                
                # Sample for SHAP (computationally expensive)
                sample_size = min(100, len(X_test))
                X_sample = X_test[:sample_size]
                
                # Create SHAP explainer
                explainer = shap.LinearExplainer(model, X_train[:1000])
                shap_values = explainer.shap_values(X_sample)
                
                # Summary plot
                st.subheader("SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                st.pyplot(fig)
                
                # Waterfall plot for a single prediction
                st.subheader("SHAP Waterfall Plot (Single Prediction)")
                idx = st.slider("Select prediction index", 0, sample_size-1, 0)
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(shap.Explanation(values=shap_values[idx], 
                                                     base_values=explainer.expected_value,
                                                     data=X_sample.iloc[idx],
                                                     feature_names=X_sample.columns), show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating SHAP values: {str(e)}")
    
    with tab3:
        st.subheader("Partial Dependence Plots")
        st.info("Partial dependence plots show the marginal effect of a feature on the predicted outcome.")
        
        if 'log_model' in st.session_state:
            model = st.session_state['log_model']
            X_train = st.session_state['X_train']
            
            # Select feature for PDP
            feature_to_plot = st.selectbox("Select Feature", X_train.columns[:20])
            
            # Sample data for faster computation
            sample_data = X_train.sample(n=min(1000, len(X_train)), random_state=42)
            
            # Calculate partial dependence
            feature_values = np.linspace(sample_data[feature_to_plot].min(), 
                                        sample_data[feature_to_plot].max(), 50)
            pdp_values = []
            
            for val in feature_values:
                temp_data = sample_data.copy()
                temp_data[feature_to_plot] = val
                scaled_temp = st.session_state['scaler'].transform(temp_data)
                preds = model.predict_proba(scaled_temp)[:, 1]
                pdp_values.append(preds.mean())
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=feature_values,
                y=pdp_values,
                mode='lines',
                name='Partial Dependence',
                line=dict(color='#667eea', width=3)
            ))
            fig.update_layout(
                title=f'Partial Dependence Plot: {feature_to_plot}',
                xaxis_title=feature_to_plot,
                yaxis_title='Average Predicted Probability',
                height=500
            )
            st.plotly_chart(fig)

def show_threshold_optimization(df_model):
    st.header("Threshold Optimization")
    
    if 'log_model' not in st.session_state:
        st.warning("Please train the models first in the 'Model Training' page.")
        return
    
    st.subheader("Find Optimal Classification Threshold")
    st.info("Adjust the classification threshold to balance precision and recall based on your business needs.")
    
    model = st.session_state['log_model']
    y_test = st.session_state['y_test']
    y_proba = st.session_state['y_proba_log']
    
    # Calculate metrics for different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    metrics_df = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        metrics_df.append({
            'Threshold': threshold,
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1': f1_score(y_test, y_pred, zero_division=0),
            'Accuracy': accuracy_score(y_test, y_pred)
        })
    
    metrics_df = pd.DataFrame(metrics_df)
    
    # Plot metrics vs threshold
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics_df['Threshold'], y=metrics_df['Precision'], 
                            name='Precision', line=dict(color='#10b981', width=3)))
    fig.add_trace(go.Scatter(x=metrics_df['Threshold'], y=metrics_df['Recall'], 
                            name='Recall', line=dict(color='#ef4444', width=3)))
    fig.add_trace(go.Scatter(x=metrics_df['Threshold'], y=metrics_df['F1'], 
                            name='F1 Score', line=dict(color='#667eea', width=3)))
    fig.add_trace(go.Scatter(x=metrics_df['Threshold'], y=metrics_df['Accuracy'], 
                            name='Accuracy', line=dict(color='#f59e0b', width=3, dash='dash')))
    
    fig.update_layout(
        title='Metrics vs Classification Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig)
    
    # Interactive threshold selector
    st.subheader("Interactive Threshold Selection")
    selected_threshold = st.slider("Select Threshold", 0.0, 1.0, 0.5, 0.01)
    
    y_pred_selected = (y_proba >= selected_threshold).astype(int)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Precision", f"{precision_score(y_test, y_pred_selected, zero_division=0):.4f}")
    with col2:
        st.metric("Recall", f"{recall_score(y_test, y_pred_selected, zero_division=0):.4f}")
    with col3:
        st.metric("F1 Score", f"{f1_score(y_test, y_pred_selected, zero_division=0):.4f}")
    with col4:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_selected):.4f}")
    
    # Confusion matrix for selected threshold
    cm = confusion_matrix(y_test, y_pred_selected)
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Non-Default', 'Default'],
        y=['Non-Default', 'Default'],
        color_continuous_scale='Blues',
        text_auto=True,
        aspect="auto",
        title=f"Confusion Matrix (Threshold = {selected_threshold:.2f})"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig)
    
    # Find optimal threshold (F1 score)
    optimal_idx = metrics_df['F1'].idxmax()
    optimal_threshold = metrics_df.loc[optimal_idx, 'Threshold']
    st.success(f"Optimal threshold (max F1): {optimal_threshold:.3f}")

def show_insights_recommendations(df_clean, df_model):
    st.header("Insights & Recommendations")
    
    st.subheader("Key Insights")
    
    insights = []
    
    if 'TARGET' not in df_clean.columns:
        st.warning("⚠️ TARGET column not found. Insights requiring target variable are unavailable.")
        st.info("Please use `application_train.csv` which contains the TARGET column for full insights.")
        return
    
    default_rate = df_clean['TARGET'].mean() * 100
    insights.append(f"**Default Rate**: {default_rate:.2f}% of applicants default on their loans")
    
    if 'EXT_SOURCE_2' in df_clean.columns:
        ext2_default = df_clean.groupby('TARGET')['EXT_SOURCE_2'].mean()
        if len(ext2_default) == 2:
            insights.append(f"**External Source 2**: Defaulters have {ext2_default[1]:.3f} avg score vs {ext2_default[0]:.3f} for non-defaulters")
    
    if 'AMT_INCOME_TOTAL' in df_clean.columns:
        income_default = df_clean.groupby('TARGET')['AMT_INCOME_TOTAL'].mean()
        if len(income_default) == 2:
            insights.append(f"**Income Gap**: Defaulters earn ${income_default[1]:,.0f} avg vs ${income_default[0]:,.0f} for non-defaulters")
    
    if 'AGE' in df_clean.columns:
        age_default = df_clean.groupby('TARGET')['AGE'].mean()
        if len(age_default) == 2:
            insights.append(f"**Age Factor**: Average age of defaulters is {age_default[1]:.1f} vs {age_default[0]:.1f} for non-defaulters")
    
    for insight in insights:
        st.markdown(f"- {insight}")
    
    st.subheader("Business Recommendations")
    
    recommendations = [
        "**Focus on External Credit Scores**: EXT_SOURCE_2 and EXT_SOURCE_3 are strong predictors.",
        "**Income Verification**: Implement stricter income verification for applicants with income below median levels.",
        "**Risk Segmentation**: Segment customers into risk tiers based on probability scores.",
        "**Threshold Optimization**: Use threshold optimization to balance false positives and false negatives.",
        "**Model Monitoring**: Regularly monitor model performance and retrain when needed.",
        "**Feature Engineering**: Consider creating more interaction features.",
        "**Advanced Models**: Experiment with ensemble methods (XGBoost, Random Forest) for better performance."
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    if 'log_model' in st.session_state:
        st.subheader("Current Model Performance Summary")
        
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred_log']
        y_proba = st.session_state['y_proba_log']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
        with col2:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        with col3:
            st.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        with col4:
            st.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")

def show_hyperparameter_tuning(df_model):
    st.header("Hyperparameter Tuning")
    
    st.info("Optimize model hyperparameters to improve performance. This may take several minutes.")
    
    if 'X_train' not in st.session_state:
        st.warning("Please train models first in the 'Model Training' page to get train/test splits.")
        return
    
    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    X_train_scaled = st.session_state['X_train_scaled']
    X_test_scaled = st.session_state['X_test_scaled']
    
    model_type = st.selectbox("Select Model to Tune", 
                              ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"])
    
    if model_type == "Logistic Regression":
        st.subheader("Logistic Regression Hyperparameter Tuning")
        
        col1, col2 = st.columns(2)
        with col1:
            C_values = st.multiselect("C (Regularization)", 
                                     [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                                     default=[0.1, 1.0, 10.0])
        with col2:
            max_iter = st.slider("Max Iterations", 100, 1000, 200, 50)
        
        if st.button("Run Grid Search", type="primary"):
            with st.spinner("Tuning hyperparameters... This may take a few minutes."):
                param_grid = {'C': C_values, 'max_iter': [max_iter]}
                grid_search = GridSearchCV(
                    LogisticRegression(random_state=42, n_jobs=-1),
                    param_grid,
                    cv=3,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                
                st.success("Hyperparameter tuning complete!")
                
                st.subheader("Best Parameters")
                st.json(grid_search.best_params_)
                
                st.subheader("Best Score (CV)")
                st.metric("ROC-AUC (Cross-Validation)", f"{grid_search.best_score_:.4f}")
                
                # Test on test set
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_scaled)
                y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                with col2:
                    st.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
                with col3:
                    st.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
                with col4:
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
                with col5:
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
                
                # Save best model
                st.session_state['tuned_log_model'] = best_model
                st.session_state['tuned_log_params'] = grid_search.best_params_
    
    elif model_type == "Random Forest":
        st.subheader("Random Forest Hyperparameter Tuning")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.multiselect("n_estimators", [50, 100, 200, 300], default=[100, 200])
        with col2:
            max_depth = st.multiselect("max_depth", [5, 10, 15, 20, None], default=[10, 15])
        with col3:
            min_samples_split = st.multiselect("min_samples_split", [2, 5, 10], default=[2, 5])
        
        if st.button("Run Grid Search", type="primary"):
            with st.spinner("Tuning hyperparameters... This may take several minutes."):
                param_grid = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split
                }
                grid_search = RandomizedSearchCV(
                    RandomForestClassifier(random_state=42, n_jobs=-1),
                    param_grid,
                    cv=3,
                    scoring='roc_auc',
                    n_iter=10,
                    n_jobs=-1,
                    random_state=42
                )
                grid_search.fit(X_train, y_train)
                
                st.success("Hyperparameter tuning complete!")
                st.subheader("Best Parameters")
                st.json(grid_search.best_params_)
                st.metric("Best CV Score", f"{grid_search.best_score_:.4f}")
                
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_proba = best_model.predict_proba(X_test)[:, 1]
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
                with col2:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                with col3:
                    st.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
                with col4:
                    st.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
                with col5:
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
                
                st.session_state['tuned_rf_model'] = best_model
    
    elif model_type == "XGBoost":
        if not XGBOOST_AVAILABLE:
            st.warning("XGBoost not installed. Install with: pip install xgboost")
        else:
            st.subheader("XGBoost Hyperparameter Tuning")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                n_estimators = st.multiselect("n_estimators", [50, 100, 200], default=[100, 200])
            with col2:
                max_depth = st.multiselect("max_depth", [3, 5, 7], default=[3, 5])
            with col3:
                learning_rate = st.multiselect("learning_rate", [0.01, 0.1, 0.3], default=[0.1, 0.3])
            
            if st.button("Run Grid Search", type="primary"):
                with st.spinner("Tuning XGBoost... This may take several minutes."):
                    param_grid = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate
                    }
                    grid_search = RandomizedSearchCV(
                        xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
                        param_grid,
                        cv=3,
                        scoring='roc_auc',
                        n_iter=10,
                        n_jobs=-1,
                        random_state=42
                    )
                    grid_search.fit(X_train, y_train)
                    
                    st.success("XGBoost tuning complete!")
                    st.subheader("Best Parameters")
                    st.json(grid_search.best_params_)
                    st.metric("Best CV Score", f"{grid_search.best_score_:.4f}")
                    
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
                    with col2:
                        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    with col3:
                        st.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
                    with col4:
                        st.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
                    with col5:
                        st.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
                    
                    st.session_state['tuned_xgb_model'] = best_model
                    st.session_state['tuned_xgb_params'] = grid_search.best_params_
    
    elif model_type == "LightGBM":
        if not LIGHTGBM_AVAILABLE:
            st.warning("LightGBM not installed. Install with: pip install lightgbm")
        else:
            st.subheader("LightGBM Hyperparameter Tuning")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                n_estimators = st.multiselect("n_estimators", [50, 100, 200], default=[100, 200])
            with col2:
                max_depth = st.multiselect("max_depth", [3, 5, 7], default=[3, 5])
            with col3:
                learning_rate = st.multiselect("learning_rate", [0.01, 0.1, 0.3], default=[0.1, 0.3])
            
            if st.button("Run Grid Search", type="primary"):
                with st.spinner("Tuning LightGBM... This may take several minutes."):
                    param_grid = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'learning_rate': learning_rate
                    }
                    grid_search = RandomizedSearchCV(
                        lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
                        param_grid,
                        cv=3,
                        scoring='roc_auc',
                        n_iter=10,
                        n_jobs=-1,
                        random_state=42
                    )
                    grid_search.fit(X_train, y_train)
                    
                    st.success("LightGBM tuning complete!")
                    st.subheader("Best Parameters")
                    st.json(grid_search.best_params_)
                    st.metric("Best CV Score", f"{grid_search.best_score_:.4f}")
                    
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)[:, 1]
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
                    with col2:
                        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                    with col3:
                        st.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
                    with col4:
                        st.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
                    with col5:
                        st.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
                    
                    st.session_state['tuned_lgb_model'] = best_model
                    st.session_state['tuned_lgb_params'] = grid_search.best_params_

def show_profit_curve(df_model):
    st.header("Profit Curve Analysis")
    
    if 'log_model' not in st.session_state:
        st.warning("Please train the models first in the 'Model Training' page.")
        return
    
    st.info("Profit curve shows the relationship between revenue and risk at different thresholds.")
    
    y_test = st.session_state['y_test']
    y_proba = st.session_state['y_proba_log']
    
    # Business parameters
    st.subheader("Business Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        revenue_per_loan = st.number_input("Revenue per Approved Loan ($)", min_value=0.0, value=1000.0, step=100.0)
    with col2:
        cost_per_false_positive = st.number_input("Cost per False Positive ($)", min_value=0.0, value=500.0, step=50.0)
    with col3:
        cost_per_false_negative = st.number_input("Cost per False Negative ($)", min_value=0.0, value=2000.0, step=100.0)
    
    # Calculate profit for different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    profits = []
    approvals = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate profit
        total_revenue = tp * revenue_per_loan + (tn + fp) * revenue_per_loan  # All approvals
        total_costs = fp * cost_per_false_positive + fn * cost_per_false_negative
        profit = total_revenue - total_costs
        
        profits.append(profit)
        approvals.append(tp + tn + fp)  # Total approvals
    
    profits = np.array(profits)
    approvals = np.array(approvals)
    
    # Find optimal threshold
    optimal_idx = np.argmax(profits)
    optimal_threshold = thresholds[optimal_idx]
    optimal_profit = profits[optimal_idx]
    
    # Plot profit curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=profits,
        mode='lines',
        name='Profit',
        line=dict(color='#10b981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.2)'
    ))
    fig.add_trace(go.Scatter(
        x=[optimal_threshold, optimal_threshold],
        y=[profits.min(), profits.max()],
        mode='lines',
        name=f'Optimal Threshold ({optimal_threshold:.2f})',
        line=dict(color='#ef4444', width=2, dash='dash')
    ))
    fig.update_layout(
        title='Profit Curve: Revenue vs Risk Trade-off',
        xaxis_title='Classification Threshold',
        yaxis_title='Expected Profit ($)',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig)
    
    # Metrics at optimal threshold
    st.subheader("Optimal Threshold Analysis")
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
    with col2:
        st.metric("Expected Profit", f"${optimal_profit:,.0f}")
    with col3:
        st.metric("Total Approvals", f"{tp + tn + fp:,}")
    with col4:
        st.metric("True Positives", f"{tp:,}")
    
    # Cost breakdown
    st.subheader("Cost Breakdown at Optimal Threshold")
    cost_breakdown = pd.DataFrame({
        'Category': ['Revenue (True Positives)', 'Revenue (Other Approvals)', 
                     'Cost (False Positives)', 'Cost (False Negatives)', 'Net Profit'],
        'Amount ($)': [
            tp * revenue_per_loan,
            (tn + fp) * revenue_per_loan,
            -fp * cost_per_false_positive,
            -fn * cost_per_false_negative,
            optimal_profit
        ]
    })
    cost_breakdown['Amount ($)'] = cost_breakdown['Amount ($)'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(cost_breakdown)
    
    # Comparison with default threshold
    st.subheader("Comparison: Default (0.5) vs Optimal Threshold")
    y_pred_default = (y_proba >= 0.5).astype(int)
    tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, y_pred_default).ravel()
    profit_default = (tp_d + tn_d + fp_d) * revenue_per_loan - fp_d * cost_per_false_positive - fn_d * cost_per_false_negative
    
    comparison_df = pd.DataFrame({
        'Metric': ['Threshold', 'Expected Profit', 'Approvals', 'False Positives', 'False Negatives'],
        'Default (0.5)': [0.5, f"${profit_default:,.0f}", tp_d + tn_d + fp_d, fp_d, fn_d],
        'Optimal': [optimal_threshold, f"${optimal_profit:,.0f}", tp + tn + fp, fp, fn],
        'Improvement': [
            f"{(optimal_threshold - 0.5):.3f}",
            f"${optimal_profit - profit_default:,.0f}",
            f"{(tp + tn + fp) - (tp_d + tn_d + fp_d):,}",
            f"{fp - fp_d:,}",
            f"{fn - fn_d:,}"
        ]
    })
    st.dataframe(comparison_df)

def show_model_registry():
    st.header("Model Registry")
    
    st.info("Track and manage different model versions with their performance metrics.")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # List existing models
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        if model_files:
            st.subheader("Registered Models")
            
            registry_data = []
            for model_file in model_files:
                try:
                    model_path = os.path.join('models', model_file)
                    metadata_path = model_path.replace('.pkl', '_metadata.json')
                    
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        registry_data.append({
                            'Model Name': model_file.replace('.pkl', ''),
                            'Version': metadata.get('version', 'N/A'),
                            'Created': metadata.get('created', 'N/A'),
                            'ROC-AUC': metadata.get('roc_auc', 'N/A'),
                            'Accuracy': metadata.get('accuracy', 'N/A'),
                            'Model Type': metadata.get('model_type', 'N/A')
                        })
                except Exception as e:
                    st.warning(f"Error loading {model_file}: {str(e)}")
            
            if registry_data:
                registry_df = pd.DataFrame(registry_data)
                st.dataframe(registry_df)
            else:
                st.info("No registered models found.")
        else:
            st.info("No models registered yet. Train and save a model to see it here.")
    else:
        st.info("Models directory not found. Train and save a model first.")
    
    # Save current model
    st.subheader("Save Current Model")
    if 'log_model' in st.session_state:
        model_name = st.text_input("Model Name", value=f"logistic_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model_version = st.text_input("Version", value="1.0.0")
        
        if st.button("Save Model to Registry", type="primary"):
            try:
                # Save model
                model_path = os.path.join('models', f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(st.session_state['log_model'], f)
                
                # Save metadata
                y_test = st.session_state['y_test']
                y_pred = st.session_state['y_pred_log']
                y_proba = st.session_state['y_proba_log']
                
                metadata = {
                    'version': model_version,
                    'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model_type': 'Logistic Regression',
                    'roc_auc': float(roc_auc_score(y_test, y_proba)),
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, zero_division=0))
                }
                
                metadata_path = model_path.replace('.pkl', '_metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                st.success(f"Model saved as {model_name}.pkl")
                st.rerun()
            except Exception as e:
                st.error(f"Error saving model: {str(e)}")
    else:
        st.warning("No model trained yet. Please train a model first.")


def show_ensemble_methods(df_model):
    """Ensemble methods page for combining multiple models."""
    st.header("Ensemble Methods")
    
    st.info("Combine multiple models to improve prediction performance using ensemble techniques.")
    
    if 'log_model' not in st.session_state:
        st.warning("Please train models first in the 'Model Training' page.")
        return
    
    # Check which models are available
    available_models = []
    if 'log_model' in st.session_state:
        available_models.append('logistic')
    if 'tree_model' in st.session_state:
        available_models.append('tree')
    if 'rf_model' in st.session_state:
        available_models.append('rf')
    if 'xgb_model' in st.session_state:
        available_models.append('xgb')
    if 'lgb_model' in st.session_state:
        available_models.append('lgb')
    
    if len(available_models) < 2:
        st.warning("Need at least 2 models for ensemble. Please train more models first.")
        return
    
    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    X_train_scaled = st.session_state['X_train_scaled']
    X_test_scaled = st.session_state['X_test_scaled']
    
    tab1, tab2, tab3 = st.tabs(["Voting Classifier", "Stacking", "Blending"])
    
    with tab1:
        st.subheader("Voting Classifier")
        st.write("Combines predictions from multiple models using voting (hard or soft).")
        
        col1, col2 = st.columns(2)
        with col1:
            voting_type = st.selectbox("Voting Type", ["soft", "hard"])
        with col2:
            use_weights = st.checkbox("Use Model Weights", value=False)
        
        if st.button("Train Voting Classifier", type="primary"):
            with st.spinner("Training voting classifier..."):
                from sklearn.ensemble import VotingClassifier
                
                # Get base models dynamically
                estimators = []
                if 'log_model' in st.session_state:
                    estimators.append(('logistic', st.session_state['log_model']))
                if 'tree_model' in st.session_state:
                    estimators.append(('tree', st.session_state['tree_model']))
                if 'rf_model' in st.session_state:
                    estimators.append(('rf', st.session_state['rf_model']))
                if 'xgb_model' in st.session_state:
                    estimators.append(('xgb', st.session_state['xgb_model']))
                if 'lgb_model' in st.session_state:
                    estimators.append(('lgb', st.session_state['lgb_model']))
                
                if len(estimators) < 2:
                    st.error("Need at least 2 models for voting classifier")
                    return
                
                if use_weights:
                    weights = [0.6, 0.4]  # Give more weight to logistic regression
                    voting_clf = VotingClassifier(estimators=estimators, voting=voting_type, weights=weights)
                else:
                    voting_clf = VotingClassifier(estimators=estimators, voting=voting_type)
                
                # Combine predictions from all available models
                if voting_type == "soft":
                    # Get probabilities from all models
                    probas = []
                    weights_list = []
                    
                    if 'log_model' in st.session_state:
                        probas.append(st.session_state['log_model'].predict_proba(X_test_scaled))
                        weights_list.append(0.3 if use_weights else 1.0)
                    if 'tree_model' in st.session_state:
                        probas.append(st.session_state['tree_model'].predict_proba(X_test))
                        weights_list.append(0.2 if use_weights else 1.0)
                    if 'rf_model' in st.session_state:
                        probas.append(st.session_state['rf_model'].predict_proba(X_test))
                        weights_list.append(0.2 if use_weights else 1.0)
                    if 'xgb_model' in st.session_state:
                        probas.append(st.session_state['xgb_model'].predict_proba(X_test))
                        weights_list.append(0.2 if use_weights else 1.0)
                    if 'lgb_model' in st.session_state:
                        probas.append(st.session_state['lgb_model'].predict_proba(X_test))
                        weights_list.append(0.1 if use_weights else 1.0)
                    
                    # Normalize weights
                    if use_weights:
                        total_weight = sum(weights_list)
                        weights_list = [w / total_weight for w in weights_list]
                    
                    # Weighted average
                    y_proba_ensemble = np.zeros((len(X_test), 2))
                    for proba, weight in zip(probas, weights_list):
                        y_proba_ensemble += proba * weight
                    
                    y_pred_ensemble = (y_proba_ensemble[:, 1] >= 0.5).astype(int)
                else:
                    # Hard voting - get predictions from all models
                    preds = []
                    weights_list = []
                    
                    if 'log_model' in st.session_state:
                        preds.append(st.session_state['log_model'].predict(X_test_scaled))
                        weights_list.append(0.3 if use_weights else 1.0)
                    if 'tree_model' in st.session_state:
                        preds.append(st.session_state['tree_model'].predict(X_test))
                        weights_list.append(0.2 if use_weights else 1.0)
                    if 'rf_model' in st.session_state:
                        preds.append(st.session_state['rf_model'].predict(X_test))
                        weights_list.append(0.2 if use_weights else 1.0)
                    if 'xgb_model' in st.session_state:
                        preds.append(st.session_state['xgb_model'].predict(X_test))
                        weights_list.append(0.2 if use_weights else 1.0)
                    if 'lgb_model' in st.session_state:
                        preds.append(st.session_state['lgb_model'].predict(X_test))
                        weights_list.append(0.1 if use_weights else 1.0)
                    
                    # Weighted majority vote
                    if use_weights:
                        total_weight = sum(weights_list)
                        weights_list = [w / total_weight for w in weights_list]
                        weighted_sum = np.zeros(len(X_test))
                        for pred, weight in zip(preds, weights_list):
                            weighted_sum += pred * weight
                        y_pred_ensemble = (weighted_sum >= 0.5).astype(int)
                    else:
                        # Simple majority vote
                        preds_array = np.array(preds).T
                        y_pred_ensemble = (preds_array.sum(axis=1) >= len(preds) / 2).astype(int)
                
                # Calculate metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_ensemble):.4f}")
                with col2:
                    st.metric("Precision", f"{precision_score(y_test, y_pred_ensemble, zero_division=0):.4f}")
                with col3:
                    st.metric("Recall", f"{recall_score(y_test, y_pred_ensemble, zero_division=0):.4f}")
                with col4:
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred_ensemble, zero_division=0):.4f}")
                with col5:
                    if voting_type == "soft":
                        auc = roc_auc_score(y_test, y_proba_ensemble[:, 1])
                        st.metric("ROC-AUC", f"{auc:.4f}")
                    else:
                        st.metric("ROC-AUC", "N/A (hard voting)")
                
                # Store in session state
                st.session_state['voting_ensemble'] = {
                    'type': voting_type,
                    'weights': use_weights,
                    'y_pred': y_pred_ensemble,
                    'y_proba': y_proba_ensemble if voting_type == "soft" else None
                }
                
                st.success("Voting classifier trained successfully!")
    
    with tab2:
        st.subheader("Stacking Classifier")
        st.write("Uses a meta-learner to combine predictions from base models.")
        
        meta_learner = st.selectbox("Meta Learner", ["Logistic Regression", "Random Forest"])
        
        if st.button("Train Stacking Classifier", type="primary"):
            with st.spinner("Training stacking classifier... This may take a few minutes."):
                from sklearn.ensemble import StackingClassifier
                
                # Base models - use trained models if available, otherwise create new ones
                base_estimators = []
                
                if 'log_model' in st.session_state:
                    base_estimators.append(('logistic', st.session_state['log_model']))
                else:
                    base_estimators.append(('logistic', LogisticRegression(max_iter=200, random_state=42, n_jobs=-1)))
                
                if 'rf_model' in st.session_state:
                    base_estimators.append(('rf', st.session_state['rf_model']))
                else:
                    base_estimators.append(('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)))
                
                if 'xgb_model' in st.session_state and XGBOOST_AVAILABLE:
                    base_estimators.append(('xgb', st.session_state['xgb_model']))
                elif XGBOOST_AVAILABLE:
                    base_estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')))
                
                # Meta learner
                if meta_learner == "Logistic Regression":
                    meta_model = LogisticRegression(random_state=42, n_jobs=-1)
                else:
                    meta_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                
                # Create stacking classifier
                stacking_clf = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=meta_model,
                    cv=3,
                    n_jobs=-1
                )
                
                # Fit stacking classifier
                # Note: If using pre-trained models, we need to refit or use predictions
                # For simplicity, we'll fit new base models here
                if len([e for e in base_estimators if e[0] in ['logistic', 'rf', 'xgb']]) > 0:
                    # Use original data for tree-based models, scaled for logistic
                    # Create a wrapper for logistic regression to handle scaling
                    from sklearn.pipeline import Pipeline
                    
                    # Rebuild estimators with proper preprocessing
                    final_estimators = []
                    for name, est in base_estimators:
                        if name == 'logistic':
                            # Logistic needs scaling
                            pipe = Pipeline([('scaler', StandardScaler()), ('model', est)])
                            final_estimators.append((name, pipe))
                        else:
                            final_estimators.append((name, est))
                    
                    stacking_clf = StackingClassifier(
                        estimators=final_estimators,
                        final_estimator=meta_model,
                        cv=3,
                        n_jobs=-1
                    )
                    stacking_clf.fit(X_train, y_train)
                else:
                    stacking_clf.fit(X_train, y_train)
                
                # Predictions
                y_pred_stack = stacking_clf.predict(X_test)
                y_proba_stack = stacking_clf.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_stack):.4f}")
                with col2:
                    st.metric("Precision", f"{precision_score(y_test, y_pred_stack, zero_division=0):.4f}")
                with col3:
                    st.metric("Recall", f"{recall_score(y_test, y_pred_stack, zero_division=0):.4f}")
                with col4:
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred_stack, zero_division=0):.4f}")
                with col5:
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba_stack):.4f}")
                
                # Store in session state
                st.session_state['stacking_ensemble'] = {
                    'meta_learner': meta_learner,
                    'model': stacking_clf,
                    'y_pred': y_pred_stack,
                    'y_proba': y_proba_stack
                }
                
                st.success("Stacking classifier trained successfully!")
                
                # Show comparison with base models
                st.subheader("Comparison with Base Models")
                comparison_data = []
                
                if 'log_model' in st.session_state:
                    comparison_data.append({
                        'Model': 'Logistic Regression',
                        'Accuracy': accuracy_score(y_test, st.session_state['y_pred_log']),
                        'ROC-AUC': roc_auc_score(y_test, st.session_state['y_proba_log']),
                        'F1 Score': f1_score(y_test, st.session_state['y_pred_log'], zero_division=0)
                    })
                
                if 'rf_model' in st.session_state:
                    comparison_data.append({
                        'Model': 'Random Forest',
                        'Accuracy': accuracy_score(y_test, st.session_state['y_pred_rf']),
                        'ROC-AUC': roc_auc_score(y_test, st.session_state['y_proba_rf']),
                        'F1 Score': f1_score(y_test, st.session_state['y_pred_rf'], zero_division=0)
                    })
                
                if 'xgb_model' in st.session_state:
                    comparison_data.append({
                        'Model': 'XGBoost',
                        'Accuracy': accuracy_score(y_test, st.session_state['y_pred_xgb']),
                        'ROC-AUC': roc_auc_score(y_test, st.session_state['y_proba_xgb']),
                        'F1 Score': f1_score(y_test, st.session_state['y_pred_xgb'], zero_division=0)
                    })
                
                comparison_data.append({
                    'Model': 'Stacking',
                    'Accuracy': accuracy_score(y_test, y_pred_stack),
                    'ROC-AUC': roc_auc_score(y_test, y_proba_stack),
                    'F1 Score': f1_score(y_test, y_pred_stack, zero_division=0)
                })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.style.highlight_max(axis=0))
    
    with tab3:
        st.subheader("Blending")
        st.write("Simple weighted average of model predictions.")
        
        col1, col2 = st.columns(2)
        with col1:
            weight_log = st.slider("Logistic Regression Weight", 0.0, 1.0, 0.5, 0.1)
        with col2:
            weight_tree = st.slider("Decision Tree Weight", 0.0, 1.0, 0.5, 0.1)
        
        # Normalize weights
        total_weight = weight_log + weight_tree
        if total_weight > 0:
            weight_log = weight_log / total_weight
            weight_tree = weight_tree / total_weight
        
        if st.button("Create Blended Predictions", type="primary"):
            with st.spinner("Creating blended predictions..."):
                # Get probabilities from available models
                probas = []
                weights = []
                
                if 'log_model' in st.session_state:
                    probas.append(st.session_state['y_proba_log'])
                    weights.append(weight_log)
                if 'tree_model' in st.session_state:
                    probas.append(st.session_state['y_proba_tree'])
                    weights.append(weight_tree)
                if 'rf_model' in st.session_state:
                    probas.append(st.session_state['y_proba_rf'])
                    weights.append(0.2)
                if 'xgb_model' in st.session_state:
                    probas.append(st.session_state['y_proba_xgb'])
                    weights.append(0.2)
                if 'lgb_model' in st.session_state:
                    probas.append(st.session_state['y_proba_lgb'])
                    weights.append(0.1)
                
                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                # Blend probabilities
                y_proba_blend = np.zeros(len(y_test))
                for proba, weight in zip(probas, weights):
                    y_proba_blend += proba * weight
                
                y_pred_blend = (y_proba_blend >= 0.5).astype(int)
                
                # Calculate metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_blend):.4f}")
                with col2:
                    st.metric("Precision", f"{precision_score(y_test, y_pred_blend, zero_division=0):.4f}")
                with col3:
                    st.metric("Recall", f"{recall_score(y_test, y_pred_blend, zero_division=0):.4f}")
                with col4:
                    st.metric("F1 Score", f"{f1_score(y_test, y_pred_blend, zero_division=0):.4f}")
                with col5:
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba_blend):.4f}")
                
                # Store in session state
                # Store weights dict
                weights_dict = {}
                if 'log_model' in st.session_state:
                    weights_dict['logistic'] = weights[0] if len(weights) > 0 else 0
                if 'tree_model' in st.session_state:
                    weights_dict['tree'] = weights[1] if len(weights) > 1 else 0
                
                st.session_state['blended_ensemble'] = {
                    'weights': weights_dict,
                    'y_pred': y_pred_blend,
                    'y_proba': y_proba_blend
                }
                
                st.success("Blended predictions created successfully!")
                
                # Visualize weight impact (if we have at least 2 models)
                if len(probas) >= 2:
                    st.subheader("Weight Impact Visualization")
                    weights_range = np.arange(0, 1.1, 0.1)
                    blend_aucs = []
                    
                    for w in weights_range:
                        # Simple 2-model blend for visualization
                        w1 = w
                        w2 = 1 - w
                        y_proba_temp = w1 * probas[0] + w2 * probas[1]
                        auc_temp = roc_auc_score(y_test, y_proba_temp)
                        blend_aucs.append(auc_temp)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=weights_range,
                    y=blend_aucs,
                    mode='lines+markers',
                    name='ROC-AUC',
                    line=dict(color='#667eea', width=3)
                ))
                fig.update_layout(
                    title='ROC-AUC vs Logistic Regression Weight',
                    xaxis_title='Logistic Regression Weight',
                    yaxis_title='ROC-AUC Score',
                    height=400
                )
                st.plotly_chart(fig)


def show_feature_selection(df_model):
    """Feature selection and importance analysis page."""
    st.header("Feature Selection")
    
    st.info("Analyze and select the most important features for model training.")
    
    if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
        st.warning("Please train models first to enable feature selection analysis.")
        return
    
    X_train = st.session_state['X_train']
    y_train = st.session_state['y_train']
    
    # Try to import feature selection utilities
    FEATURE_SELECTION_AVAILABLE = False
    try:
        from utils.feature_selection import get_feature_importance_scores, select_k_best_features
        FEATURE_SELECTION_AVAILABLE = True
    except (ImportError, Exception) as e:
        st.warning(f"Feature selection utilities not available: {e}")
        st.info("Please ensure utils/feature_selection.py exists and is properly configured.")
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance Scores", "Select Features", "Comparison"])
    
    with tab1:
        st.subheader("Feature Importance Analysis")
        
        if not FEATURE_SELECTION_AVAILABLE:
            st.error("Feature selection utilities are not available. Please check the utils/feature_selection.py file.")
        elif st.button("Calculate Feature Importance", type="primary"):
            with st.spinner("Calculating feature importance scores..."):
                importance_scores = get_feature_importance_scores(X_train, y_train)
                
                st.session_state['feature_importance_scores'] = importance_scores
                
                # Display top features
                st.subheader("Top 20 Most Important Features")
                top_features = importance_scores.head(20)
                
                fig = px.bar(
                    top_features.reset_index(),
                    x='combined_score',
                    y='index',
                    orientation='h',
                    title='Top 20 Features by Combined Importance Score',
                    color='combined_score',
                    color_continuous_scale='Viridis',
                    labels={'index': 'Feature', 'combined_score': 'Importance Score'}
                )
                fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig)
                
                st.dataframe(importance_scores.head(30))
    
    with tab2:
        st.subheader("Select Features for Training")
        
        if 'feature_importance_scores' not in st.session_state:
            st.warning("Please calculate feature importance scores first.")
        else:
            importance_scores = st.session_state['feature_importance_scores']
            
            n_features = st.slider("Number of features to select", 10, min(50, len(X_train.columns)), 20)
            
            if st.button("Select Features", type="primary"):
                selected_features = importance_scores.head(n_features).index.tolist()
                st.session_state['selected_features'] = selected_features
                
                st.success(f"Selected {len(selected_features)} features")
                st.dataframe(pd.DataFrame({'Feature': selected_features}))
    
    with tab3:
        st.subheader("Feature Selection Methods Comparison")
        
        if not FEATURE_SELECTION_AVAILABLE:
            st.error("Feature selection utilities are not available.")
        elif st.button("Compare Selection Methods", type="primary"):
            with st.spinner("Comparing different feature selection methods..."):
                # K-Best
                X_kbest, features_kbest = select_k_best_features(X_train, y_train, k=20)
                
                # RF Importance
                try:
                    from utils.feature_selection import select_features_by_importance
                    X_rf, features_rf = select_features_by_importance(X_train, y_train, n_features=20)
                except (ImportError, Exception) as e:
                    st.warning(f"Random Forest feature selection unavailable: {e}")
                    features_rf = []
                    X_rf = X_train
                
                comparison_df = pd.DataFrame({
                    'Method': ['K-Best (F-test)', 'Random Forest Importance'],
                    'Number of Features': [len(features_kbest), len(features_rf)],
                    'Top 5 Features': [
                        ', '.join(features_kbest[:5]),
                        ', '.join(features_rf[:5])
                    ]
                })
                
                st.dataframe(comparison_df)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        
        st.error(f"""
        **Application Error**
        
        An error occurred while running the app:
        
        ```
        {str(e)}
        ```
        """)
        
        with st.expander("🔍 Full Error Details"):
            st.code(error_trace, language="python")
        
        st.info("""
        **Troubleshooting Steps:**
        1. Check the Streamlit Cloud logs (Manage app → Logs) for detailed error information
        2. Verify all dependencies are installed (check requirements.txt)
        3. Ensure the utils/ directory exists and contains all required files
        4. Try clearing the app cache (Manage app → Clear cache)
        5. Check if the dataset file is accessible (if using cloud storage)
        """)
        
        # Log the error for debugging
        try:
            logger.error(f"App crashed: {str(e)}\n{error_trace}")
        except:
            pass
