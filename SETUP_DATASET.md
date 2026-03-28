# Dataset Setup

The dataset files (`application_train.csv` and `application_test.csv`) are too large for GitHub (183MB total).

## Download Instructions

1. **From Kaggle:**
   - Go to: https://www.kaggle.com/c/home-credit-default-risk/data
   - Download `application_train.csv` and `application_test.csv`
   - Place them in the project root directory

2. **Alternative: Use Sample Data**
   - The app works with a sample of the data
   - Use the "Sample Size" slider in the sidebar for exploration
   - Enable "Use Full Dataset" only when training models

## File Locations

After downloading, your project structure should be:
```
kaggle/
├── application_train.csv  (158MB)
├── application_test.csv   (25MB)
├── app.py
├── api.py
└── ...
```

## For Deployment

For cloud deployments (Streamlit Cloud, Railway, etc.):
- Upload dataset to cloud storage (AWS S3, Google Cloud Storage)
- Update `load_data()` function to load from URL
- Or use a sample dataset for demo purposes

Example:
```python
@st.cache_data
def load_data():
    url = "https://your-bucket.s3.amazonaws.com/application_train.csv"
    return pd.read_csv(url)
```
