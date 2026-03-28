# Streamlit Cloud Deployment Setup

## Current Status

Your app is deployed to Streamlit Cloud! 🎉

**Quick Start:** Enable "Use Demo Mode" in the sidebar to test all features with synthetic data.

For production use with real data, follow the steps below.

## Option 1: Use Demo Mode (Quick Start)

The app now includes a "Demo Mode" that creates sample data if the dataset is missing. This allows you to:
- Test the app functionality
- Show the UI and features
- Demonstrate the workflow

**Note:** Demo mode uses synthetic data, so model performance won't reflect real results.

## Option 2: Add Dataset via Cloud Storage (Recommended for Production)

### Step 1: Upload Dataset to Cloud Storage

**AWS S3:**
1. Create an S3 bucket
2. Upload `application_train.csv` (make it publicly readable)
3. Copy the public URL (e.g., `https://your-bucket.s3.amazonaws.com/application_train.csv`)

**Google Cloud Storage:**
1. Create a GCS bucket
2. Upload the file
3. Make it publicly accessible
4. Copy the URL

**Alternative:** Use a file hosting service like Dropbox, Google Drive (with direct link), etc.

### Step 2: Add to Streamlit Cloud Secrets

1. Go to your Streamlit Cloud app: https://share.streamlit.io
2. Click on your app
3. Click the "⋮" (three dots) menu → **Settings**
4. Click **Secrets** in the sidebar
5. Add this configuration:

```toml
[secrets]
dataset_url = "https://your-bucket.s3.amazonaws.com/application_train.csv"
```

6. Click **Save**
7. The app will automatically redeploy

### Step 3: Verify

After redeployment, the app should load the dataset from the URL automatically.

## Option 3: Use Git LFS (For Smaller Datasets)

If you want to include the dataset in the repository:

1. Install Git LFS:
   ```bash
   git lfs install
   ```

2. Track CSV files:
   ```bash
   git lfs track "*.csv"
   ```

3. Add and commit:
   ```bash
   git add .gitattributes
   git add application_train.csv application_test.csv
   git commit -m "Add dataset files with Git LFS"
   git push origin main
   ```

**Note:** Git LFS has storage limits on free accounts. For 183MB of data, cloud storage is better.

## Troubleshooting

### App Shows "Dataset file not found"

**Solution:** 
- Use Demo Mode button (for testing)
- Or add dataset URL to Streamlit Secrets (for production)

### App Crashes on Load

**Check:**
1. All dependencies in `requirements.txt` are correct
2. Python version is 3.9+ (Streamlit Cloud uses 3.9 by default)
3. No syntax errors in `app.py`

### Slow Loading

**Causes:**
- Large dataset (300K+ rows)
- Network latency from cloud storage

**Solutions:**
- Use sampling in the app (already implemented)
- Consider using a smaller sample dataset for demo
- Use faster cloud storage (S3, GCS)

## Quick Commands

```bash
# Check if secrets are set (locally)
cat .streamlit/secrets.toml

# Test locally with secrets
streamlit run app.py

# Push updates to trigger redeploy
git push origin main
```

## Next Steps

1. ✅ App is deployed
2. ⏳ Add dataset URL to Streamlit Secrets (or use Demo Mode)
3. ✅ Test the app functionality
4. ✅ Share your deployed app URL!

Your app URL will be: `https://your-app-name.streamlit.app`
