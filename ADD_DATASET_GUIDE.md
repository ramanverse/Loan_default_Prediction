# Step-by-Step Guide: Adding Real Dataset to Streamlit Cloud

## Overview

Since the dataset files are too large for GitHub (183MB total), we'll upload them to cloud storage and load them via URL in the app.

## Option 1: Google Drive (Easiest - Free)

### Step 1: Upload Dataset to Google Drive

1. **Upload the file:**
   - Go to [drive.google.com](https://drive.google.com)
   - Click "New" → "File upload"
   - Upload `application_train.csv` (158MB)

2. **Get a shareable link:**
   - Right-click the file → "Share"
   - Change access to "Anyone with the link"
   - Click "Copy link"
   - The link will look like: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`

3. **Convert to direct download link:**
   - Replace `/view?usp=sharing` with `/uc?export=download&id=FILE_ID`
   - Example: `https://drive.google.com/uc?export=download&id=YOUR_FILE_ID`
   - Or use this format: `https://drive.google.com/uc?id=YOUR_FILE_ID`

### Step 2: Add to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click on your app
3. Click "⋮" (three dots) → **Settings**
4. Click **Secrets** in the left sidebar
5. Add this configuration:

```toml
[secrets]
dataset_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
```

6. Click **Save**
7. The app will automatically redeploy (takes 1-2 minutes)

### Step 3: Verify

1. Wait for redeployment to complete
2. Open your app
3. Uncheck "Use Demo Mode" in the sidebar
4. The app should load the real dataset from Google Drive

---

## Option 2: Dropbox (Free & Easy)

### Step 1: Upload to Dropbox

1. **Upload the file:**
   - Go to [dropbox.com](https://dropbox.com)
   - Upload `application_train.csv` to your Dropbox

2. **Get a direct link:**
   - Right-click the file → "Copy link"
   - The link will look like: `https://www.dropbox.com/s/xxxxx/file.csv?dl=0`

3. **Convert to direct download:**
   - Change `?dl=0` to `?dl=1`
   - Final URL: `https://www.dropbox.com/s/xxxxx/application_train.csv?dl=1`

### Step 2: Add to Streamlit Cloud

Same as Option 1, Step 2, but use the Dropbox URL:

```toml
[secrets]
dataset_url = "https://www.dropbox.com/s/xxxxx/application_train.csv?dl=1"
```

---

## Option 3: AWS S3 (Most Reliable - Requires AWS Account)

### Step 1: Create S3 Bucket

1. **Sign in to AWS:**
   - Go to [aws.amazon.com](https://aws.amazon.com)
   - Sign in or create account (free tier available)

2. **Create bucket:**
   - Go to S3 service
   - Click "Create bucket"
   - Name: `loan-default-dataset` (or your choice)
   - Region: Choose closest to you
   - Uncheck "Block all public access" (we need public access)
   - Click "Create bucket"

3. **Upload file:**
   - Click on your bucket
   - Click "Upload"
   - Add `application_train.csv`
   - Click "Upload"

4. **Make file public:**
   - Click on the uploaded file
   - Go to "Permissions" tab
   - Under "Object URL", copy the URL
   - Or manually set: Click "Edit" on "Block public access"
   - Set "Read" permission to "Everyone"

5. **Get public URL:**
   - The URL will be: `https://your-bucket.s3.region.amazonaws.com/application_train.csv`
   - Or use the Object URL from the file properties

### Step 2: Add to Streamlit Cloud

Same as Option 1, Step 2, but use the S3 URL:

```toml
[secrets]
dataset_url = "https://your-bucket.s3.us-east-1.amazonaws.com/application_train.csv"
```

---

## Option 4: GitHub Releases (For Smaller Files)

**Note:** This only works if you split the file or use Git LFS (has limits)

### Using Git LFS:

```bash
# Install Git LFS
git lfs install

# Track CSV files
git lfs track "*.csv"

# Add and commit
git add .gitattributes
git add application_train.csv
git commit -m "Add dataset with Git LFS"
git push origin main
```

**Limitation:** Free GitHub accounts have 1GB LFS storage, so this works for smaller files.

---

## Quick Reference: Streamlit Secrets Format

In Streamlit Cloud → Settings → Secrets, add:

```toml
[secrets]
dataset_url = "YOUR_DATASET_URL_HERE"
```

**Important:** 
- URL must be publicly accessible
- File must be downloadable (not require authentication)
- Use direct download links (not preview links)

---

## Testing Your Setup

### After Adding Secrets:

1. **Wait for redeployment** (1-2 minutes)
2. **Open your app**
3. **Uncheck "Use Demo Mode"** in sidebar
4. **Check the Overview page** - should show real data statistics
5. **Verify data:**
   - Should show ~307K rows (if full dataset)
   - Default rate should be ~8%
   - All features should be present

### Troubleshooting:

**If dataset doesn't load:**

1. **Check the URL:**
   - Test the URL in a browser - it should download the file
   - Make sure it's a direct download link, not a preview

2. **Check Streamlit Secrets:**
   - Go to Settings → Secrets
   - Verify the URL is correct
   - Make sure it's under `[secrets]` section

3. **Check logs:**
   - Go to "Manage app" → "Logs"
   - Look for error messages about loading the dataset

4. **Common issues:**
   - **Google Drive:** Make sure link is converted to direct download format
   - **Dropbox:** Must have `?dl=1` at the end
   - **S3:** File must be publicly readable
   - **CORS errors:** Some hosts block direct access - try a different service

---

## Recommended: Google Drive (Easiest)

**Why Google Drive:**
- ✅ Free
- ✅ Easy to set up
- ✅ Reliable
- ✅ No account limits for basic use

**Steps:**
1. Upload to Google Drive
2. Share → Anyone with link
3. Get file ID from URL
4. Use format: `https://drive.google.com/uc?export=download&id=FILE_ID`
5. Add to Streamlit Secrets

---

## Alternative: Use Demo Mode

If you don't want to set up cloud storage right now:
- Demo mode works perfectly for showcasing the app
- Shows all features and functionality
- No setup required
- Great for portfolio/resume

---

## Need Help?

- **Google Drive not working?** Try Dropbox (Option 2)
- **URL not downloading?** Make sure it's a direct download link
- **Still having issues?** Check Streamlit Cloud logs for specific errors
