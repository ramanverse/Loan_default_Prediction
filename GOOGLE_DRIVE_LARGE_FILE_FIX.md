# Fix: Google Drive Large File Download Issue

## Problem

Google Drive blocks direct downloads of files larger than 100MB with a virus scan warning page. When the app tries to download `application_train.csv` (158MB), it gets an HTML page instead of the CSV file.

## Solution Options

### Option 1: Use Dropbox (Recommended - Easiest)

**Why Dropbox?**
- Direct download links work reliably
- No file size restrictions for direct links
- Simple setup

**Steps:**
1. Upload `application_train.csv` to Dropbox
2. Right-click the file → Share → Create link
3. Get the direct download link:
   - Change `www.dropbox.com` to `dl.dropboxusercontent.com`
   - Remove `?dl=0` and add `?dl=1` at the end
   - Example: `https://dl.dropboxusercontent.com/s/FILE_ID/application_train.csv?dl=1`
4. Update Streamlit Secrets:
   ```toml
   [secrets]
   dataset_url = "https://dl.dropboxusercontent.com/s/YOUR_FILE_ID/application_train.csv?dl=1"
   ```

### Option 2: Use AWS S3 (Best for Production)

**Steps:**
1. Create an S3 bucket
2. Upload `application_train.csv` to the bucket
3. Make it publicly readable (or use a signed URL)
4. Get the public URL: `https://BUCKET_NAME.s3.REGION.amazonaws.com/application_train.csv`
5. Update Streamlit Secrets with the S3 URL

### Option 3: Use GitHub Releases

**Steps:**
1. Create a new release in your GitHub repo
2. Upload `application_train.csv` as a release asset
3. Get the direct download URL from the release
4. Update Streamlit Secrets

### Option 4: Compress the File

**Steps:**
1. Compress `application_train.csv` to `.zip` or `.gz`
2. Upload compressed file to Google Drive (should be <100MB)
3. Update the app to decompress after download

## Current Status

The app now:
- ✅ Detects Google Drive virus scan warnings
- ✅ Attempts to use confirmed download URLs
- ✅ Shows clear error messages with solutions
- ✅ Provides alternative hosting recommendations

## Quick Fix: Use Dropbox

1. **Upload to Dropbox:**
   - Go to https://www.dropbox.com
   - Upload `application_train.csv`
   - Right-click → Share → Copy link

2. **Convert to Direct Download:**
   - Original link: `https://www.dropbox.com/s/abc123/file.csv?dl=0`
   - Direct link: `https://dl.dropboxusercontent.com/s/abc123/file.csv?dl=1`

3. **Update Streamlit Secrets:**
   ```toml
   [secrets]
   dataset_url = "https://dl.dropboxusercontent.com/s/YOUR_FILE_ID/application_train.csv?dl=1"
   ```

4. **Save and wait for redeployment**

The app will now load the dataset directly from Dropbox without any issues!
