# Deployment Guide

This guide covers multiple deployment options for the Loan Default Prediction Platform.

## Prerequisites

- Python 3.9+
- Git repository (GitHub recommended)
- Dataset files: `application_train.csv` and `application_test.csv`

---

## Option 1: Streamlit Cloud (Easiest - Recommended)

**Best for:** Quick deployment, free tier, no server management

### Steps:

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/loan-default-prediction.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Add Dataset Files**
   - Streamlit Cloud doesn't support large files by default
   - Options:
     - Use Streamlit Secrets for file paths (if stored elsewhere)
     - Upload to cloud storage (S3, GCS) and load from URL
     - Use a smaller sample dataset for demo

**Note:** Streamlit Cloud has a 1GB file size limit. For large datasets, consider other options.

---

## Option 2: Railway (Recommended for Full Stack)

**Best for:** Full-stack deployment, easy setup, good free tier

### Steps:

1. **Install Railway CLI**
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Create railway.json**
   ```json
   {
     "$schema": "https://railway.app/railway.schema.json",
     "build": {
       "builder": "DOCKERFILE",
       "dockerfilePath": "Dockerfile"
     },
     "deploy": {
       "startCommand": "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0",
       "restartPolicyType": "ON_FAILURE",
       "restartPolicyMaxRetries": 10
     }
   }
   ```

3. **Deploy**
   ```bash
   railway init
   railway up
   ```

4. **Set Environment Variables** (in Railway dashboard)
   - `PORT=8501` (auto-set by Railway)

5. **Add Dataset Files**
   - Upload via Railway dashboard or use volume mounts
   - Or use Railway's file storage

**Cost:** Free tier available, then $5/month

---

## Option 3: Render

**Best for:** Simple deployment, good documentation

### Steps:

1. **Create render.yaml**
   ```yaml
   services:
     - type: web
       name: loan-default-app
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
       envVars:
         - key: PORT
           value: 8501
   ```

2. **Deploy**
   - Connect GitHub repo to Render
   - Render will auto-detect and deploy
   - Or use Render CLI:
     ```bash
     render deploy
     ```

**Cost:** Free tier available, then $7/month

---

## Option 4: Heroku

**Best for:** Traditional PaaS, well-documented

### Steps:

1. **Install Heroku CLI**
   ```bash
   # macOS
   brew tap heroku/brew && brew install heroku
   
   # Or download from heroku.com
   ```

2. **Create/Update Procfile**
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

3. **Create runtime.txt** (optional)
   ```
   python-3.9.18
   ```

4. **Deploy**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

5. **Add Dataset Files**
   - Use Heroku's ephemeral filesystem (temporary)
   - Or use cloud storage (S3 recommended)
   - Or use Heroku add-ons for persistent storage

**Note:** Heroku's free tier was discontinued. Paid plans start at $7/month.

---

## Option 5: Docker (Local/Server)

**Best for:** Full control, self-hosting, production deployments

### Local Deployment

1. **Using Docker Compose (Recommended)**
   ```bash
   # Build and start services
   docker-compose up -d
   
   # View logs
   docker-compose logs -f
   
   # Stop services
   docker-compose down
   ```

2. **Using deploy.sh script**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Manual Docker commands**
   ```bash
   # Build image
   docker build -t loan-default-app .
   
   # Run Streamlit app
   docker run -d \
     -p 8501:8501 \
     --name streamlit-app \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/application_train.csv:/app/application_train.csv:ro \
     loan-default-app
   
   # Run API (separate container)
   docker run -d \
     -p 8000:8000 \
     --name api-app \
     -v $(pwd)/models:/app/models \
     loan-default-app \
     uvicorn api:app --host 0.0.0.0 --port 8000
   ```

### Server Deployment (VPS/Cloud)

1. **On AWS EC2 / DigitalOcean / Linode**
   ```bash
   # SSH into server
   ssh user@your-server-ip
   
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   
   # Clone repository
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
   
   # Deploy
   docker-compose up -d
   ```

2. **Set up Nginx reverse proxy** (optional)
   ```nginx
   # /etc/nginx/sites-available/loan-default
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Set up SSL with Let's Encrypt**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

---

## Option 6: AWS (Advanced)

### Using AWS App Runner

1. **Create apprunner.yaml**
   ```yaml
   version: 1.0
   runtime: python3
   build:
     commands:
       build:
         - pip install -r requirements.txt
   run:
     runtime-version: 3.9
     command: streamlit run app.py --server.port=8000 --server.address=0.0.0.0
     network:
       port: 8000
       env: PORT
   ```

2. **Deploy via AWS Console**
   - Go to AWS App Runner
   - Create new service
   - Connect GitHub repository
   - Configure build settings

### Using AWS ECS/Fargate

1. **Push Docker image to ECR**
   ```bash
   aws ecr create-repository --repository-name loan-default-app
   docker tag loan-default-app:latest your-account.dkr.ecr.region.amazonaws.com/loan-default-app:latest
   aws ecr get-login-password | docker login --username AWS --password-stdin your-account.dkr.ecr.region.amazonaws.com
   docker push your-account.dkr.ecr.region.amazonaws.com/loan-default-app:latest
   ```

2. **Create ECS task definition and service**
   - Use AWS Console or Terraform/CloudFormation

---

## Option 7: Google Cloud Platform

### Using Cloud Run

1. **Deploy with gcloud CLI**
   ```bash
   # Build and push to GCR
   gcloud builds submit --tag gcr.io/PROJECT-ID/loan-default-app
   
   # Deploy to Cloud Run
   gcloud run deploy loan-default-app \
     --image gcr.io/PROJECT-ID/loan-default-app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

---

## Environment Variables

Set these in your deployment platform:

```bash
# Optional - for production
PYTHONUNBUFFERED=1
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0

# For API
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Handling Large Dataset Files

Since `application_train.csv` is large (300K+ rows), consider:

1. **Cloud Storage** (Recommended)
   - Upload to AWS S3, Google Cloud Storage, or Azure Blob
   - Load from URL in app:
     ```python
     @st.cache_data
     def load_data():
         url = "https://your-bucket.s3.amazonaws.com/application_train.csv"
         return pd.read_csv(url)
     ```

2. **Sample Dataset**
   - Create a smaller sample for demo
   - Keep full dataset for local/production use

3. **Database**
   - Load data into PostgreSQL/MySQL
   - Query on-demand (slower but scalable)

---

## Quick Start Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Run API
uvicorn api:app --reload
```

### Docker Local
```bash
docker-compose up -d
```

### Production Checklist
- [ ] Set up environment variables
- [ ] Configure logging
- [ ] Set up monitoring/alerting
- [ ] Configure backups for models
- [ ] Set up CI/CD pipeline
- [ ] Configure domain name and SSL
- [ ] Set up error tracking (Sentry, etc.)

---

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8501
# Kill process
kill -9 <PID>
```

### Docker Build Fails
```bash
# Clear Docker cache
docker system prune -a
# Rebuild
docker-compose build --no-cache
```

### Memory Issues
- Reduce sample size in app
- Use smaller dataset for demo
- Increase server memory allocation

---

## Recommended Deployment Flow

1. **Development:** Local with `streamlit run app.py`
2. **Staging:** Docker Compose on test server
3. **Production:** 
   - **Quick:** Streamlit Cloud or Railway
   - **Scalable:** AWS/GCP with Docker
   - **Enterprise:** Kubernetes on cloud provider

---

## Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| Streamlit Cloud | ✅ Yes | N/A | Quick demos |
| Railway | ✅ Yes | $5/mo | Full-stack apps |
| Render | ✅ Yes | $7/mo | Simple deployments |
| Heroku | ❌ No | $7/mo | Traditional PaaS |
| AWS/GCP | ✅ Limited | Pay-as-you-go | Enterprise |
| Self-hosted | ✅ Yes | Server costs | Full control |

---

## Need Help?

- Check logs: `docker-compose logs -f`
- Streamlit docs: [docs.streamlit.io](https://docs.streamlit.io)
- Railway docs: [docs.railway.app](https://docs.railway.app)
- Render docs: [render.com/docs](https://render.com/docs)
