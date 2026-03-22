"""
FastAPI REST API for Loan Default Prediction
Production-ready API endpoint for model predictions
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
import os
import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager

# Setup logging
from utils.logger import setup_logger
logger = setup_logger("api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting Loan Default Prediction API")
    yield
    # Shutdown
    logger.info("Shutting down Loan Default Prediction API")


app = FastAPI(
    title="Loan Default Prediction API",
    description="REST API for predicting loan default risk. Supports single and batch predictions with comprehensive validation.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests and response times."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Load model and scaler (in production, load from model registry)
MODEL_PATH = "models"
MODEL = None
SCALER = None

def load_model():
    """Load the latest model from registry"""
    global MODEL, SCALER
    if os.path.exists(MODEL_PATH):
        model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pkl')]
        if model_files:
            # Load the most recent model
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(MODEL_PATH, latest_model)
            with open(model_path, 'rb') as f:
                MODEL = pickle.load(f)
            # In production, load scaler separately
            # For now, assume it's available in session state or saved separately

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Loading models and initializing API...")
    load_model()
    logger.info("API startup complete")

# Request/Response models
class PredictionRequest(BaseModel):
    """Single prediction request with validation"""
    amt_income_total: float = Field(..., ge=0, description="Total income (must be positive)")
    amt_credit: float = Field(..., ge=0, description="Credit amount (must be positive)")
    amt_annuity: float = Field(..., ge=0, description="Annuity amount")
    cnt_children: int = Field(0, ge=0, le=20, description="Number of children")
    ext_source_2: Optional[float] = Field(None, ge=0.0, le=1.0, description="External source 2 score")
    ext_source_3: Optional[float] = Field(None, ge=0.0, le=1.0, description="External source 3 score")
    age: float = Field(..., ge=18, le=100, description="Age in years")
    employ_years: float = Field(0.0, ge=0, le=50, description="Years of employment")
    
    @validator('amt_income_total', 'amt_credit')
    def validate_positive(cls, v, field):
        if v <= 0:
            raise ValueError(f'{field.name} must be positive')
        return v

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    records: List[PredictionRequest]

class PredictionResponse(BaseModel):
    """Prediction response"""
    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str
    prediction_timestamp: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_records: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Loan Default Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API health status and model availability.
    """
    try:
        health_status = {
            "status": "healthy",
            "model_loaded": MODEL is not None,
            "scaler_loaded": SCALER is not None,
            "timestamp": datetime.now().isoformat()
        }
        logger.debug("Health check requested")
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Predict default risk for a single applicant.
    
    **Example Request:**
    ```json
    {
        "amt_income_total": 200000.0,
        "amt_credit": 400000.0,
        "amt_annuity": 25000.0,
        "cnt_children": 0,
        "ext_source_2": 0.5,
        "ext_source_3": 0.5,
        "age": 35.0,
        "employ_years": 5.0
    }
    ```
    
    **Returns:**
    - `default_probability`: Probability of default (0-1)
    - `risk_level`: Risk category (Low/Medium/High)
    - `prediction_timestamp`: When prediction was made
    """
    start_time = time.time()
    
    if MODEL is None or SCALER is None:
        logger.error("Model or scaler not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Please train and save a model first.")
    
    try:
        logger.info(f"Processing prediction request for applicant")
        
        # Validate input
        if request.amt_income_total <= 0 or request.amt_credit <= 0:
            raise HTTPException(status_code=400, detail="Income and credit must be positive")
        
        # Prepare input (simplified - in production, use full preprocessing)
        # Note: This is a simplified version. Full implementation would use complete preprocessing pipeline.
        input_data = np.array([[
            request.amt_income_total,
            request.amt_credit,
            request.amt_annuity,
            request.cnt_children,
            request.ext_source_2 or 0.5,
            request.ext_source_3 or 0.5,
            request.age,
            request.employ_years
        ]])
        
        # Scale and predict
        input_scaled = SCALER.transform(input_data)
        probability = float(MODEL.predict_proba(input_scaled)[0, 1])
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        elapsed_time = time.time() - start_time
        logger.info(f"Prediction completed in {elapsed_time:.3f}s - Probability: {probability:.3f}, Risk: {risk_level}")
        
        return {
            "default_probability": probability,
            "risk_level": risk_level,
            "prediction_timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict default risk for multiple applicants in batch.
    
    **Example Request:**
    ```json
    {
        "records": [
            {
                "amt_income_total": 200000.0,
                "amt_credit": 400000.0,
                "amt_annuity": 25000.0,
                "cnt_children": 0,
                "ext_source_2": 0.5,
                "ext_source_3": 0.5,
                "age": 35.0,
                "employ_years": 5.0
            }
        ]
    }
    ```
    
    **Returns:**
    - `predictions`: List of prediction results
    - `total_records`: Total number of predictions made
    """
    start_time = time.time()
    
    if MODEL is None or SCALER is None:
        logger.error("Model or scaler not loaded for batch prediction")
        raise HTTPException(status_code=503, detail="Model not loaded. Please train and save a model first.")
    
    try:
        logger.info(f"Processing batch prediction for {len(request.records)} records")
        
        if len(request.records) > 10000:
            raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 10,000 records")
        
        predictions = []
        for i, record in enumerate(request.records):
            try:
                # Validate and prepare input
                pred_request = PredictionRequest(**record)
                
                input_data = np.array([[
                    pred_request.amt_income_total,
                    pred_request.amt_credit,
                    pred_request.amt_annuity,
                    pred_request.cnt_children,
                    pred_request.ext_source_2 or 0.5,
                    pred_request.ext_source_3 or 0.5,
                    pred_request.age,
                    pred_request.employ_years
                ]])
                
                # Scale and predict
                input_scaled = SCALER.transform(input_data)
                probability = float(MODEL.predict_proba(input_scaled)[0, 1])
                
                # Determine risk level
                if probability < 0.3:
                    risk_level = "Low"
                elif probability < 0.7:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                predictions.append({
                    "default_probability": probability,
                    "risk_level": risk_level,
                    "prediction_timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Error processing record {i}: {str(e)}")
                # Continue with other records
                continue
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch prediction completed in {elapsed_time:.3f}s for {len(predictions)} records")
        
        return {
            "predictions": predictions,
            "total_records": len(predictions)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get comprehensive information about the loaded model.
    
    Returns model type, version, performance metrics, and feature information.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_path, _ = find_latest_model()
    metadata_info = {}
    
    if model_path:
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata_info = json.load(f)
    
    return {
        "model_type": type(MODEL).__name__,
        "model_loaded": True,
        "scaler_loaded": SCALER is not None,
        "imputer_loaded": IMPUTER is not None,
        "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        "features": list(FEATURE_NAMES) if FEATURE_NAMES else [],
        "performance": {
            "roc_auc": metadata_info.get('roc_auc', 'N/A'),
            "accuracy": metadata_info.get('accuracy', 'N/A'),
            "precision": metadata_info.get('precision', 'N/A'),
            "recall": metadata_info.get('recall', 'N/A'),
            "f1": metadata_info.get('f1', 'N/A')
        } if metadata_info else {},
        "version": metadata_info.get('version', 'N/A'),
        "created": metadata_info.get('created', 'N/A')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
