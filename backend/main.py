#!/usr/bin/env python3
"""
soil-crop-recommender-ui - FastAPI Backend
===========================================
Production API serving the crop recommendation model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from pathlib import Path
from contextlib import asynccontextmanager

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH = Path(__file__).parent.parent / "data" / "crop_recommender_model.pkl"

# Global model variable (loaded once at startup)
model_data = None


# =============================================================================
# LIFESPAN - Load model at startup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup on shutdown."""
    global model_data
    
    print("🚀 Starting Crop Recommender API...")
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    model_data = joblib.load(MODEL_PATH)
    print(f"   ✓ Model loaded: {len(model_data['classes'])} crop classes")
    print(f"   ✓ Features: {model_data['features']}")
    print(f"   ✓ Model accuracy: {model_data['accuracy']:.2%}")
    print("✅ API ready!")
    
    yield
    
    print("👋 Shutting down Crop Recommender API...")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Crop Recommender API",
    description="Crop Recommendation System powered by Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
# Read allowed origins from environment variable (comma-separated string)
# Default to localhost for development
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SCHEMAS
# =============================================================================

class PredictionInput(BaseModel):
    """Input schema for crop prediction."""
    temp: float = Field(..., description="Annual Mean Temperature (°C)", ge=-50, le=50)
    rain: float = Field(..., description="Annual Total Precipitation (mm)", ge=0, le=20000)
    ph: float = Field(..., description="Soil pH", ge=3.0, le=11.0)
    clay: float = Field(..., description="Soil Clay Content (%)", ge=0, le=100)
    sand: float = Field(..., description="Soil Sand Content (%)", ge=0, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "temp": 18.5,
                "rain": 850.0,
                "ph": 6.5,
                "clay": 25.0,
                "sand": 40.0
            }
        }


class CropPrediction(BaseModel):
    """Single crop prediction with probability."""
    name: str
    prob: float


class PredictionOutput(BaseModel):
    """Output schema for crop prediction."""
    winner: str
    top_5: list[CropPrediction]


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Crop Recommender API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "n_classes": len(model_data['classes']) if model_data else 0,
        "category_names": model_data.get('category_names', {}) if model_data else {}
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict the best crop categories for given environmental conditions.
    
    Returns the winning category and top 5 recommendations with probabilities.
    """
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model = model_data['model']
    classes = model_data['classes']
    category_names = model_data.get('category_names', {})
    
    # Prepare features in correct order
    features = np.array([[
        input_data.temp,
        input_data.rain,
        input_data.ph,
        input_data.clay,
        input_data.sand
    ]])
    
    # Get prediction probabilities
    probabilities = model.predict_proba(features)[0]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(probabilities)[-5:][::-1]
    
    top_5 = [
        CropPrediction(
            name=classes[idx],
            prob=round(float(probabilities[idx]), 4)
        )
        for idx in top_5_indices
    ]
    
    # Winner is the top prediction
    winner = top_5[0].name
    
    return PredictionOutput(winner=winner, top_5=top_5)


@app.get("/categories")
async def get_categories():
    """Get list of all crop categories with their display names."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    category_names = model_data.get('category_names', {})
    
    return {
        "categories": sorted(model_data['classes']),
        "category_names": category_names,
        "count": len(model_data['classes'])
    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
