"""FastAPI application for insurance renewal prediction"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
from src.api.predictor import Predictor, PredictionRequest, PredictionResponse
from src.utils.config import load_config

# Load config
config = load_config()

# Initialize app
app = FastAPI(
    title="Insurance Renewal Prediction API",
    description="SOTA machine learning API for predicting insurance customer renewal",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
models_dir = Path(config['paths']['models_dir'])
feature_engineer_path = models_dir / 'feature_engineer.pkl'

try:
    predictor = Predictor(models_dir, feature_engineer_path)
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    predictor = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Insurance Renewal Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, model: str = "ensemble"):
    """Predict insurance renewal"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        result = predictor.predict(request, model_name=model)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest], model: str = "ensemble"):
    """Batch prediction"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        results = predictor.predict_batch(requests, model_name=model)
        return {"predictions": [r.dict() for r in results]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models"""
    if predictor is None:
        return {"models": []}
    
    return {"models": list(predictor.models.keys())}


if __name__ == "__main__":
    api_config = config.get('api', {})
    uvicorn.run(
        "app:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', False)
    )
