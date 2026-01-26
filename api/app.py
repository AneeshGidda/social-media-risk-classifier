import sys
import os
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

from fastapi import FastAPI, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

# Import model, tokenizer, and label_map from src/infer.py
from src.infer import model, tokenizer, label_map, device

app = FastAPI(
    title="Social Media Risk Classifier API",
    description="API for classifying social media content risk levels using DistilBERT",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    text: str
    prob_not_suicide: float
    prob_potential_suicide: float


@app.get("/")
async def root():
    return {
        "message": "Social Media Risk Classifier API",
        "endpoints": {
            "/health": "GET - Health check endpoint",
            "/predict": "POST - Classify social media content risk level",
            "/docs": "GET - API documentation (Swagger UI)",
            "/redoc": "GET - Alternative API documentation"
        }
    }


@app.get("/health")
async def health(response: Response):
    """
    Health check endpoint to verify the API is running and the model is loaded.
    
    Returns:
        - 200 OK: API is healthy and model is loaded
        - 503 Service Unavailable: API is unhealthy (model not loaded or error)
    """
    try:
        # Check if model and tokenizer are loaded
        model_loaded = model is not None
        tokenizer_loaded = tokenizer is not None
        
        if not model_loaded or not tokenizer_loaded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {
                "status": "unhealthy",
                "model_loaded": model_loaded,
                "tokenizer_loaded": tokenizer_loaded
            }
        
        return {
            "status": "healthy",
            "model_loaded": model_loaded,
            "tokenizer_loaded": tokenizer_loaded,
            "device": str(device)
        }
    except Exception as e:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """
    Classify the risk level of the input social media text.
    
    Args:
        input_data: JSON object with 'text' field containing the text to analyze
    
    Returns:
        JSON object with:
        - label: Prediction label ("Not Suicide post" or "Potential Suicide post")
        - confidence: Confidence score (0.0 to 1.0)
        - text: The input text that was analyzed
    """
    try:
        if not input_data.text or not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        text = input_data.text.strip()
        
        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass (inference only)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
        
        # Get probabilities for both classes
        prob_not_suicide = probs[0][0].item()
        prob_potential_suicide = probs[0][1].item()
        
        label = label_map[pred]
        confidence = probs[0][pred].item()
        
        return PredictionResponse(
            label=label,
            confidence=confidence,
            text=text,
            prob_not_suicide=prob_not_suicide,
            prob_potential_suicide=prob_potential_suicide
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
