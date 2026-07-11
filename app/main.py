import os
import pathlib
import glob
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
import time
import logging
from prometheus_fastapi_instrumentator import Instrumentator

# Set up logging environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

app = FastAPI(title="Heart Disease Prediction API", version="1.0")

# Prometheus instrumentation setup
Instrumentator().instrument(app).expose(app)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Duration: {process_time:.4f}s")
    return response

# --- FIXED MODEL DISCOVERY ---
print("Configuring MLflow tracking and searching for active model artifacts...")

# 1. Use absolute path to ensure Jenkins finds the 'mlruns' folder
workspace_path = os.getcwd()
mlruns_path = os.path.join(workspace_path, "mlruns")
tracking_uri = pathlib.Path(mlruns_path).resolve().as_uri()
mlflow.set_tracking_uri(tracking_uri)

model = None
model_name = "Unknown"

# 2. Use a 'wildcard' search that ignores specific naming patterns
# This looks for 'model' artifacts deep inside the mlruns folder tree
model_search_path = os.path.join(mlruns_path, "*", "*", "artifacts", "model")
valid_model_paths = glob.glob(model_search_path)

print(f"DEBUG: Searching for models in {model_search_path}")
print(f"DEBUG: Found these paths: {valid_model_paths}")

if valid_model_paths:
    # Get the latest modified one
    latest_model_path = max(valid_model_paths, key=os.path.getctime)
    
    # Extract the run folder name (two levels up from 'model')
    latest_run_dir = os.path.dirname(os.path.dirname(latest_model_path))
    model_name = os.path.basename(latest_run_dir)
    
    MODEL_URI = pathlib.Path(latest_model_path).resolve().as_uri()
    print(f"Loading latest active model from: {MODEL_URI}")
    model = mlflow.sklearn.load_model(MODEL_URI)

if model is None:
    print("Warning: Active registry models not found in mlruns!")

class PatientData(BaseModel):
    age: float; sex: float; cp: float; trestbps: float; chol: float
    fbs: float; restecg: float; thalach: float; exang: float; oldpeak: float
    slope: float; ca: float; thal: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        return {"status": "error", "message": "Model engine has not loaded properly."}
        
    input_data = pd.DataFrame([data.model_dump()]) 
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)
    confidence = probabilities[0][1] 
    
    return {
        "prediction": int(prediction),
        "confidence_score": float(confidence),
        "status": "success",
        "model_version_used": model_name
    }

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running!"}