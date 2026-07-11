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

# Dynamic discovery of the latest experiment and run model path
print("Configuring MLflow tracking and searching for active model artifacts...")
tracking_uri = pathlib.Path("mlruns").resolve().as_uri()
mlflow.set_tracking_uri(tracking_uri)

# Fallback initialization structure
model = None
model_name = "Unknown"

# Dynamically locate the experiment folder within mlruns
experiment_dirs = glob.glob(os.path.join("mlruns", "[0-9]*"))
if experiment_dirs:
    # Scan all runs inside experiment subdirectories
    run_dirs = []
    for exp in experiment_dirs:
        run_dirs.extend(glob.glob(os.path.join(exp, "*")))
    
    # Exclude non-directory entries or special folders
    valid_runs = [r for r in run_dirs if os.path.isdir(r) and os.path.exists(os.path.join(r, "artifacts", "model"))]
    
    if valid_runs:
        # Retrieve the single latest modified run directory
        latest_run_dir = max(valid_runs, key=os.path.getctime)
        model_name = os.path.basename(latest_run_dir)
        model_path = os.path.join(latest_run_dir, "artifacts", "model")
        MODEL_URI = pathlib.Path(model_path).resolve().as_uri()
        
        print(f"Loading latest active model path from URI: {MODEL_URI}")
        model = mlflow.sklearn.load_model(MODEL_URI)

if model is None:
    print("Warning: Active registry models not found. Ensure train.py has run successfully.")

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
    confidence = probabilities[0][1] # Probability score for class 1
    
    return {
        "prediction": int(prediction),
        "confidence_score": float(confidence),
        "status": "success",
        "model_version_used": model_name
    }

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running!"}