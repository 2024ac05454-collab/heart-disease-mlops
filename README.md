End-to-End MLOps Pipeline: Heart Disease Prediction
Author: [Your Name/ID]
Repository: [Link to your GitHub repo]
1. Setup & Installation Instructions
Prerequisites
Docker Desktop (with Kubernetes enabled)
Python 3.9+
Git
kubectl
Environment Setup
# Clone the repository
git clone [Your-Repo-Link]
cd heart-disease-mlops

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


ML Pipeline Execution
Run the training script to generate MLflow artifacts:
python src/train.py


Containerization & Kubernetes Deployment
Build the image and deploy to local Kubernetes:
docker build --no-cache -t heart-disease-api:latest .
kubectl apply -f k8s/deployment.yaml
kubectl port-forward svc/heart-disease-service 8080:80


Test the deployment using the LoadBalancer endpoint:
curl -X 'POST' 'http://localhost:8080/predict' \
  -H 'accept: application/json' -H 'Content-Type: application/json' \
  -d '{"age": 50, "sex": 1, "cp": 1, "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 2}'


2. Exploratory Data Analysis (EDA) & Modeling Choices
EDA Summary: [Write 2-3 sentences about what you found in the data, e.g., handled missing values, scaled numerical features using StandardScaler, encoded categorical variables].
Visualizations: [Insert screenshots of your histograms/correlation heatmaps here]
Modeling: Trained Logistic Regression and Random Forest models. [Specify which model was chosen] was selected based on [mention your metric, e.g., highest ROC-AUC and accuracy].

 **EDA Summary:** The UCI Heart Disease dataset was analyzed to check for missing values, which were handled by dropping incomplete rows. A class balance check revealed a relatively balanced distribution between healthy and diseased patients. The correlation heatmap indicated that `cp` (chest pain type) and `thalach` (maximum heart rate) have the strongest positive correlation with the target, while `age` and `chol` showed weaker correlations. Numerical features were scaled using `StandardScaler` to prepare for model training.
3. Experiment Tracking
MLflow was used to track parameters, metrics, and models across multiple runs.
[Insert screenshot of MLflow UI showing your experiment runs here]
4. Architecture Diagram
The pipeline flows from data acquisition to model registry, followed by containerization (Docker) and local orchestration (Kubernetes).
[Insert your architecture diagram/screenshot here]
5. CI/CD & Monitoring
CI/CD: GitHub Actions automatically tests the FastAPI endpoints using pytest and builds the Docker image on every push.
[Insert screenshot of successful GitHub Actions workflow here]
Monitoring: The API includes a custom middleware for logging request duration and status codes, plus a /health endpoint for uptime monitoring.
[Insert screenshot of terminal logs showing a 200 OK request through the middleware here]
