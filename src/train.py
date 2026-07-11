import os
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from ucimlrepo import fetch_ucirepo 

# 1. Force the env var inside Python before MLflow loads
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

print("Fetching Heart Disease dataset from UCI...")
heart_disease = fetch_ucirepo(id=45) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
y = y['num'].apply(lambda val: 1 if val > 0 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Common preprocessing pipeline
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Models and Hyperparameter grids for GridSearchCV
model_definitions = {
    "Logistic_Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0]
        }
    },
    "Random_Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 5, 10]
        }
    }
}

# Setup cross-platform pathing for tracking
tracking_uri = pathlib.Path("mlruns").resolve().as_uri()
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Heart_Disease_V3")

os.makedirs("outputs", exist_ok=True)

print(f"Tracking URI set to: {tracking_uri}")
print("Starting tuned model training and MLflow tracking...")

for model_name, config in model_definitions.items():
    with mlflow.start_run(run_name=model_name):
        # Create pipeline matching the architecture
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', config["model"])
        ])
        
        # Hyperparameter tuning setup
        grid_search = GridSearchCV(full_pipeline, param_grid=config["params"], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]
        
        # Compute performance metrics
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)
        
        # Log params and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("best_params", str(grid_search.best_params_))
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc})
        
        # Generate and save Confusion Matrix plot
        cm = confusion_matrix(y_test, predictions)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
        fig, ax = plt.subplots(figsize=(6, 6))
        cm_display.plot(ax=ax, cmap="Blues")
        cm_path = f"outputs/{model_name}_confusion_matrix.png"
        plt.title(f"{model_name} Confusion Matrix")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # Generate and save ROC Curve plot
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        roc_path = f"outputs/{model_name}_roc_curve.png"
        plt.savefig(roc_path)
        plt.close()
        mlflow.log_artifact(roc_path)
        
        # Log model artifact safely
        mlflow.sklearn.log_model(best_model, "model", serialization_format="cloudpickle")
        print(f"--> Tuned {model_name} logged. Best Accuracy: {acc:.4f}")