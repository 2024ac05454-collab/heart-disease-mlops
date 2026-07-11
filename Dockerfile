# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the MLflow run history
COPY app/ /app/app/
COPY mlruns/ /app/mlruns/

# Expose the port that Uvicorn runs on
EXPOSE 8000

# Command to start the server when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]