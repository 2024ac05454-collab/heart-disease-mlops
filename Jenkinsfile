pipeline {
    agent any

    stages {
        stage('Install Dependencies') {
            steps {
                sh 'python -m pip install --upgrade pip'
                sh 'pip install flake8 pytest'
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('Code Linting') {
            steps {
                // Fails the build if Python syntax errors are found
                sh 'flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics'
            }
        }
        
        stage('Model Training') {
            steps {
                // Trains the model and updates the local mlruns directory
                sh 'python src/train.py'
            }
        }
        
        stage('Unit Testing') {
            steps {
                // Tests the FastAPI endpoints
                sh 'pytest tests/'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                // Packages the API and the fresh models into a container
                sh 'docker build -t heart-disease-api:latest .'
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline completed successfully! Docker image is ready.'
        }
        failure {
            echo 'Pipeline failed. Check the logs for errors.'
        }
    }
}