pipeline {
    agent any

    stages {
        stage('Install Dependencies') {
            steps {
                sh '''
                # Inside this multi-line string, bash executes it, so '#' is fine here!
                python -m venv venv
                
                ./venv/bin/python -m pip install --upgrade pip
                ./venv/bin/pip install flake8 pytest
                ./venv/bin/pip install -r requirements.txt
                '''
            }
        }
        
      stage('Code Linting') {
            steps {
                // Added --exclude=venv so it only checks your actual project code
                sh './venv/bin/flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=venv'
            }
        }
        
        stage('Model Training') {
            steps {
                // Use the venv's Python to train the model
                sh './venv/bin/python src/train.py'
            }
        }
        
        stage('Unit Testing') {
            steps {
                // Use the venv's pytest
                sh './venv/bin/pytest tests/'
            }
        }
        
        stage('Build Docker Image') {
            steps {
                // Docker handles its own environment, so this stays the same
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