# Food Security Sandbox

A collaborative machine learning platform for agricultural data analysis and model training with privacy-preserving features.

## Project Description

The Food Security Sandbox is a comprehensive web application that enables farmers and researchers to collaborate on machine learning models while preserving data privacy. The platform provides tools for dataset management, collaborative model training, and privacy risk analysis in agricultural contexts.

**Key Features:**
- Dataset upload and management
- Collaborative machine learning model training
- Privacy-preserving data sharing
- Model repository with risk analysis
- Chat for collaboration
- Similar farmer identification

## Tags

**PADI** 

**Digital-Agriculture** 

## License

MIT License

Copyright (c) 2025 Food Security Sandbox
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## References

### Key Technologies and Libraries
- **Frontend**: React.js, Material-UI, Axios
- **Backend**: Flask, Python, MongoDB
- **Machine Learning**: TensorFlow, Scikit-learn, Adversarial Robustness Toolbox
- **Privacy**: Differential Privacy, Membership Inference Attack Detection
- **Authentication**: TACC Tapis Authentication

### External Documentation
- [React Documentation](https://reactjs.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Adversarial Robustness Toolbox](https://adversarial-robustness-toolbox.readthedocs.io/)

### Key Concepts
- **Differential Privacy**: A mathematical framework for providing privacy guarantees when analyzing data
- **Membership Inference Attack**: An attack that determines whether a particular data point was used to train a machine learning model
- **Collaborative Learning**: A machine learning approach where multiple parties contribute to model training without sharing raw data
- **Federated Learning**: A machine learning technique that trains an algorithm across multiple decentralized edge devices or servers

## Acknowledgements

This research was supported in part by the National Science Foundation (NSF) under awards OAC-2112606 and 2112533. Also, this research was partly supported by the United States Department of Agriculture (USDA) under grant number NR233A750004G019.

## Tutorials

### Getting Started with the Food Security Sandbox

#### Prerequisites
- Docker and Docker Compose installed
- Node.js (v16 or higher)
- Python (v3.8 or higher)
- MongoDB (or use the provided Docker container)

#### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Digital-Agriculture-Sandbox
   ```

2. **Config Environment**
   Create .env file in app-server with the following keys:
   - MONGODB_URI=mongodb://mongodb:27017/digital_agriculture
   - CLIENT_ID= your client id
   - CLIENT_KEY= your client key
   - TAPIS_BASE_URL=https://icicleai.tapis.io
   - TENANT=icicleai
   - APP_BASE_URL=http://localhost:3000
   - CALLBACK_URL=http://localhost:5003/api/oauth2/callback
     
   Create .env files in farmer-server and param-server with the following keys:
   - TAPIS_BASE_URL=https://icicleai.tapis.io
   - TENANT=icicleai

3. **Start the Application**
   ```bash
   docker compose -p digital-agriculture-sandbox up --build
   ```

4. **Access the Application**
   - Frontend: http://localhost:3000
   - App Server: http://localhost:5003
   - Farmer Server: http://localhost:5001
   - Param Server: http://localhost:5002

5. **Authentication**
   - Use TACC Tapis authentication
   - Register at https://accounts.tacc.utexas.edu/register
   - Use any other CILogon account such as Google, GitHub, ORCID



## How-To Guides

### How to Upload a Dataset

**Problem**: You need to upload agricultural data for analysis and model training.

**Solution**:
1. Navigate to the Home page
2. Click on "Finding Similar Farmers" function
3. In the left panel, use the upload form
4. Enter a dataset name
5. Select a CSV file with your agricultural data
6. Click "Upload Dataset"

**Troubleshooting**:
- Ensure your CSV file is properly formatted
- Check that the file size is within limits
- Verify that the dataset name is unique

### How to Train a Collaborative Model

**Problem**: You want to train a machine learning model using data from multiple farmers while preserving privacy.

**Solution**:
1. Navigate to "Collaborative Machine Learning"
2. Select a dataset from your uploaded datasets
3. Choose model parameters:
   - Model Name
   - Model Type (Dense, Conv1D, LSTM)
   - Model Visibility (Public/Private)
4. Review selected collaborators
5. Click "Start Training"

**Advanced Tips**:
- Use different model types for different data characteristics
- Consider privacy settings based on your data sensitivity
- Monitor training progress through the interface

### How to Analyze Model Privacy Risks

**Problem**: You need to assess the privacy risks associated with your trained models.

**Solution**:
1. Go to the Model Repository
2. Find your model in the list
3. Click the red privacy risk icon (GppMaybeIcon)
4. Review the risk analysis sections:
   - Model Logs
   - Model Activity
   - Risk Analysis (Membership Inference Attack accuracy)

**Troubleshooting**:
- If risk analysis is not available, ensure the model training completed successfully
- Check that the model has sufficient data for analysis

## Explanation

### System Architecture

The Food Security Sandbox follows a microservices architecture with four main components:

1. **Frontend (React.js)**: User interface for data upload, model training, and collaboration
2. **App Server (Flask)**: Main application server handling authentication and coordination
3. **Farmer Server (Flask)**: Handles dataset management and privacy-preserving operations
4. **Param Server (Flask)**: Manages model training and parameter aggregation

### Privacy-Preserving Mechanisms

The system implements several privacy-preserving techniques:

1. **Differential Privacy**: Adds calibrated noise to data to prevent individual identification
2. **PCA Transformation**: Reduces data dimensionality while preserving important features
3. **Membership Inference Attack Detection**: Monitors and reports potential privacy breaches
4. **Sandboxed Processing**: Isolates data processing to prevent unauthorized access

### Collaborative Learning Workflow

1. **Data Preparation**: Farmers upload datasets with metadata
2. **Similarity Matching**: System identifies farmers with similar data characteristics
3. **Privacy Enhancement**: Data is processed with differential privacy techniques
4. **Model Training**: Collaborative model training using federated learning principles
5. **Risk Assessment**: Privacy risks are analyzed and reported
6. **Model Deployment**: Trained models are stored in the repository for future use





