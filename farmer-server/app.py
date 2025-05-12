from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from jose import jwt
import datetime
from tapipy.tapis import Tapis
import logging
import traceback
import pandas as pd
import datetime
from bson.objectid import ObjectId
from multiprocessing import Process, Queue
import numpy as np
import pickle
import threading
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import torch
import torch.nn as nn
import torch.optim as optim


# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Apply to all routes

# MongoDB connection
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://mongodb:27017/digital_agriculture')
client = MongoClient(mongo_uri)
db = client.digital_agriculture
epsilon = 5

def load_PCA_model():
    filename = 'pca_model.pkl'
    with open(filename, 'rb') as file:
        loaded_pca = pickle.load(file)

    return loaded_pca

# Define a simple neural network model (for illustration)
class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.layer1 = nn.Linear(4, 32)  # Iris dataset has 4 features
        self.layer2 = nn.Linear(32, 3)  # 3 classes in Iris (output layer)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x

# Simulate the local training process
def train_nn_model(model, X_train, y_train, optimizer, criterion):
    model.train()
    # Convert numpy arrays to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Long type for classification
    
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_tensor)
    
    # Compute the loss
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    return model.state_dict()  # Get the model weights after training


pcaGlobal = load_PCA_model()
models = {
    "LR": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "NN": NNModel()
}

# JWT Configuration
JWT_SECRET = '6554a9038d6a07bbf3cb17973c13ce2c5f24a71c247210b1f2a8d04cfb8a6907a102064629058d7d89ed4d03a5503fa485e3898346f3baeef1ed510268e680f65d6d7ccaed5ca755586702e55142e1c07e53f5b38b7055b4bb55a70baf0dcdc0d4150347041a1509fc7d12d705ffe4c8e9ff9cb8f9bba5ffd6129128b62e84de4e9087d21d342a10d87a53c59eec2323dcf3a3d2276d62793df37c5e96eacbabc44f1ce1930e7e8ceb97c88f83d75d4fdcb2cebda1ceea7b99294c6d0c4db8fa71d2295b7b73f80813a734447983d47f430d0dddbd90c5ff81a35b46cad10cde33901456e3fe6f7166152366693224a072d7182b40c38bbf04c3ccf76ff3b6db' # Change this in production

def calculateSensitivities(df):
  sens = []
  for col in df.columns:
    sens.append(max(abs(df[col].max() - df[col].min()), 1))
  return sens

def sandboxed_privacy_enabling(username, metadata, result_queue):
    try:
        datasetCollection = db['datasets']  
        user_collection = datasetCollection[username]
        matching_dataset = user_collection.find_one({'metadata': metadata})
        df = pd.DataFrame(matching_dataset['data'])
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        clientPCA = pcaGlobal.transform(x)
        noisy_X = np.zeros_like(x, dtype=float)
        sen = np.array(calculateSensitivities(pd.DataFrame(x)))
        esps = np.array([epsilon/len(sen)]*len(sen))
        scales = sen / esps
        # Add Laplace noise to each column of PCA data
        for i in range(x.shape[1]):
            # Generate Laplace noise
            noise = np.random.laplace(0, scales[i], size=x.shape[0])
            # Add noise to the column
            noisy_X[:, i] = x[:, i] + noise

        result_queue.put((username, noisy_X, y))
    except Exception as e:
        print(str(e))
        result_queue.put((username, f"error: {str(e)}"))

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "farmer-server"})


@app.route('/api/get_user_datasets', methods=['GET'])
def get_user_datasets():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Invalid token"}), 401
        
        token = auth_header.split(' ')[1]

        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        
        # Get stored Tapis token
        session = db.sessions.find_one({"username": payload['username']})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = payload['username']
        # Query the datasets collection for the user's datasets
        datasets = db['datasets'][user_id].find()

        # Convert the cursor to a list of datasets
        datasets_list = [{"dataset_name": dataset["dataset_name"], "_id": str(dataset["_id"]), "num_records": str(dataset["num_records"]), "metadata": str(dataset["metadata"])} for dataset in datasets]

        return jsonify(datasets_list), 200
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while loading datasets', 'error': str(e)}), 500
    
@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            logging.error("Authorization header missing")
            return jsonify({"error": "Authorization header missing"}), 401

        token = auth_header.split()[1]
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        
        # Get stored Tapis token
        session = db.sessions.find_one({"username": payload['username']})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = payload['username']
        # user_id = "TestAccount4"

        # Check if the post request has the file part
        if 'file' not in request.files:
            logging.error('No file part in the request')
            return jsonify({'message': 'No file part in the request'}), 400

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            logging.error('No selected file')
            return jsonify({'message': 'No selected file'}), 400

        if file and file.filename.endswith('.csv'):
            dataset_name = request.form.get('field1')

            try:
                # Read the file directly into a DataFrame, setting the first column as sample_id
                df = pd.read_csv(file, header=0, index_col=0)
# Check if DataFrame is not empty
                if not df.empty:
                    # Convert DataFrame to dictionary
                    data = df.to_dict(orient="records")

                    # Insert records into the datasets collection
                    db['datasets'][user_id].insert_one({
                        "user_id": str(user_id),
                        "dataset_name": str(dataset_name),
                        "metadata": list(df.columns),
                        "num_records": str(len(data)),
                        "data": data
                    })
                else:
                    logging.error('CSV file is empty')
                    return jsonify({'message': 'CSV file is empty'}), 400

            except Exception as e:
                logging.error(f'Error reading CSV or inserting into DB: {str(e)}')
                logging.error(traceback.format_exc())
                return jsonify({'message': 'An error occurred while processing the file', 'error': str(e)}), 500

            return jsonify({'message': 'CSV file processed successfully'}), 200
        else:
            logging.error('Unsupported file type')
            return jsonify({'message': 'Unsupported file type'}), 400
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while processing the file', 'error': str(e)}), 500


@app.route('/api/delete_dataset', methods=['GET'])
def delete_dataset():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Invalid token"}), 401
        
        token = auth_header.split(' ')[1]

        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        
        # Get stored Tapis token
        session = db.sessions.find_one({"username": payload['username']})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = payload['username']
        
        data = request.get_json()  # Get the JSON data from the request body
        datasetID = data.get('datasetId')  # Extract selectedDataset

        print(datasetID)

        collection = db['datasets.'+user_id]  # Replace with the actual collection name

        # Delete the record
        result = collection.delete_one({"_id": ObjectId(str(datasetID))})

        # Check if the deletion was successful
        if result.deleted_count > 0:
            print("Dateset deleted successfully.")
            return jsonify({'message': 'Dateset Deleted'}), 200
        else:
            print("No record found with _id .")
            return jsonify({'message': 'Dateset Not Deleted'}), 500

    
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while loading datasets', 'error': str(e)}), 500
    
@app.route('/api/load_datasets', methods=['GET'])
def load_datasets():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Invalid token"}), 401
        
        token = auth_header.split(' ')[1]

        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        
        # Get stored Tapis token
        session = db.sessions.find_one({"username": payload['username']})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = payload['username']
        data = request.args # Get the JSON data from the request body
        datasetID = data['datasetId']  # Extract selectedDataset

        datasetCollection = db['datasets']  
        userDatasetCollection = datasetCollection[user_id]
        document = userDatasetCollection.find_one({"_id": ObjectId(str(datasetID))})
        
        results = {}
        # Check if the document was found and if it contains the 'metadata' key
        if document and 'metadata' in document:
            # Get a list of all collection names in the database
            
            all_users = [doc['username'] for doc in db['sessions'] .find()]

            result_queue = Queue()
            processes = []

            for username in all_users:
                p = Process(target=sandboxed_privacy_enabling, args=(username, document['metadata'], result_queue))
                p.start()
                processes.append(p)

                user_collection = datasetCollection[username]
                matching_dataset = user_collection.find_one({'metadata': document['metadata']})
            
            for p in processes:
                p.join()

            results = {}
            
            while not result_queue.empty():
                farmer_id, noisyX, noisyY = result_queue.get()
                results[farmer_id] = {"noisyX":noisyX.tolist(), "noisyY" : noisyY.tolist()}

            return jsonify(results)
             
        else:
            print(f"Document with id '{datasetID}' not found or does not contain 'metadata'.")
            return None
        
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while loading datasets', 'error': str(e)}), 500

@app.route('/api/trainLocalModel', methods=['POST'])
def train_local_model():
    start_time = time.time()
    receivedData = request.get_json()
    userID = receivedData['userID'] # Get userID from the request.
    hyperparameters = receivedData['hyperparameters'] # Get hyperparameters from the request.

    #  In a real Federated Learning scenario, you would load
    #  the user's local data here.  For this example, we'll use the
    #  entire iris dataset, but you might have a dictionary mapping
    #  user_id to their specific data.
    #  For example:
    #  user_data = {
    #      1: (X_train_user1, y_train_user1),
    #      2: (X_train_user2, y_train_user2),
    #      ...
    #  }
    #  X_train_local, y_train_local = user_data.get(user_id, (X_train, y_train)) # default to the whole dataset

    iris = datasets.load_iris()
    X = iris.data          # Features (sepal and petal measurements)
    y = iris.target        # Labels (species)

    # Optional: Convert to a DataFrame for better visualization
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y

    # Step 2: Split the data (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5
    )
  
    model = models[hyperparameters['modelName']]  # Increased max_iter for convergence
    weights = None
    intercept = None
    if hyperparameters['modelName'] == 'NN':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        # Train the model locally
        weights = train_nn_model(model, X_train, y_train, optimizer, criterion)
    else:



        model.fit(X_train, y_train)  # Train on the full dataset (or user-specific data)

    # Get the model weights (updates).  This is what we send back to Server A.
    #  For Logistic Regression, this is the coefficients and the intercept.
        weights = model.coef_.tolist()
        intercept = model.intercept_.tolist()
    #  Important:  In a real Federated Learning scenario, you might want to
    #  send back only the *difference* between the initial model and the
    #  updated model, to reduce the amount of data being transferred.

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Server B: Trained model for user {userID} in {training_time:.4f} seconds.")

    return jsonify({'weights': weights, 'intercept': intercept, 'training_time': training_time, 'userID': userID}) # Include timing



@app.route('/api/test', methods=['GET'])
def test():
    datasetCollection = db['sessions'] 
    users = []
    for doc in datasetCollection.find():
        users.append(doc['username'])
        
    return str(users), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 