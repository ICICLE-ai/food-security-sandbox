from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from jose import jwt
import datetime
import logging
import traceback
import pandas as pd
import datetime
from bson.objectid import ObjectId
from multiprocessing import Process, Queue
import numpy as np
import pickle
import threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import tensorflow as tf
import json

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


pcaGlobal = load_PCA_model()


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

def create_model(model_type, input_shape, num_classes):
    """Creates a TensorFlow model based on the specified type."""
    if model_type == "dense":
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=input_shape, dtype='float32'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif model_type == "conv1d":
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_shape[0], 1)),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif model_type == "lstm":
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(input_shape[0], 1)),
            tf.keras.layers.LSTM(units=10),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def compile_model(model):
    """Compiles the TensorFlow model."""
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, client_data, client_labels, epochs=5, batch_size=32):
    """Trains a local TensorFlow model on client data."""
    
    model.fit(client_data, client_labels, epochs=epochs, batch_size=batch_size, verbose=0)
    return [weight.tolist() for weight in model.get_weights()]


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

@app.route('/api/get_datasets_metadata', methods=['GET'])
def get_datasets_metadata():
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
        
        # Check if the document was found and if it contains the 'metadata' key
        if document and 'metadata' in document:
            print(str(document['metadata']))
            df = pd.DataFrame(document['data'], columns=document['metadata'])
            yTrain = df['label'].values
            return jsonify({'metadata': document['metadata'], 'num_classes' : len(np.unique(yTrain)), 'classes' : list(np.unique(yTrain))})
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
    metadata = receivedData['metadata'] # Get metadata from the request.
    hyperparameters = receivedData['hyperparameters'] # Get hyperparameters from the request.
    
    document = db['datasets'][str(userID)].find_one({"metadata": {'$in': metadata}})
    
    df = pd.DataFrame(document['data'], columns=document['metadata'])
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    xTrain = df.drop(['label', 'label_encoded'], axis=1).values
    yTrain = df['label_encoded'].values


    local_model = create_model('dense', xTrain.shape[1:], len(np.unique(yTrain)))
    local_model = compile_model(local_model)
    #local_model.set_weights(global_model.get_weights())
    local_weights = train_model(local_model, xTrain, yTrain)

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Server B: Trained model for user {userID} in {training_time:.4f} seconds.")

    return jsonify({'weights': local_weights, 'training_time': training_time, 'userID': userID}) # Include timing

@app.route('/api/predict_eval', methods=['POST'])
def predict_eval():
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

        receivedData = request.get_json()

        model_info = request.get_json()['model_info']
        eval_data = request.get_json()['eval_data']
        metadata = np.array(model_info['metadata'])
        classes = np.array(model_info['classes'])
        
        if model_info['modelVisibility'] == 'Public':
            model_info =  db['models'].find_one({"_id": ObjectId(model_info['_id']['$oid'])})

            local_model = create_model('dense', metadata[:-1].shape, model_info['num_classes'])
            local_model = compile_model(local_model)
            print('Setting Weights')

            loaded_weights_as_lists = json.loads(model_info['modelWeights'])
            loaded_weights_as_ndarrays = [np.array(weight_list) for weight_list in loaded_weights_as_lists]
            local_model.set_weights(loaded_weights_as_ndarrays)
            
            print('Weights Set')
            
            eval_data = np.array(json.loads(eval_data))

            # Prediction on multiple samples (a batch)
            predictions_batch = local_model.predict(eval_data)
            print(f"\nPredictions for batch (raw output, first 2 rows):\n{predictions_batch[:2]}")

            # Interpretation for multiple samples (for a softmax output)
            predicted_classes_batch = np.argmax(predictions_batch, axis=1)
            predicted_confidence_batch = [predictions_batch[idx][predicted_classes_batch[idx]] for idx in range(0, len(predicted_classes_batch))]
            print(f"Predicted classes for batch: {predicted_classes_batch}")
            print(f"Predicted classes for batch: {predicted_confidence_batch}")
            merged_prediction_output = []
            for name, weight in zip([classes[idx] for idx in predicted_classes_batch.tolist()], predicted_confidence_batch):
                merged_prediction_output.append(f"{name} ({weight:0.2f})")
            return jsonify({'predicted_classes_batch': merged_prediction_output}) 

        else:
            print('find private model')
            model_info =  db['models'][user_id].find_one({"_id": ObjectId(model_info['_id']['$oid'])})

            local_model = create_model('dense', metadata[:-1].shape, model_info['num_classes'])
            local_model = compile_model(local_model)
            print('Setting Weights')

            loaded_weights_as_lists = json.loads(model_info['modelWeights'])
            loaded_weights_as_ndarrays = [np.array(weight_list) for weight_list in loaded_weights_as_lists]
            local_model.set_weights(loaded_weights_as_ndarrays)
            
            print('Weights Set')
            
            eval_data = np.array(json.loads(eval_data))

            # Prediction on multiple samples (a batch)
            predictions_batch = local_model.predict(eval_data)
            print(f"\nPredictions for batch (raw output, first 2 rows):\n{predictions_batch[:2]}")

            # Interpretation for multiple samples (for a softmax output)
            predicted_classes_batch = np.argmax(predictions_batch, axis=1)
            predicted_confidence_batch = [predictions_batch[idx][predicted_classes_batch[idx]] for idx in range(0, len(predicted_classes_batch))]
            print(f"Predicted classes for batch: {predicted_classes_batch}")
            print(f"Predicted classes for batch: {predicted_confidence_batch}")
            merged_prediction_output = []
            for name, weight in zip([classes[idx] for idx in predicted_classes_batch.tolist()], predicted_confidence_batch):
                merged_prediction_output.append(f"{name} ({weight:0.2f})")
            return jsonify({'predicted_classes_batch': merged_prediction_output}) 


    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while loading datasets', 'error': str(e)}), 500


@app.route('/api/test', methods=['GET'])
def test():
    datasetCollection = db['datasets']  
    userDatasetCollection = datasetCollection['osamazafar98']
    metadata_list = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    doc = userDatasetCollection.find_one({"metadata": metadata_list})
        
    return str(doc['user_id']), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 