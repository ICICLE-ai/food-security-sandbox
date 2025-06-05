from flask import Flask, jsonify, request, render_template
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
import requests
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import threading
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Apply to all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# MongoDB connection
mongo_uri = os.getenv('MONGODB_URI', 'mongodb://mongodb:27017/digital_agriculture')
REACT_APP_FARMER_API_URL = os.getenv('REACT_APP_FARMER_API_URL', 'http://digital-agriculture-sandbox-farmer-server-1:5001')

client = MongoClient(mongo_uri)
db = client.digital_agriculture
messages_collection = db['messages']

# TACC Configuration
JWT_SECRET = '6554a9038d6a07bbf3cb17973c13ce2c5f24a71c247210b1f2a8d04cfb8a6907a102064629058d7d89ed4d03a5503fa485e3898346f3baeef1ed510268e680f65d6d7ccaed5ca755586702e55142e1c07e53f5b38b7055b4bb55a70baf0dcdc0d4150347041a1509fc7d12d705ffe4c8e9ff9cb8f9bba5ffd6129128b62e84de4e9087d21d342a10d87a53c59eec2323dcf3a3d2276d62793df37c5e96eacbabc44f1ce1930e7e8ceb97c88f83d75d4fdcb2cebda1ceea7b99294c6d0c4db8fa71d2295b7b73f80813a734447983d47f430d0dddbd90c5ff81a35b46cad10cde33901456e3fe6f7166152366693224a072d7182b40c38bbf04c3ccf76ff3b6db'  # Change this in production

# Convert numpy arrays to lists before JSON serialization
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    return obj

def create_jwt_token(username):
    """Create a JWT token for the authenticated user"""
    payload = {
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def send_request_to_train_local_model(user_id, metadata, updates_list, hyperparameters, lock):
    try:
        response = requests.post(f"{REACT_APP_FARMER_API_URL}/api/trainLocalModel", json={'userID': str(user_id), 'metadata':metadata, "hyperparameters": hyperparameters})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        data = response.json()
        print(data['userID'], ' ', data['training_time'])
        # print(f"Received update from Farmer Server for user {user_id}: {data}") #  too verbose
        with lock:
            updates_list.append(data)
    except requests.exceptions.RequestException as e:
        print(f"Error sending request to Farmer Server for user {user_id}: {e}")
        # Consider adding error handling here, such as retrying the request
        # or logging the error to a file.  For now, we'll just print to the console.
        with lock:
            updates_list.append(None) # Append None to keep the order correct.

def aggregate_updates(client_updates):
    """Aggregates the model updates from all clients."""
    num_clients = len(client_updates)
    
    
    aggregated_weights = [np.mean([updates['weights'][i] for updates in client_updates], axis=0) for i in range(len(client_updates[0]['weights']))]
    
    return aggregated_weights

def update_global_model(global_model, aggregated_weights):
    """Updates the global model with the aggregated weights."""
    global_model.set_weights(aggregated_weights)
    return global_model

def evaluate_model(model, test_data, test_labels):
    """Evaluates the performance of the model on the test set."""
    test_data_reshaped = np.expand_dims(test_data, axis=-1) if model.layers[0].input_shape[-1] == 1 and len(test_data.shape) == 2 else test_data
    _, accuracy = model.evaluate(test_data_reshaped, test_labels, verbose=0)
    return accuracy

def start_training_process(collaborators, metadata, model_id, hyperparameters, token):
    # data_received = request.get_json()  # Get the JSON data from the request body
    # n_users = int(data_received.get('n'))

    try:
        
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        
        # Get stored Tapis token
        session = db.sessions.find_one({"username": payload['username']})
        if not session:
            print({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            print({"status": "error", "message": "Tapis token expired"}), 401
    
        user_id = payload['username']

        client_updates = []
        threads = []
        lock = threading.Lock() #  Use a lock for thread-safe access to the updates list.

        print(f"Starting training for {len(collaborators)} users...")
        start_time = time.time()

        headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                    }
        # response = requests.get("http://digitalagriculturesandbox-farmer-server-1:5001/api/get_datasets_metadata", headers=headers, params={'datasetId' : hyperparameters['datasetName']})
        # metadata = response.json()['metadata']
        # num_classes = response.json()['num_classes']
        
        # global_model = create_model('dense', metadata[1:].shape[1:], num_classes)
        # global_model = compile_model(global_model)

        # Create and start a thread for each user.
        for collaborator_id in collaborators:
            thread = threading.Thread(target=send_request_to_train_local_model, args=(collaborator_id, metadata, client_updates, hyperparameters, lock))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete.
        for thread in threads:
            thread.join()

        # Check for any failed requests (None values in updates list)
        if None in client_updates:
            print("Warning: Some requests to Farmer Server failed.  Aggregation may be incomplete.")
            #  Remove the None values before proceeding with aggregation
            client_updates = [u for u in client_updates if u is not None]

        aggregated_weights = aggregate_updates(client_updates)
        # Parameter server: Update global model
        # global_model = update_global_model(global_model, aggregated_weights)
        string_model_weights = json.dumps(convert_numpy_to_list(aggregated_weights))

        # Convert the string ID to a MongoDB ObjectId
        object_id = ObjectId(model_id)
        # Use find_one to retrieve the document with the matching _id
        document = db['models'].find_one({'_id': object_id})
        new_values = {
            '$set': {
                'modelWeights': string_model_weights,
                'status': 'Done'
            }
        }
        if document:
            db['models'].update_one({'_id': object_id}, new_values)
            print('Found public and updated')
        else:
            print('Not Found Public', user_id, ' ', model_id)
            doc = db['models'][user_id].find_one({'_id': object_id})
            if doc:
                db['models'][user_id].update_one({'_id': object_id}, new_values)
                print('Found private and updated')
            else:
                print('Not Found Private!')
        end_time = time.time()
        total_time = end_time - start_time
        print(f"All training threads completed in {total_time:.4f} seconds.")
        print(f"Received {len(client_updates)} updates from Farmer Server.")

        

        
        print({'message': 'Training complete', 'total_time': total_time, 'num_updates': len(client_updates)}) # Return timing

    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        print({'message': 'An error occurred while processing the file', 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "app-server"})



@app.route('/api/start_training', methods=['POST'])
def train():
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
    
        data_received = request.get_json()  # Get the JSON data from the request body

        thread = threading.Thread(target=start_training_process, args=(data_received.get('collaborators'), data_received.get('metadata'), data_received.get('model_id'), data_received.get('hyperparameters'), token))
        thread.start()
    
        return jsonify({'message': 'Training process started.'}), 200 # Return timing
    
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while processing the file', 'error': str(e)}), 500


    
@app.route('/api/test', methods=['GET'])
def test():
    # Find all messages where user is either sender or receiver
    messages = db['messages'].find({
        "$or": [
            {"senderID": 'osamazafar98'},
            {"receiverID": 'osamazafar98'}
        ]
    })
        
    # Create a dictionary to store latest message timestamp for each partner
    conversation_partners = {}
    for message in messages:
        partner_id = message['receiverID'] if message['senderID'] == 'osamazafar98' else message['senderID']
        timestamp = message.get('timestamp')
        
        # Update timestamp only if it's more recent
        if partner_id not in conversation_partners or timestamp > conversation_partners[partner_id]['timestamp']:
            conversation_partners[partner_id] = {
                'partner_id': partner_id,
                'timestamp': timestamp,
                'last_message': message.get('content', '')
            }
    print(list(messages))
    # Convert to list and sort by timestamp
    sorted_partners = sorted(
        conversation_partners.values(),
        key=lambda x: x['timestamp'],
        reverse=True  # Most recent first
    )

    return jsonify(sorted_partners), 200
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True) 