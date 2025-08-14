from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import datetime
import logging
import traceback
import pandas as pd
from bson.objectid import ObjectId
from multiprocessing import Process, Queue
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
import tensorflow as tf
import json
import requests
from config import app_settings

from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import KerasClassifier


# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Apply to all routes

# MongoDB connection
# mongo_uri = os.getenv('MONGODB_URI', 'mongodb://mongodb:27017/digital_agriculture')
client = MongoClient(app_settings.mongodb_uri)
db = client.digital_agriculture
epsilon = 5

def load_PCA_model():
    filename = 'pca_model.pkl'
    with open(filename, 'rb') as file:
        loaded_pca = pickle.load(file)

    return loaded_pca

pcaGlobal = load_PCA_model()


def calculateSensitivities(df):
  sens = []
  for col in df.columns:
    sens.append(max(abs(df[col].max() - df[col].min()), 1))
  return sens

def diamension_check(x):
    if x.shape[1] == 7:
        return x
    elif x.shape[1] > 7:
        return x[:, :7]
    else:  # x.shape[1] < 7
        padding = np.zeros((x.shape[0], 7 - x.shape[1]))
        return np.hstack((x, padding))

def sandboxed_privacy_enabling(username, metadata, result_queue):
    try:
        datasetCollection = db['datasets']  
        user_collection = datasetCollection[username]
        matching_dataset = user_collection.find_one({'metadata': metadata})
        if(matching_dataset == None):
            result_queue.put((username, None, None))
            return
        df = pd.DataFrame(matching_dataset['data'])
        x = diamension_check(df.iloc[:, :-1].to_numpy())
        y = df.iloc[:, -1].to_numpy()
        clientPCA = pcaGlobal.transform(x)
        noisy_X = np.zeros_like(clientPCA, dtype=float)
        sen = np.array(calculateSensitivities(pd.DataFrame(clientPCA)))
        esps = np.array([epsilon/len(sen)]*len(sen))
        scales = sen / esps
        # Add Laplace noise to each column of PCA data
        for i in range(clientPCA.shape[1]):
            # Generate Laplace noise
            noise = np.random.laplace(0, scales[i], size=clientPCA.shape[0])
            # Add noise to the column
            noisy_X[:, i] = clientPCA[:, i] + noise

        result_queue.put((username, noisy_X, y))
    except Exception as e:
        print(str(e))
        result_queue.put((username, f"error: {str(e)}"))

def create_model(model_type, input_shape, num_classes):
    """Creates a TensorFlow model based on the specified type."""
    print(num_classes)
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

def risk_analysis(model,x_train,x_test,y_train,y_test):

    attack_train_ratio = 0.5
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)
    
    classifier = KerasClassifier(model=model) 

    bb_attack = MembershipInferenceBlackBox(classifier)

    # train attack model
    bb_attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                x_test[:attack_test_size], y_test[:attack_test_size])
    # get inferred values
    inferred_train_bb = bb_attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
    inferred_test_bb = bb_attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
    # check accuracy
    train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
    test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
    acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
    print(f"Members Accuracy: {train_acc:.4f}")
    print(f"Non Members Accuracy {test_acc:.4f}")
    print(f"Attack Accuracy {acc:.4f}")
    return acc

def get_username(token):
    """
    Validate a Tapis `token`, and resolve it to a username.
    """
    headers = {'Content-Type': 'text/html'}
    # call the userinfo endpoint
    url = f"{app_settings.tapis_base_url}/v3/oauth2/userinfo"
    headers = {'X-Tapis-Token': token}
    try:
        rsp = requests.get(url, headers=headers)
        rsp.raise_for_status()
        username = rsp.json()['result']['username']
    except Exception as e:
        raise Exception(f"Error looking up token info; debug: {e}")
    return username

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
        
        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = username
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
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(' ')[1]
        
        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = username
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
        print('auth_header',auth_header)
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(' ')[1]
         
        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = username
        print(user_id)
        datasetID = request.args.get('datasetId')  # Extract selectedDataset

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
        
        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = username
        
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
            print(all_users)
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
                if noisyX is None and noisyY is None:
                    continue
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
        
        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = username
        
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
    try:
        start_time = time.time()
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(' ')[1]
        
        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = username
        
        receivedData = request.get_json()
        userID = receivedData['userID'] # Get userID from the request.
        metadata = receivedData['metadata'] # Get metadata from the request.
        hyperparameters = receivedData['hyperparameters'] # Get hyperparameters from the request.
        
        document = db['datasets'][str(userID)].find_one({"metadata": {'$in': metadata}})
        
        df = pd.DataFrame(document['data'], columns=document['metadata'])
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['label'])

        X = df.drop(['label', 'label_encoded'], axis=1).values
        y = df['label_encoded'].values

        xTrain, X_test, yTrain, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

        local_model = create_model('dense', xTrain.shape[1:], len(np.unique(y)))
        local_model = compile_model(local_model)
        #local_model.set_weights(global_model.get_weights())
        local_weights = train_model(local_model, xTrain, yTrain)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"Server B: Trained model for user {userID} in {training_time:.4f} seconds.")

        return jsonify({'weights': local_weights, 'training_time': training_time, 'userID': userID}) # Include timing
    
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while loading datasets', 'error': str(e)}), 500
    
@app.route('/api/model_risk_analysis', methods=['POST'])
def model_risk_analysis():
    try:    
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(' ')[1]
        
        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = username
        

        model_info = json.loads(request.get_json()['model_info'])
        metadata = np.array(model_info['metadata'])
        classes = np.array(model_info['classes'])

        start_time = time.time()
        
        document = db['datasets'][str(user_id)].find_one({"metadata": {'$in': metadata.tolist()}})
        
        df = pd.DataFrame(document['data'], columns=document['metadata'])
        le = LabelEncoder()
        df['label_encoded'] = le.fit_transform(df['label'])

        X = df.drop(['label', 'label_encoded'], axis=1).values
        y = df['label_encoded'].values

        xT_rain, X_test, yTrain, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        

        
        if model_info['modelVisibility'] == 'Public':
            model_info =  db['models'].find_one({"_id": ObjectId(model_info['_id']['$oid'])})
            print('Found public and updated')
            local_model = create_model('dense', metadata[:-1].shape, model_info['num_classes'])
            local_model = compile_model(local_model)
            print('Setting Weights')

            loaded_weights_as_lists = json.loads(model_info['modelWeights'])
            loaded_weights_as_ndarrays = [np.array(weight_list) for weight_list in loaded_weights_as_lists]
            local_model.set_weights(loaded_weights_as_ndarrays)
            
            print('Weights Set')
            
            
            acc = risk_analysis(local_model,xT_rain, X_test, yTrain, y_test)

            db['models'].update_one({"_id": ObjectId(model_info['_id'])}, {
                '$set':{
                    f'mia_attack_acc':float(acc)
                }
            })
            
            return jsonify({'Attack Accuracy': float(acc)}) 

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
            
            acc = risk_analysis(local_model,xT_rain, X_test, yTrain, y_test)

            db['models'].update_one({"_id": ObjectId(model_info['_id'])}, {
                '$set':{
                    f'mia_attack_acc':float(acc)
                }
            })
            
            return jsonify({'Attack Accuracy': acc}) 



    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while loading datasets', 'error': str(e)}), 500


@app.route('/api/predict_eval', methods=['POST'])
def predict_eval():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(' ')[1]
        
        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(session['tapis_token']['expires_at'])
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        
        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401
        
        user_id = username
        

        model_info = request.get_json()['model_info']
        eval_data = request.get_json()['eval_data']
        metadata = np.array(model_info['metadata'])
        classes = np.array(model_info['classes'])
        
        if model_info['modelVisibility'] == 'Public':
            model_info =  db['models'].find_one({"_id": ObjectId(model_info['_id']['$oid'])})
            print('Found public and updated')
            local_model = create_model('dense', metadata[:-1].shape, model_info['num_classes'])
            local_model = compile_model(local_model)
            print('Setting Weights')

            loaded_weights_as_lists = json.loads(model_info['modelWeights'])
            loaded_weights_as_ndarrays = [np.array(weight_list) for weight_list in loaded_weights_as_lists]
            local_model.set_weights(loaded_weights_as_ndarrays)
            
            print('Weights Set')
            
            eval_data = np.array(json.loads(eval_data))
            
            # model_logs = {}
            # if 'model_logs' in model_info.keys():
            #     model_logs = model_info['model_logs']
            #     if user_id in model_logs.keys():
            #         model_logs[user_id] = model_logs[user_id] + np.size(eval_data, 0)
            #     else:
            #         model_logs[user_id] = np.size(eval_data, 0)
            # else:
            #     model_logs = {
            #         user_id:np.size(eval_data, 0)
            #     }
            # new_values = {
            #     '$set': {
            #         'model_logs': model_logs
            #     }
            # }
            # print(model_logs,str(np.size(eval_data)))
            # db['models'].update_one({"_id": ObjectId(model_info['_id'])}, new_values)

            db['models'].update_one({"_id": ObjectId(model_info['_id'])}, {
                '$inc':{
                    f'model_logs.{user_id}':np.size(eval_data, 0)
                }
            })
            db['models'].update_one({"_id": ObjectId(model_info['_id'])}, {
                '$push':{
                    'model_activity':[user_id, np.size(eval_data, 0), datetime.datetime.now(datetime.timezone.utc)]
                }
            })

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
    
    return str("Test"), 200

if __name__ == '__main__':
    app.run(host=app_settings.host, port=app_settings.port, debug=app_settings.debug)