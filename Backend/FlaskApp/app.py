import traceback

from pandas import DataFrame

from calculate_coefficients import compute_coefficients_array
import numpy as np
from flask import Flask, request, jsonify, g
from flask_login import LoginManager, login_user, logout_user, UserMixin
from flask_cors import CORS
from importlib_metadata import metadata
from itsdangerous import Serializer, SignatureExpired, BadSignature
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from bson.json_util import dumps
from dotenv import load_dotenv, find_dotenv
from werkzeug.utils import secure_filename
from pymongo.mongo_client import MongoClient
import os
import jwt
import datetime
import time
from flask_httpauth import HTTPBasicAuth
import json
import pandas as pd
import logging
from calculate_coefficients import compute_coefficients_array
from fuzzywuzzy import process
import uuid
from tapipy.tapis import Tapis
# from stats import calc_chi_pvalue
# from stats import calc_chi_pvalue

app = Flask(__name__)
load_dotenv(find_dotenv())
auth = HTTPBasicAuth()
base_url  = 'https://tacc.tapis.io'
app.config["MONGO_URI"] = os.getenv("MONGO_URI")
app.config["PORT"] = os.getenv("PORT")
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
CORS(app, supports_credentials=True)  # Initialize CORS
logging.basicConfig(level=logging.INFO)

login_manager = LoginManager()
login_manager.init_app(app)

def get_database():
    uri = app.config["MONGO_URI"]
    client = MongoClient(uri)
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client['test']

db = get_database()

blacklisted_tokens = []

class User(UserMixin):
    def __init__(self, user_json):
        self.user_json = user_json

    @property
    def id(self):
        return str(self.user_json["_id"])

    @property
    def email(self):
        return self.user_json["email"]

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password, method='sha256')

    def generate_auth_token(self, expires_in=100000):
        return jwt.encode(
            {'id': self.id, 'exp': time.time() + expires_in},
            app.config['SECRET_KEY'], algorithm='HS256')

    @staticmethod
    def verify_auth_token(token):
        if token in blacklisted_tokens:
            return None

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'],
                              algorithms=['HS256'])
        except:
            return
        user = db.users.find_one({"_id": ObjectId(data['id'])})
        if user:
            return User(user)
        return None

@login_manager.user_loader
def load_user(user_id):
    u = db.users.find_one({"_id": ObjectId(user_id)})
    if not u:
        return None
    return User(u)

def get_current_user():
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        logging.error("Authorization header missing")
        return None, jsonify({"error": "Authorization header missing"})

    try:
        token = auth_header.split()[1]
    except IndexError:
        logging.error("Invalid Authorization header format")
        return None, jsonify({"error": "Invalid Authorization header format"})

    current_user = User.verify_auth_token(token)
    if not current_user:
        logging.error("Invalid or expired token")
        return None, jsonify({"error": "Invalid or expired token"})

    return current_user, None

#######################################################################################################

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pickle
import keras
from keras import utils
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Activation,Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.layers import Input
import seaborn as sns
import matplotlib.pyplot as pltn
from sklearn.preprocessing import normalize
import io
import json

def standardize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

import numpy as np

def calculateSensitivities(df):
  sens = []
  for col in df.columns:
    sens.append(max(abs(df[col].max() - df[col].min()), 1))
  return sens


def add_noise(sample,sens,esps):
  newSample = np.zeros_like(sample)
  for idx in range(len(sample)):
    value = sample[idx]
    noise = np.random.laplace(scale= sens[idx] / esps[idx])
    newSample[idx] = value + noise
  return newSample

def create_model():
  model = Sequential()
  model.add(Dense(256, input_shape=(7,), activation='relu'))
  model.add(Dense(128,  activation='relu'))
  model.add(Dense(22, activation='softmax'))
	# Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def load_PCA_model():
    filename = 'pca_model.pkl'
    with open(filename, 'rb') as file:
        loaded_pca = pickle.load(file)

    return loaded_pca

def trainClient(last_update = None):
  clientModelWeights = []
  for i in range(numCollaborators):
    cModel = create_model()
    if last_update:
      cModel.set_weights(last_update)
    cY = clientY[PotentialCollaborators[i+1]]
    cModel.fit(clientX[PotentialCollaborators[i+1]],np.array([labels[list(set(y)).index(i)] for i in cY]),batch_size=20,epochs=25,verbose=0)
    clientModelWeights.append(cModel.get_weights())
  return clientModelWeights

def laplace_mechanism_2d(data, esp):
    newData = np.zeros_like(data)
    ndf = pd.DataFrame(data)
    sens = calculateSensitivities(ndf)
    totalEsp = esp
    esps = [totalEsp/len(sens)]*len(sens)
    print(esps)
    for i in range(len(data)):
      newData[i] = add_noise(data[i], sens, esps)
    return newData

def findSimilarFarmer(clientX,clientY,initiator):
    pcaGlobal = load_PCA_model()
    clientPCAs = [pcaGlobal.transform(client_data) for client_data in clientX]
    epsilon = 5
    clientNoisyPCAs = []
    for data in clientPCAs:
        noisy_X = np.zeros_like(data, dtype=float)
        sen = np.array(calculateSensitivities(pd.DataFrame(data)))
        esps = np.array([epsilon/len(sen)]*len(sen))
        scales = sen / esps
        # Add Laplace noise to each column of PCA data
        for i in range(data.shape[1]):
            # Generate Laplace noise
            noise = np.random.laplace(0, scales[i], size=data.shape[0])
            # Add noise to the column
            noisy_X[:, i] = data[:, i] + noise
        clientNoisyPCAs.append(noisy_X)

    cropLabels = list(set(clientY[0]))
    clientProportions = []
    for client in clientY:
        row = [0 for i in range(len(cropLabels))]
        for labelY in client:
            row[cropLabels.index(labelY)] += 1
        clientProportions.append(np.array([element/len(client) for element in row]))
    clientProportions = np.array(clientProportions)
    clientAverages = np.array([np.mean(client, axis=0) for client in clientX])

    columns = list([i for i in range(7)])
    df = pd.DataFrame(clientAverages, columns= columns)
    print(df.head())
    distances = pairwise_distances([df[columns].values[initiator]], df[columns].values, metric='euclidean')
    PotentialCollaborators = [b[0] for b in sorted(enumerate(distances[0]),key=lambda i:i[1])]
    print(PotentialCollaborators)
    return PotentialCollaborators
    

@app.route('/api/get_similar_farmers', methods=['POST'])
def get_similar_farmers():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            logging.error("Authorization header missing")
            return jsonify({"error": "Authorization header missing"}), 401

        token = auth_header.split()[1]
        current_user = User.verify_auth_token(token)

        if not current_user:
            logging.error("Invalid token or user not found")
            return jsonify({"error": "Invalid token or user not found"}), 401

        user_id = current_user.id
        data = request.get_json()  # Get the JSON data from the request body
        selected_DS_ID = data.get('selectedDataset')  # Extract selectedDataset

        datasets = db['datasets'].aggregate([
            {
                '$group': {
                    '_id': '$user_id',  # Group by user_id
                    'dataset': {'$first': '$$ROOT'}  # Get the first dataset for each user
                }
            }
        ])

        # Convert the cursor to a list of datasets
        datasets_list = [
            {
                **dataset['dataset'],
                '_id': str(dataset['dataset']['_id'])  # Convert ObjectId to string
            } for dataset in datasets
        ]
        initiator = 0
        for i in range(len(datasets_list)):
            if selected_DS_ID == datasets_list[i]['_id']:
                initiator = i
                break
        clientX = []
        clientY = []

        for dataset in datasets_list:
            df = pd.DataFrame(dataset['data'])
            x = df.iloc[:, :-1].to_numpy()
            y = df.iloc[:, -1].to_numpy()
            clientX.append(x)
            clientY.append(y)
        
        collaboratorsIdx = findSimilarFarmer(clientX,clientY,initiator)
        collaborators = []
        for idx in collaboratorsIdx[1:4]:
            collaborators.append({ 'id': datasets_list[idx]['_id'], 'name': datasets_list[idx]['user_id'] })

        return jsonify({'collaborators': collaborators}), 200

    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while processing the file', 'error': str(e)}), 500



#######################################################################################################


@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        logging.error("Missing required fields")
        return jsonify({'message': 'Name, email, and password are required'}), 400

    user = db.users.find_one({'email': email})

    if user:
        return jsonify({'message': 'Email already exists'}), 409

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
    db.users.insert_one({
        'name': name,
        'email': email,
        'password': hashed_password
    })

    return jsonify({'message': 'User created successfully'}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email']
    password = data['password']
    user = db.users.find_one({'email': email})

    if user and check_password_hash(user['password'], password):
        user_obj = User(user)
        login_user(user_obj)
        g.user = user_obj  # Set g.user to the logged-in user
        token = g.user.generate_auth_token()
        return jsonify({'message': 'Login successful', 'token': token, 'redirect' : '/home'}), 200

    return jsonify({'message': 'Invalid email or password'}), 401

@app.route('/api/profile', methods=['GET'])
def get_profile():
    print(get_current_user())
    current_user, error_response = get_current_user()
    if error_response:
        return error_response
    user_id = current_user.get_id()
    user = db.users.find_one({"_id": ObjectId(user_id)}, {"email": 1, "name": 1})
    if not user:
        return jsonify({"message": "User not found"}), 404
    logging.info(f"User email: {user['email']}, User name: {user['name']}")
    return jsonify({"email": user['email'], "name": user['name']})

@app.route('/api/profile', methods=['PUT'])
def update_profile():
    current_user, error_response = get_current_user()
    if error_response:
        return error_response
    user_id = current_user.get_id()
    data = request.json
    name = data.get('name')
    current_password = data.get('currentPassword')
    new_password = data.get('newPassword')
    confirm_new_password = data.get('confirmNewPassword')

    if new_password and new_password != confirm_new_password:
        return jsonify({"message": "New passwords do not match"}), 400

    user = db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        return jsonify({"message": "User not found"}), 404

    if name and not current_password:
        return jsonify({"message": "Current password is required to update the name"}), 400

    # Check if current password is correct
    if current_password and not check_password_hash(user['password'], current_password):
        return jsonify({"message": "Current password is incorrect"}), 400

    # Update password if new password is provided
    if new_password:
        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
        db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"password": hashed_password}})

    # Update name if provided
    if name:
        db.users.update_one({"_id": ObjectId(user_id)}, {"$set": {"name": name}})

    return jsonify({"message": "Profile updated successfully"})


@auth.verify_password
def verify_password(email_or_token, password):
    # first try to authenticate by token
    user = User.verify_auth_token(email_or_token)
    if not user:
        # try to authenticate with username/password
        user = db.users.find_one({'email': email_or_token})
        if not user or not user.verify_password(password):
            return False
    g.user = user
    return True

# @app.route('/api/logout', methods=['POST'])
# def logout():
#     auth_header = request.headers.get('Authorization')
#     if not auth_header:
#         logging.error("Authorization header missing")
#         return jsonify({"error": "Authorization header missing"}), 401
#     token = auth_header.split()[1]

#     # Add the token to the blacklist
#     blacklisted_tokens.append(token)

#     logout_user()
#     return jsonify({'message': 'Logout successful'})

@app.route('/api/logout', methods=['POST'])
def logout():
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        logging.error("Authorization header missing")
        return jsonify({"error": "Authorization header missing"}), 401

    parts = auth_header.split()
    if len(parts) != 2 or parts[0] != 'Bearer':
        logging.error("Invalid Authorization header format")
        return jsonify({"error": "Invalid Authorization header format"}), 401

    token = parts[1]

    # Add the token to the blacklist
    blacklisted_tokens.append(token)

    logout_user()
    return jsonify({'message': 'Logout successful'})



@app.route('/api/resource')
@auth.login_required
def get_resource():
    return jsonify({'data': 'Hello, %s!' % g.user.email})

@app.route('/api/users', methods=['GET'])
def get_users():
    users = db.users.find({})
    users_list = [{"email": user["email"], "_id": str(user["_id"])} for user in users]
    # Convert the list to JSON, `dumps` from `bson.json_util` handles MongoDB ObjectId
    return dumps(users_list), 200


@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            logging.error("Authorization header missing")
            return jsonify({"error": "Authorization header missing"}), 401

        token = auth_header.split()[1]
        current_user = User.verify_auth_token(token)

        if not current_user:
            logging.error("Invalid token or user not found")
            return jsonify({"error": "Invalid token or user not found"}), 401

        user_id = current_user.id

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
                    db['datasets'].insert_one({
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


@app.route('/api/get_user_datasets', methods=['GET'])
def get_user_datasets():
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            logging.error("Authorization header missing")
            return jsonify({"error": "Authorization header missing"}), 401

        token = auth_header.split()[1]
        current_user = User.verify_auth_token(token)

        if not current_user:
            logging.error("Invalid token or user not found")
            return jsonify({"error": "Invalid token or user not found"}), 401

        user_id = current_user.id
        # Query the datasets collection for the user's datasets
        datasets = db['datasets'].find({"user_id": user_id})

        # Convert the cursor to a list of datasets
        datasets_list = [{"dataset_name": dataset["dataset_name"], "_id": str(dataset["_id"]), "num_records": str(dataset["num_records"]), "metadata": str(dataset["metadata"])} for dataset in datasets]

        return jsonify(datasets_list), 200
    except Exception as e:
        logging.error(f'Unexpected error: {str(e)}')
        logging.error(traceback.format_exc())
        return jsonify({'message': 'An error occurred while processing the file', 'error': str(e)}), 500
    
    


@app.route('/api/calculations', methods=['GET'])
def calculate_cofficients():
    data = request.json
    user1 = data['user1']
    user2 = data['user2']

    

    # Connect to MongoDB
    client = MongoClient(os.getenv("MONGO_URI"))

    try:
        # Get datasets for both users
        df_user1 = get_user_dataset(client, user1)
        df_user2 = get_user_dataset(client, user2)

        # Merge the datasets

        merged_data = pd.concat([df_user1, df_user2], axis=1)
  
        # Compute coefficients
        coeff_arr = compute_coefficients_array(merged_data)

        # Convert the results to a table format (DataFrame)
        results_table = pd.DataFrame(list(coeff_arr.items()), columns=['Pair', 'Coefficient'])
        
        # Return the table as a JSON response
        return results_table.to_json(orient='records'), 200

    except ValueError as ve:
        return jsonify({'message': str(ve)}), 404
    except Exception as e:
        return jsonify({'message': 'An error occurred', 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

