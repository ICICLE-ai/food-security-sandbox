from flask import Flask, jsonify, request, redirect
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from jose import jwt
import datetime
import logging
import traceback
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from flask_socketio import SocketIO
import json
from bson import json_util


from config import app_settings, auth_settings

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Apply to all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# MongoDB connection
# mongo_uri = os.getenv('MONGODB_URI', 'mongodb://mongodb:27017/digital_agriculture')
# app_settings.sandbox_server_url = os.getenv('app_settings.sandbox_server_url', 'http://localhost:5001')
# app_settings.param_server_url = os.getenv('app_settings.param_server_url', 'http://localhost:5002')

client = MongoClient(
    app_settings.mongodb_uri,
    maxPoolSize=5,  # Limit concurrent connections
    minPoolSize=1,  # Keep minimum connections alive
    maxIdleTimeMS=30000,  # Close idle connections after 30s
    serverSelectionTimeoutMS=5000,  # Fail fast instead of 30s timeout
    socketTimeoutMS=20000,  # Socket timeout
    connectTimeoutMS=20000,  # Connection timeout
    retryWrites=True,        # Retry failed writes
    tls=True
)
db = client.get_default_database()
messages_collection = db["messages"]


@app.route("/api/auth/login", methods=["GET"])
def login():
    try:
        tapis_url = f"{auth_settings.tapis_base_url}/v3/oauth2/authorize?client_id={auth_settings.client_id}&redirect_uri={auth_settings.callback_url}&response_type=code"
        return redirect(tapis_url, code=302)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/auth/verify", methods=["GET"])
def verify_token():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(" ")[1]

        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(
            session["tapis_token"]["expires_at"]
        )
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401

        return jsonify(
            {
                "status": "success",
                "username": username,
                "tapis_token": session["tapis_token"],
            }
        )

    except jwt.ExpiredSignatureError:
        return jsonify({"status": "error", "message": "Token expired"}), 401
    except jwt.JWTError:
        return jsonify({"status": "error", "message": "Invalid token"}), 401
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Enhanced health check with MongoDB status"""
    try:
        # Check MongoDB connection
        db.client.admin.command('ping')
        mongo_healthy = True
    except:
        mongo_healthy = False

    health_status = {
        "status": "healthy" if mongo_healthy else "degraded",
        "service": "app-server",
        "mongodb": "connected" if mongo_healthy else "disconnected",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    status_code = 200 if mongo_healthy else 503
    return jsonify(health_status), status_code

@app.route("/api/auth/get_username", methods=["GET"])
def get_username(token):
    """
    Validate a Tapis JWT, `token`, and resolve it to a username.
    """
    headers = {"Content-Type": "text/html"}
    # call the userinfo endpoint
    url = f"{auth_settings.tapis_base_url}/v3/oauth2/userinfo"
    headers = {"X-Tapis-Token": token}
    try:
        rsp = requests.get(url, headers=headers)
        rsp.raise_for_status()
        username = rsp.json()["result"]["username"]
    except Exception as e:
        raise Exception(f"Error looking up token info; debug: {e}")
    return username


@app.route("/api/test_db", methods=["GET"])
def test_db():
    """Test endpoint to check MongoDB connection health"""
    try:
        # Test 1: Basic ping
        start_time = datetime.datetime.now()
        db.client.admin.command("ping")
        ping_time = (datetime.datetime.now() - start_time).total_seconds()

        # Test 2: Simple write operation
        test_doc = {
            "test_id": f"test_{int(datetime.datetime.now().timestamp())}",
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }

        result = db.test_collection.insert_one(test_doc)

        # Test 3: Read it back
        retrieved = db.test_collection.find_one({"_id": result.inserted_id})

        # Clean up
        db.test_collection.delete_one({"_id": result.inserted_id})

        return jsonify(
            {
                "status": "success",
                "ping_time_seconds": ping_time,
                "write_successful": result.acknowledged,
                "read_successful": retrieved is not None,
                "mongodb_server": str(db.client.address),
                "topology": str(db.client.topology_description.topology_type),
            }
        )

    except Exception as e:
        return jsonify(
            {"status": "error", "error": str(e), "error_type": type(e).__name__}
        ), 500


@app.route("/api/oauth2/callback", methods=["GET"])
def callback():
    """
    Process a callback from a Tapis authorization server:
      1) Get the authorization code from the query parameters.
      2) Exchange the code for a token
      3) Add the user and token to the session
      4) Redirect to the /data endpoint.
    """
    code = request.args.get("code")
    if not code:
        raise Exception(f"Error: No code in request; debug: {request.args}")
    url = f"{auth_settings.tapis_base_url}/v3/oauth2/tokens"
    data = {
        "code": code,
        "redirect_uri": auth_settings.callback_url,
        "grant_type": "authorization_code",
    }
    try:
        response = requests.post(
            url, data=data, auth=(auth_settings.client_id, auth_settings.client_key)
        )
        print(response.text)
        response.raise_for_status()
        json_resp = json.loads(response.text)
        token = json_resp["result"]["access_token"]["access_token"]
        tapis_token_info = {
            "access_token": json_resp["result"]["access_token"]["access_token"],
            "expires_at": json_resp["result"]["access_token"]["expires_at"],
            "jti": json_resp["result"]["access_token"]["jti"],
        }

        username = get_username(json_resp["result"]["access_token"]["access_token"])

        # Store user session in MongoDB
        db.sessions.update_one(
            {"username": username},
            {
                "$set": {
                    "username": username,
                    "tapis_token": tapis_token_info,
                    "last_login": datetime.datetime.now(datetime.timezone.utc),
                }
            },
            upsert=True,
        )
    except Exception as e:
        raise Exception(f"Error generating Tapis token; debug: {e}")

    return redirect(
        auth_settings.app_base_url
        + "/?tapis_token="
        + str(token)
        + "&username="
        + str(username),
        code=302,
    )


@app.route("/api/get_similar_farmers", methods=["POST"])
def get_similar_farmers():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(" ")[1]

        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(
            session["tapis_token"]["expires_at"]
        )
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401

        user_id = username

        data = request.get_json()  # Get the JSON data from the request body
        selected_DS_ID = data.get("selectedDataset")  # Extract selectedDataset

        response = None

        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

            response = requests.get(
                f"{app_settings.sandbox_server_url}/sandbox/load_datasets",
                headers=headers,
                params={"datasetId": selected_DS_ID},
            )
        except requests.exceptions.RequestException as e:
            print(f"Error during request: {e}")
            return jsonify({"status": f"Error during request: {e}"}), 500

        protectedData = response.json()
        userNames = list(protectedData.keys())

        initiator = userNames.index(user_id)
        clientX = [protectedData[user]["noisyX"] for user in userNames]
        clientY = [protectedData[user]["noisyY"] for user in userNames]
        cropLabels = list(set(clientY[0]))
        clientProportions = []
        for client in clientY:
            row = [0 for i in range(len(cropLabels))]
            for labelY in client:
                row[cropLabels.index(labelY)] += 1
            clientProportions.append(
                np.array([element / len(client) for element in row])
            )
        clientProportions = np.array(clientProportions)
        clientAverages = np.array([np.mean(client, axis=0) for client in clientX])

        columns = list([i for i in range(2)])
        df = pd.DataFrame(clientAverages, columns=columns)

        distances = pairwise_distances(
            [df[columns].values[initiator]], df[columns].values, metric="euclidean"
        )
        collaboratorsIdx = [
            b[0] for b in sorted(enumerate(distances[0]), key=lambda i: i[1])
        ]
        collaborators = []
        for idx in collaboratorsIdx[1:4]:
            collaborators.append({"username": userNames[idx]})

        return jsonify({"collaborators": collaborators}), 200

    except Exception as e:
        print(e)
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify(
            {"message": "An error occurred while processing the file", "error": str(e)}
        ), 500


@app.route("/api/getMessages", methods=["POST"])
def get_messages():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(" ")[1]

        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(
            session["tapis_token"]["expires_at"]
        )
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401

        user_id = username

        sender_id = request.get_json().get("sender_id")
        receiver_id = request.get_json().get("receiver_id")

        # Fetching messages from the database
        messages = db["messages"].find(
            {
                "$or": [
                    {"senderID": sender_id, "receiverID": receiver_id},
                    {"senderID": receiver_id, "receiverID": sender_id},
                ]
            }
        )

        # Formatting the response
        messages_list = []
        for message in messages:
            messages_list.append(
                {
                    "senderID": message["senderID"],
                    "receiverID": message["receiverID"],
                    "message": message["message"],
                    "timestamp": message["timestamp"],
                }
            )
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"message": "An error occurred ", "error": str(e)}), 500

    return jsonify(messages_list)


@app.route("/api/sendMessage", methods=["POST"])
def send_message():
    try:
        auth_header = request.headers.get("Authorization")
        print(auth_header)
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(" ")[1]

        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(
            session["tapis_token"]["expires_at"]
        )
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401

        user_id = username

        data = request.json
        message = {
            "senderID": data["senderID"],
            "receiverID": data["receiverID"],
            "message": data["message"],
            "timestamp": data["timestamp"],
        }

        db["messages"].insert_one(message)
        return jsonify({"status": "Message sent"}), 201
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"message": "An error occurred ", "error": str(e)}), 500


@app.route("/api/conversations", methods=["POST"])
def get_conversations():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(" ")[1]

        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(
            session["tapis_token"]["expires_at"]
        )
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401

        user_id = username

        sender_id = request.get_json().get("sender_id")
        print(sender_id)
        if not sender_id:
            return jsonify({"error": "sender_id is required"}), 400

        # Find all messages where user is either sender or receiver
        messages = db["messages"].find(
            {"$or": [{"senderID": sender_id}, {"receiverID": sender_id}]}
        )

        # Create a dictionary to store latest message timestamp for each partner
        conversation_partners = {}
        for message in messages:
            partner_id = (
                message["receiverID"]
                if message["senderID"] == sender_id
                else message["senderID"]
            )
            timestamp = message.get("timestamp")

            # Update timestamp only if it's more recent
            if (
                partner_id not in conversation_partners
                or timestamp > conversation_partners[partner_id]["timestamp"]
            ):
                conversation_partners[partner_id] = {
                    "partner_id": partner_id,
                    "timestamp": timestamp,
                    "last_message": message.get("content", ""),
                }
        # Convert to list and sort by timestamp
        sorted_partners = sorted(
            conversation_partners.values(),
            key=lambda x: x["timestamp"],
            reverse=True,  # Most recent first
        )

        return jsonify(sorted_partners), 200

    except Exception as e:
        logging.error(f"Error in get_conversations: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/train", methods=["POST"])
def train():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(" ")[1]

        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(
            session["tapis_token"]["expires_at"]
        )
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401

        user_id = username

        data_received = request.get_json()  # Get the JSON data from the request body
        collaborators = [i["username"] for i in data_received.get("collaborators")]
        # collaborators.append(user_id)
        hyperparameters = data_received.get("hyperparameters")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = requests.get(
            f"{app_settings.sandbox_server_url}/sandbox/get_datasets_metadata",
            headers=headers,
            params={"datasetId": hyperparameters["datasetName"]},
        )

        metadata = response.json()["metadata"]
        num_classes = response.json()["num_classes"]
        classes = response.json()["classes"]

        data = {
            "modelName": hyperparameters["modelName"],
            "modelType": hyperparameters["modelType"],
            "modelVisibility": hyperparameters["modelVisibility"],
            "metadata": metadata,
            "modelOwner": user_id,
            "modelWeights": None,
            "modelReadme": hyperparameters["readme"],
            "collaborators": collaborators,
            "num_classes": num_classes,
            "classes": classes,
            "status": "Training",
        }
        result = None
        if hyperparameters["modelVisibility"] == "Public":
            result = db["models"].insert_one(data)
        else:
            result = db["models"][user_id].insert_one(data)

        model_id = str(result.inserted_id)

        response = requests.post(
            f"{app_settings.param_server_url}/api/start_training",
            headers=headers,
            json={
                "collaborators": collaborators,
                "metadata": metadata,
                "model_id": model_id,
                "hyperparameters": hyperparameters,
            },
        )

        if response:
            print(response)
        return jsonify({"message": "Training process started."}), 200  # Return timing

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify(
            {"message": "An error occurred while processing the file", "error": str(e)}
        ), 500


@app.route("/api/predict_model", methods=["POST"])
def get_model_prediction():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(" ")[1]

        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(
            session["tapis_token"]["expires_at"]
        )
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401

        user_id = username

        model_info = request.get_json()["model_info"]
        eval_data = request.get_json()["eval_data"]

        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            f"{app_settings.sandbox_server_url}/sandbox/predict_eval",
            headers=headers,
            json={"model_info": model_info, "eval_data": eval_data},
        )

        return jsonify(response.json()), 200

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify(
            {"message": "An error occurred while processing the file", "error": str(e)}
        ), 500


@app.route("/api/get_public_models", methods=["GET"])
def get_public_models():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(" ")[1]

        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(
            session["tapis_token"]["expires_at"]
        )
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401

        user_id = username

        # Query the datasets collection for the user's datasets
        models = db["models"].find()
        # Remove modelWeights field from each model
        models_list = list(models)
        for model in models_list:
            if "modelWeights" in model:
                del model["modelWeights"]

        return jsonify(json_util.dumps(models_list)), 200

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify(
            {"message": "An error occurred while loading datasets", "error": str(e)}
        ), 500


@app.route("/api/get_user_models", methods=["GET"])
def get_user_models():
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "message": "Invalid token"}), 401

        token = auth_header.split(" ")[1]

        username = get_username(token)

        # Get stored Tapis token
        session = db.sessions.find_one({"username": username})
        if not session:
            return jsonify({"status": "error", "message": "Session not found"}), 401

        # Check if Tapis token is expired
        expires_at = datetime.datetime.fromisoformat(
            session["tapis_token"]["expires_at"]
        )
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        if now_utc > expires_at:
            return jsonify({"status": "error", "message": "Tapis token expired"}), 401

        user_id = username

        # Query the datasets collection for the user's datasets
        models = db["models"][user_id].find()
        # Remove modelWeights field from each model
        models_list = list(models)
        for model in models_list:
            if "modelWeights" in model:
                del model["modelWeights"]

        return jsonify(json_util.dumps(models_list)), 200

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify(
            {"message": "An error occurred while loading datasets", "error": str(e)}
        ), 500


@app.route("/api/test", methods=["GET"])
def test():
    # Find all messages where user is either sender or receiver
    collections = db["datasets"]["osamazafar98"].find({})
    documents_to_copy = list(collections.sort("_id", 1).skip(1))

    if not documents_to_copy:
        return jsonify(
            {
                "status": "error",
                "message": "No documents found to copy (or only one document exists).",
            }
        )

    # Ensure we have exactly 4 documents if that's the strict requirement,
    # or adjust if you want to copy all available documents after the first.
    # For "last 4 documents", if source has 5, this should work.
    # If source has less than 5, this will copy whatever is available after the first.
    if len(documents_to_copy) < 4:
        print(
            f"Warning: Found only {len(documents_to_copy)} documents to copy after skipping the first."
        )

    # Define the new collection names
    new_collection_names = ["Test1", "Test2", "Test3", "Test4"]

    # This will hold status messages for each operation
    results = {}

    # Loop through the new collection names and the documents to copy
    # We use zip to pair them up. If there are fewer than 4 documents to copy,
    # zip will naturally stop at the shortest sequence.
    for i, (new_name, doc) in enumerate(zip(new_collection_names, documents_to_copy)):
        target_collection = db["datasets"][
            new_name
        ]  # Access the new collection (it will be created if it doesn't exist)

        # Remove the '_id' field from the document to be inserted.
        # MongoDB automatically generates a new unique _id for each new insertion,
        # preventing 'DuplicateKeyError' if the original _id were kept.
        if "_id" in doc:
            del doc["_id"]

        try:
            inserted_result = target_collection.insert_one(doc)
            results[new_name] = {
                "status": "success",
                "inserted_id": str(inserted_result.inserted_id),
                "message": f"Document copied to {new_name}",
            }
        except Exception as e:
            results[new_name] = {
                "status": "error",
                "message": f"Failed to insert document into {new_name}: {str(e)}",
            }

    return jsonify(json.dumps("Done", default=str)), 200


if __name__ == "__main__":
    app.run(host=app_settings.host, port=app_settings.port, debug=app_settings.debug)
