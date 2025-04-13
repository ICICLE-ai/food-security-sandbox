from flask import Flask, request, jsonify
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb+srv://osamazafar98:MseZi2o4OHRdZvTT@cluster0.ydfwh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
  # your MongoDB URI
mongo = PyMongo(app)

@app.route("/getMessages", methods=["GET"])
def get_messages():
    sender_id = request.args.get("sender_id")
    receiver_id = request.args.get("receiver_id")
    
    # Fetching messages from the database
    messages = mongo.db.messages.find({
        "$or": [
            {"senderID": sender_id, "receiverID": receiver_id},
            {"senderID": receiver_id, "receiverID": sender_id}
        ]
    })

    # Formatting the response
    messages_list = []
    for message in messages:
        messages_list.append({
            "senderID": message["senderID"],
            "receiverID": message["receiverID"],
            "message": message["message"],
            "timestamp": message["timestamp"]
        })
    
    return jsonify(messages_list)

@app.route("/sendMessage", methods=["POST"])
def send_message():
    data = request.json
    message = {
        "senderID": data["senderID"],
        "receiverID": data["receiverID"],
        "message": data["message"],
        "timestamp": data["timestamp"]
    }

    mongo.db.messages.insert_one(message)
    return jsonify({"status": "Message sent"}), 201

if __name__ == "__main__":
    app.run(debug=True)
