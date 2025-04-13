import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Chat.css';

function ChatApp() {
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState("");
  const senderID = 1001;
  const receiverID = 1002;

  // Fetch messages on component mount
  useEffect(() => {
    axios
      .get(`http://localhost:5000/getMessages?sender_id=${senderID}&receiver_id=${receiverID}`)
      .then((response) => {
        console.log(response)
        setMessages(response.data);
      })
      .catch((error) => console.error(error));
  }, [senderID, receiverID]);

  // Send message function
  const sendMessage = () => {
    const timestamp = new Date().toISOString();
    axios
      .post("http://localhost:5000/sendMessage", {
        senderID,
        receiverID,
        message,
        timestamp
      })
      .then((response) => {
        setMessages([...messages, { senderID, receiverID, message, timestamp }]);
        setMessage(""); // Clear input
      })
      .catch((error) => console.error(error));
  };

  return (
    <div className="chat-container">
      <div className="header">
        <h2>Chat</h2>
      </div>
      <div className="messages">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.senderID === senderID ? "sent" : "received"}`}
          >
            <div className="message-content">
              <p>{msg.message}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="input-container">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type a message"
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default ChatApp;
