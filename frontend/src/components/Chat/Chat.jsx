import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Chat.css';
import { useLocation } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';

function ChatApp() {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const [senderID, setSenderID] = useState(null); // Initialize senderID to null
  const [receiverID, setReceiverID] = useState(null); // Initialize receiverID to null as well
  const [loading, setLoading] = useState(true); // Initially set loading to true
  const rID = queryParams.get('receiver_id');
  const navigate = useNavigate();

  // Fetch username on component mount
  useEffect(() => {
    const fetchUsername = async () => {
      const username = localStorage.getItem('tapis_username');
      if (username) {
        setSenderID(username);
      } else {
        navigate("/login");
      }
      setLoading(false); // Set loading to false after attempting to get senderID
    };

    fetchUsername();

    if (rID != null) {
      setReceiverID(rID);
    }
  }, [navigate, rID]);

  return (
    <div className='chatApp'>
      {loading ? (
        <div>Loading</div>
      ) : senderID !== null ? ( // Conditionally render only if senderID is loaded
        <>
          <Conversations senderID={senderID} receiverID={receiverID} setReceiverID={setReceiverID} setLoading={setLoading} />
          <ChatBox senderID={senderID} receiverID={receiverID} setLoading={setLoading} />
        </>
      ) : (
        <div>Redirecting to login...</div> // Or some other fallback UI
      )}
    </div>
  );
}

const Conversations = ({ senderID, receiverID, setReceiverID, setLoading }) => {
  const [conversations, setConversations] = useState([]);

  useEffect(() => {
    if (senderID) { // Only fetch conversations if senderID is available
      const token = localStorage.getItem('tapis_token');
      axios.post(`/api/conversations`,{"sender_id":senderID},{
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }).then((response) => {
          console.log(response.data);
          setConversations(response.data);
        })
        .catch((error) => console.error(error));
    }
  }, [senderID]); // Re-run effect when senderID changes

  return (
    <div className='conversations'>
      <div className="header">
        <h2>Conversations</h2>
      </div>
      {conversations.map((msg, index) => (
        <div
          key={index}
          className={receiverID==msg.partner_id?`selectedConversationItem`:`converationItem`}
          onClick={() => setReceiverID(msg.partner_id)}
        >
          <p>{msg.partner_id}</p>
        </div>
      ))}
    </div>
  );
};

const ChatBox = (({ senderID, receiverID, setLoading }) => {
  const [messages, setMessages] = useState([]);
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (senderID && receiverID !== null) { // Only fetch messages if both IDs are available
      axios
        .post(`/api/getMessages`,{"sender_id":senderID,"receiver_id":receiverID},{
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('tapis_token')}`
          }
      })
        .then((response) => {
          console.log(response.data);
          setMessages(response.data);
          setLoading(false);
        })
        .catch((error) => console.error(error));
    }
  }, [senderID, receiverID]); // Re-run effect when senderID or receiverID changes

  // Send message function
  const sendMessage = () => {
    if (senderID && receiverID !== null && message) {
      const timestamp = new Date().toISOString();
      axios
        .post(`/api/sendMessage`, {
          senderID,
          receiverID,
          message,
          timestamp
        },{
            headers: {
            'Authorization': `Bearer ${localStorage.getItem('tapis_token')}`
          }
        })
        .then((response) => {
          setMessages([...messages, { senderID, receiverID, message, timestamp }]);
          setMessage(""); // Clear input
        })
        .catch((error) => console.error(error));
    }
  };

  const formatDate = (dateString) => {
    const options = {
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  return (
    <div className="chat-container">
      <div className="header">
        <h2>{receiverID === null ? "No Conversation Selected" : receiverID}</h2>
      </div>
      <div className="messages">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.senderID === senderID ? "sent" : "received"}`}
          >
            <div className="message-content">
              <p>{msg.message}</p>
              <p className="timestamp">{formatDate(msg.timestamp)}</p>
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
});

export default ChatApp;