import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './Chat.css';
import { useLocation } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import socket from '../../socket';

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
      axios.post(`${process.env.REACT_APP_API_URL}/conversations`,{"sender_id":senderID},{
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

  useEffect(() => {
    if (!senderID) return;

    const handleNewMessage = (msg) => {
      const partnerId = msg.senderID === senderID ? msg.receiverID : msg.senderID;
      setConversations((prev) => {
        const existing = prev.find((c) => c.partner_id === partnerId);
        const updated = {
          partner_id: partnerId,
          timestamp: msg.timestamp,
          last_message: msg.message,
        };
        const rest = prev.filter((c) => c.partner_id !== partnerId);
        return [updated, ...rest];
      });
    };

    socket.on('new_message', handleNewMessage);
    return () => socket.off('new_message', handleNewMessage);
  }, [senderID]);

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
  const [liveArrivals, setLiveArrivals] = useState(new Set());
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (senderID && receiverID !== null) { // Only fetch messages if both IDs are available
      axios
        .post(`${process.env.REACT_APP_API_URL}/getMessages`,{"sender_id":senderID,"receiver_id":receiverID},{
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

  useEffect(() => {
    if (!senderID || receiverID === null) return;

    const handleNewMessage = (msg) => {
      const belongsToOpenConversation =
        (msg.senderID === senderID && msg.receiverID === receiverID) ||
        (msg.senderID === receiverID && msg.receiverID === senderID);
      if (belongsToOpenConversation) {
        setLiveArrivals((prev) => new Set(prev).add(`${msg.senderID}-${msg.timestamp}`));
        setMessages((prev) => [...prev, msg]);
      }
    };

    socket.on('new_message', handleNewMessage);
    return () => socket.off('new_message', handleNewMessage);
  }, [senderID, receiverID]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send message function
  const sendMessage = () => {
    if (senderID && receiverID !== null && message) {
      const timestamp = new Date().toISOString();
      axios
        .post(`${process.env.REACT_APP_API_URL}/sendMessage`, {
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
          setMessage(""); // Clear input - the sent message arrives back via the 'new_message' socket event
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
            className={`message ${msg.senderID === senderID ? "sent" : "received"} ${liveArrivals.has(`${msg.senderID}-${msg.timestamp}`) ? "message-arriving" : ""}`}
          >
            <div className="message-content">
              <p>{msg.message}</p>
              <p className="timestamp">{formatDate(msg.timestamp)}</p>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="input-container">
        <input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') sendMessage(); }}
          placeholder="Type a message"
        />
        <button onClick={sendMessage} disabled={!message.trim()}>Send</button>
      </div>
    </div>
  );
});

export default ChatApp;
