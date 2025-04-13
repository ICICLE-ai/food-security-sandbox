import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Paper, 
  TextField, 
  Button, 
  List, 
  ListItem, 
  Typography, 
  Avatar,
  Divider,
  ListItemAvatar,
  ListItemText,
  IconButton,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import PersonIcon from '@mui/icons-material/Person';

const Chat = () => {
  const [conversations, setConversations] = useState([]);
  const [selectedConversation, setSelectedConversation] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const messagesEndRef = useRef(null);
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    // Load conversations
    loadConversations();
  }, []);

  useEffect(() => {
    if (selectedConversation) {
      loadChatHistory(selectedConversation.id);
    }
  }, [selectedConversation]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadConversations = async () => {
    try {
      const response = await fetch('/api/chat/conversations', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setConversations(data);
      }
    } catch (error) {
      console.error('Error loading conversations:', error);
    }
  };

  const loadChatHistory = async (userId) => {
    try {
      const response = await fetch(`/api/chat/history/${userId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setMessages(data);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  const handleSend = async () => {
    if (!newMessage.trim() || !selectedConversation) return;

    try {
      const response = await fetch('/api/chat/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          receiver_id: selectedConversation.id,
          message: newMessage
        })
      });

      if (response.ok) {
        setNewMessage('');
        loadChatHistory(selectedConversation.id);
      }
    } catch (error) {
      console.error('Error sending message:', error);
    }
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      height: '90vh',
      bgcolor: '#f5f5f5',
      p: 2,
      gap: 2
    }}>
      {/* Conversations List */}
      <Paper sx={{ 
        width: '300px',
        display: 'flex',
        flexDirection: 'column'
      }}>
        <Typography variant="h6" sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          Conversations
        </Typography>
        <List sx={{ flex: 1, overflow: 'auto' }}>
          {conversations.map((conversation) => (
            <ListItem
              key={conversation.id}
              button
              selected={selectedConversation?.id === conversation.id}
              onClick={() => setSelectedConversation(conversation)}
              sx={{
                '&.Mui-selected': {
                  backgroundColor: '#e3f2fd',
                }
              }}
            >
              <ListItemAvatar>
                <Avatar>
                  <PersonIcon />
                </Avatar>
              </ListItemAvatar>
              <ListItemText 
                primary={conversation.name}
                secondary={conversation.lastMessage}
                secondaryTypographyProps={{
                  noWrap: true,
                  style: { maxWidth: '200px' }
                }}
              />
            </ListItem>
          ))}
        </List>
      </Paper>

      {/* Chat Messages */}
      <Paper sx={{ 
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
      }}>
        {selectedConversation ? (
          <>
            <Box sx={{ 
              p: 2, 
              borderBottom: 1, 
              borderColor: 'divider',
              bgcolor: '#e3f2fd'
            }}>
              <Typography variant="h6">
                {selectedConversation.name}
              </Typography>
            </Box>

            <Box sx={{ 
              flex: 1, 
              overflow: 'auto', 
              p: 2,
              display: 'flex',
              flexDirection: 'column',
              gap: 1
            }}>
              {messages.map((message) => (
                <Box
                  key={message._id}
                  sx={{
                    alignSelf: message.sender_id === selectedConversation.id ? 'flex-start' : 'flex-end',
                    maxWidth: '70%'
                  }}
                >
                  <Paper
                    sx={{
                      p: 1,
                      bgcolor: message.sender_id === selectedConversation.id ? '#fff' : '#e3f2fd',
                      borderRadius: 2
                    }}
                  >
                    <Typography variant="body1">{message.message}</Typography>
                    <Typography variant="caption" color="textSecondary">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </Typography>
                  </Paper>
                </Box>
              ))}
              <div ref={messagesEndRef} />
            </Box>

            <Box sx={{ 
              p: 2, 
              borderTop: 1, 
              borderColor: 'divider',
              display: 'flex',
              gap: 1
            }}>
              <TextField
                fullWidth
                variant="outlined"
                size="small"
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Type a message..."
              />
              <IconButton 
                color="primary"
                onClick={handleSend}
                disabled={!newMessage.trim()}
              >
                <SendIcon />
              </IconButton>
            </Box>
          </>
        ) : (
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <Typography variant="h6" color="textSecondary">
              Select a conversation to start messaging
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default Chat;
