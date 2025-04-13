import React from 'react';
import { Button } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import URL from '../../config';



const CollaborationRequestButton = ({ receiverId, senderId }) => {
  const handleClick = () => {
    // Prepare data to send in the request
    const requestData = {
      receiver_id: receiverId,
      sender_id: senderId,
      // research_project_id: projectId
    };

    // Send POST request to the backend endpoint
    fetch(`${URL}/api/sendinvitation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      // Handle successful response
      console.log('Invitation sent successfully:', data);
      // You can perform further actions here if needed
    })
    .catch(error => {
      // Handle error
      console.error('There was a problem sending the invitation:', error);
      // You can display an error message or take other actions as needed
    });
  };

  return (
    <Button 
      variant="contained" 
      color="primary" 
      startIcon={<SendIcon />} 
      onClick={handleClick}
    >
      Send Request
    </Button>
  );
};

export default CollaborationRequestButton;
