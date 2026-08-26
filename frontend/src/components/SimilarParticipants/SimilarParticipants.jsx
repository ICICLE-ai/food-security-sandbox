import React, { useState } from 'react';
import {
  Typography,
  Box,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Divider
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import PersonIcon from '@mui/icons-material/Person';
import ChatIcon from '@mui/icons-material/Chat';
import axios from 'axios';


const SimilarParticipants = ({userName, selectedDataset, userID}) => {
  const navigate = useNavigate();
  const [participants, setParticipants] = useState([]);
  const [identifyParticipantsClicked, setIdentifyParticipantsClicked] = useState(false);
  const handleFindParticipants = () => {
    setIdentifyParticipantsClicked(true);
    // Simply set the mock data
    const fetchProfiles = async () => {
      try {
        const token = localStorage.getItem('tapis_token');
        const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/get_similar_participants`, {selectedDataset},{
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        console.log(response.data.collaborators)
        setParticipants(response.data.collaborators)

      } catch (error) {
        console.error('Error fetching datasets:', error);
      }
    };

    fetchProfiles();
  };

  return (
    <Box sx={{ p: 2 }}>
      <Typography
        variant="h5"
        gutterBottom
        sx={{
          fontWeight: 'light',
          textAlign: 'center',
          mb: 3
        }}
      >
        Finding Similar Participants
      </Typography>

      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
        {identifyParticipantsClicked == false ?<Button
          variant="contained"
          onClick={handleFindParticipants}
          sx={{
            backgroundColor: 'primary.main',
            '&:hover': {
              backgroundColor: 'primary.dark',
            },
            px: 4,
            minWidth: 'fit-content',
            whiteSpace: 'nowrap',
            width: 'auto'
          }}
        >
          Find Similar Participants
        </Button>:<></>
        }
      </Box>

      {participants.length > 0 && (
        <List sx={{ width: '100%', bgcolor: 'background.paper' }}>
          {participants.map((participant, index) => (
            <React.Fragment key={index}>
              <ListItem alignItems="center">
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: 'primary.main' }}>
                    <PersonIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={participant.username}
                  secondary="Participant"
                />
                <Button
                  onClick={(event) => {
                    event.stopPropagation();
                    //handleChat(participant.id);
                    navigate(`/chat/?receiver_id=${participant.username}`);
                  }}
                  sx={{
                    color: 'primary.main',
                    minWidth: '40px'
                  }}
                >
                  <ChatIcon sx={{ color: 'primary.main' }}/>
                </Button>
              </ListItem>
              {index < participants.length - 1 && <Divider variant="inset" component="li" />}
            </React.Fragment>
          ))}
        </List>
      )}
      {(participants.length == 0 && !identifyParticipantsClicked) && (
        <Box sx={{ width: '100%', bgcolor: 'background.paper', textAlign: 'center', justifyContent:'center', alignItems: 'center' }}>
        <h3 style={{color:'Green'}}>Click Identify to find similar participants.</h3>
        </Box>
        )}
      {(participants.length == 0 && identifyParticipantsClicked) && (
        <Box sx={{ width: '100%', bgcolor: 'background.paper', textAlign: 'center', justifyContent:'center', alignItems: 'center' }}>
        <h3 style={{color:'red'}}>No similar participants identified.</h3>
        </Box>
        )}
    </Box>
  );
};

export default SimilarParticipants;
