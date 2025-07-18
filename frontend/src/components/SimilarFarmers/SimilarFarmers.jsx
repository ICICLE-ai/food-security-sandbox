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


const SimilarFarmers = ({userName, selectedDataset, userID}) => {
  const navigate = useNavigate();
  const [farmers, setFarmers] = useState([]);
  const [identifyFarmersClicked, setIdentifyFarmersClicked] = useState(false);
  const handleFindFarmers = () => {
    setIdentifyFarmersClicked(true);
    // Simply set the mock data
    const fetchProfiles = async () => {
      try {
        const token = localStorage.getItem('tapis_token');
        const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/get_similar_farmers`, {selectedDataset},{
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        console.log(response.data.collaborators)
        setFarmers(response.data.collaborators)

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
        Finding Similar Farmers
      </Typography>

      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
        {identifyFarmersClicked == false ?<Button
          variant="contained"
          onClick={handleFindFarmers}
          sx={{ 
            backgroundColor: '#008000',
            '&:hover': {
              backgroundColor: '#009900',
            },
            px: 4,
            minWidth: 'fit-content',
            whiteSpace: 'nowrap',
            width: 'auto'
          }}
        >
          Find Similar Farmers
        </Button>:<></>
        }
      </Box>

      {farmers.length > 0 && (
        <List sx={{ width: '100%', bgcolor: 'background.paper' }}>
          {farmers.map((farmer, index) => (
            <React.Fragment key={index}>
              <ListItem alignItems="center">
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: '#008000' }}>
                    <PersonIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={farmer.username}
                  secondary={`Type: Farmer`}
                />
                <Button 
                  onClick={(event) => {
                    event.stopPropagation();
                    //handleChat(farmer.id);
                    navigate(`/chat/?receiver_id=${farmer.username}`);
                  }}
                  sx={{ 
                    color: 'primary.main',
                    minWidth: '40px'
                  }}
                >
                  <ChatIcon style={{ color: '#008000'}}/>
                </Button>
              </ListItem>
              {index < farmers.length - 1 && <Divider variant="inset" component="li" />}
            </React.Fragment>
          ))}
        </List>
      )}
      {(farmers.length == 0 && !identifyFarmersClicked) && (
        <Box sx={{ width: '100%', bgcolor: 'background.paper', textAlign: 'center', justifyContent:'center', alignItems: 'center' }}>
        <h3 style={{color:'Green'}}>Click Identify Similar Farmer Identified.</h3>
        </Box>
        )}
      {(farmers.length == 0 && identifyFarmersClicked) && (
        <Box sx={{ width: '100%', bgcolor: 'background.paper', textAlign: 'center', justifyContent:'center', alignItems: 'center' }}>
        <h3 style={{color:'red'}}>No Similar Farmer Identified.</h3>
        </Box>
        )}
    </Box>
  );
};

export default SimilarFarmers; 