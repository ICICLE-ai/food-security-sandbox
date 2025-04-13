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
import URL from '../../config';

const SimilarFarmers = ({userName, selectedDataset}) => {
  const navigate = useNavigate();
  const [farmers, setFarmers] = useState([]);

  const handleFindFarmers = () => {
    // Simply set the mock data
    const fetchProfiles = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await axios.post(`${URL}/api/get_similar_farmers`, {selectedDataset},{
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
    // setFarmers([
    //   { id: 1, name: "John Smith", similarity: "90%" },
    //   { id: 2, name: "Maria Garcia", similarity: "85%" },
    //   { id: 3, name: "David Johnson", similarity: "82%" },
    //   { id: 4, name: "Sarah Wilson", similarity: "78%" },
    // ]);
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
        <Button
          variant="contained"
          onClick={handleFindFarmers}
          sx={{ 
            backgroundColor: '#008000',
            '&:hover': {
              backgroundColor: '#009900',
            },
            px: 4
          }}
        >
          Find Similar Farmers
        </Button>
      </Box>

      {farmers.length > 0 && (
        <List sx={{ width: '100%', bgcolor: 'background.paper' }}>
          {farmers.map((farmer, index) => (
            <React.Fragment key={farmer.id}>
              <ListItem alignItems="center">
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: '#008000' }}>
                    <PersonIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={farmer.name}
                  secondary={`Type: Farmer`}
                />
                <Button 
                  onClick={(event) => {
                    event.stopPropagation();
                    //handleChat(farmer.id);
                    console.log(farmer.id);
                    navigate(`/chat/?${farmer.id}`);
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

    </Box>
  );
};

export default SimilarFarmers; 