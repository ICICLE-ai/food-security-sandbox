import { useState, useEffect } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { Typography, List, ListItem, ListItemIcon, Box, Grid, Snackbar, CircularProgress } from '@mui/material';
import UploadForm from '../Upload/Upload';
import SearchPage from '../Search/Search';
import SimilarFarmers from '../SimilarFarmers/SimilarFarmers';
import UploadedDatasets from '../UploadedDatasets/UploadedDatasets';
import axios from 'axios';
import FunctionsIcon from '@mui/icons-material/Functions';

const HomePage = () => {
  const [message, setMessage] = useState('');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [userName, setUserName] = useState('');
  const [userID, setUserID] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState('')
  const [update, setUpdate] = useState(false)
  const [selectedFunction, setSelectedFunction] = useState('')
  const [onFunctionSelected, setOnFunctionSelected] = useState(false)
  const functions = [
    { name: 'Finding Similar Farmers', logo: <FunctionsIcon fontSize='large'/> },
    { name: 'Link Public Private Dataset For Farmer ', logo: <FunctionsIcon fontSize='large'/> },
    { name: 'Link Public Private Dataset For Food Security', logo: <FunctionsIcon fontSize='large'/> },
    { name: 'Link Farm2Fact to Production Data', logo: <FunctionsIcon fontSize='large'/> },
    // Add more functions as needed
  ];
  const navigate = useNavigate();

  const getToken = () => localStorage.getItem('tapis_token');
  const getUserName = () => localStorage.getItem('tapis_username');
  const token = getToken();

  useEffect(() => {
    const fetchUserName = async () => {
      try {
        
        setUserName(getUserName());
        setUserID(getUserName())
      } catch (error) {
        setError('There was an error fetching the user name!');
        console.error('There was an error fetching the user name!', error);
      } finally {
        setLoading(false);
      }
    };

    if (token) { 
      fetchUserName();
    } else {
      setError('No token found.');
      setLoading(false);
      navigate("/")
    }
    console.log(selectedDataset,"...........")
  }, [token,selectedDataset]);

  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };

  if (loading) return (
    <Box 
      sx={{ 
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
      }}
    >
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        borderRadius: '50%',
        padding: 2
      }}>
        <CircularProgress size={60} sx={{ color: '#008000' }} />
      </Box>
    </Box>
  );

  if (error) return (
    <Box sx={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      height: '100vh',
      color: 'error.main'
    }}>
      <Typography variant="h6">{error}</Typography>
    </Box>
  );

  return (
    <Box sx={{ mt: 2 }}>
      <Typography 
        variant="h4" 
        gutterBottom 
        sx={{ 
          fontWeight: 'bold', 
          color: '#333', 
          justifyContent: 'center', 
          display: 'flex',
          mb: 2
        }}
      >
        {onFunctionSelected? `${selectedFunction}`:`Welcome Back, ${userName.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(' ')}`}
      </Typography>
      {!onFunctionSelected ?
      <Grid container spacing={1} justifyContent="center">
        <Grid item xs={12} sm={6}>
        <Box>
          <Typography variant="h6" gutterBottom>
            Available Functions
          </Typography>
          <List>
            {functions.map((func, index) => (
              <ListItem
              button // This makes the ListItem clickable
              key={index}
              onClick={() => {setSelectedFunction(func.name); setOnFunctionSelected(true)}} 
              sx={{ 
                border: '1px solid #ccc', // Add border
                borderRadius: 1, // Optional: rounded corners
                mb: 1, // Optional: margin bottom for spacing between items
                padding: 1, // Optional: padding for better spacing
                cursor: 'pointer', // Ensures that the cursor is a hand on hover
                '&:hover': {
                  backgroundColor: '#f0f0f0', // Optional: change background color on hover
                }
              }}
            >
                <ListItemIcon sx={{ minWidth: 40 }}>{func.logo}</ListItemIcon>
                <Typography variant="body1" sx={{ fontSize: '1.5rem' }}>{func.name}</Typography>
              </ListItem>
            ))}
          </List>
        </Box>
        </Grid>
      </Grid>:
      <Grid container spacing={2} justifyContent="center">
        <Grid item xs={12} sm={6}>
          <Box sx={{ 
            border: '1px solid #ccc', 
            p: 2, 
            m: 1,
            borderRadius: 2, 
            height: '450px',
            overflow: 'auto' 
          }}>
            <UploadForm setUpdate={setUpdate}/>
          </Box>
        </Grid>
        <Grid item xs={12} sm={6}>
          <Box sx={{ 
            border: '1px solid #ccc', 
            p: 2, 
            m: 1,
            borderRadius: 2, 
            height: '450px',
            overflow: 'auto' 
          }}>
            {selectedDataset == ""?
              <UploadedDatasets  userName={userName} setSelectedDataset={setSelectedDataset} update={update}/>
              :
              <SimilarFarmers userName={userName} selectedDataset={selectedDataset} userID={userID}/>
            }
          </Box>
        </Grid>
      </Grid>
      }
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message={message}
      />
    </Box>
  );
};

export default HomePage;
