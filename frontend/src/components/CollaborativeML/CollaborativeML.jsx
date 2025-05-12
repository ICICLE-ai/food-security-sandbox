// web_application/Frontend/src/components/Training/TrainingComponent.jsx
import axios from 'axios';
import { useState, useEffect } from 'react';
import { Typography, Box, Grid, Snackbar, CircularProgress } from '@mui/material';
import UploadForm from '../Upload/Upload';
import SimilarFarmers from '../SimilarFarmers/SimilarFarmers';
import UploadedDatasets from '../UploadedDatasets/UploadedDatasets';
import ModelTraining from '../ModelTraining/ModelTraining';
import ModelRepository from '../ModelRepository/ModelRepository'

const CollaborativeML = () => {
  const [message, setMessage] = useState('');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [userID, setUserID] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState('')

  const getToken = () => localStorage.getItem('tapis_token');
  const getUserName = () => localStorage.getItem('tapis_username');
  const token = getToken();

  useEffect(() => {
    const fetchUserName = async () => {
      try {
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
        Collaborative Machine Learning
      </Typography>
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
            {selectedDataset == ""?
              <UploadedDatasets  userName={userID} setSelectedDataset={setSelectedDataset}/>
              :
              <ModelTraining  userName={userID} selectedDataset={selectedDataset}/>

            }
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
              <ModelRepository userName={userID} />
          </Box>
        </Grid>
      </Grid>
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        message={message}
      />
    </Box>
  );
};

export default CollaborativeML;
