// web_application/Frontend/src/components/Training/TrainingComponent.jsx
import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Snackbar } from '@mui/material';
import axios from 'axios';
import URL from '../../config';

const ModelTraining = ({userName, selectedDataset}) => {
  const [trainingLog, setTrainingLog] = useState('');

  const handleStartTraining = async () => {
    setTrainingLog("Model Training Started")
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
        Model Training
      </Typography>
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
      <Button variant="contained" onClick={handleStartTraining}>
        Start Training
      </Button>
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
        <Typography>{trainingLog}</Typography>
      </Box>
      
    </Box>
  );
};

export default ModelTraining;