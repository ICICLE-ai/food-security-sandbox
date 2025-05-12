// web_application/Frontend/src/components/Training/TrainingComponent.jsx
import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Snackbar } from '@mui/material';
import axios from 'axios';


const ModelTraining = ({userName, selectedDataset}) => {
  const [trainingLog, setTrainingLog] = useState('');
  const [selectedModel, setSelectedModel] = useState("LR");

  const handleStartTraining = async () => {
      const token = localStorage.getItem('tapis_token');
      console.log('Identifying Collaborators')
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/get_similar_farmers`, {selectedDataset},{
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      console.log(response.data.collaborators)
      console.log('Starting Training Process')
      

      axios.post(`${process.env.REACT_APP_API_URL}/train`,{'collaborators': response.data.collaborators, hyperparameters: {'modelName':selectedModel}},{
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }).then((response) => {
          console.log(response.data);
          setTrainingLog("Model Training In Progress");
        })
        .catch((error) => console.error(error));
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
      <div className="mb-4 p-4 border rounded-lg bg-white shadow-sm">
        <h3 className="text-lg font-semibold mb-3">Training Parameters</h3>
        <div className="mb-3">
          <label htmlFor="modelSelect" className="block text-sm font-medium text-gray-700 mb-1">
            Model Name
          </label>
          <select
            id="modelSelect"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="LR">Logistic Regression</option>
            <option value="KNN">K-Nearest Neighbors</option>
            <option value="SVM">Support Vector Machine</option>
            <option value="RF">Random Forest</option>
            <option value="DT">Decision Tree</option>
          </select>
        </div>
      </div>
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
        <Button
          variant="contained"
          onClick={handleStartTraining}
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