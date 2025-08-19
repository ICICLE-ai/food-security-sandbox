// web_application/Frontend/src/components/Training/TrainingComponent.jsx
import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Snackbar, Avatar, TextField } from '@mui/material';
import axios from 'axios';
import './ModelTraining.css';
import PersonIcon from '@mui/icons-material/Person';


const ModelTraining = ({userName, selectedDataset}) => {
  const [trainingLog, setTrainingLog] = useState('');
  const [selectedModel, setSelectedModel] = useState("Dense");
  const [modelName, setModelName] = useState("");
  const [modelReadme, setModelReadme] = useState("");
  const [modelVisibility, setModelVisibility] = useState("Public");
  const [collaborators, setCollaborators] = useState([]);
  const [selectedCollaborators, setSelectedCollaborators] = useState([]);
  const [collaborationStatus, setCollaborationStatus] = useState(false);
  const token = localStorage.getItem('tapis_token');

  useEffect(() => {
    const fetchSimilarFarmers = async () => {
      console.log('Identifying Collaborators')
      const response = await axios.post(`/api/get_similar_farmers`, {selectedDataset},{
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      console.log(response.data.collaborators)
      console.log('Starting Training Process')
      setCollaborators(response.data.collaborators);
      setSelectedCollaborators(response.data.collaborators);
    }
    fetchSimilarFarmers();

  }, []);

  const handleStartTraining = async () => {

      if(modelName == ""){
        alert('Please Enter Model Name!')
        return;
      }

      

      axios.post(`/api/train`,
        {'collaborators': selectedCollaborators, 
          hyperparameters: {
                'modelName': modelName, 
                'modelType':selectedModel, 
                'modelVisibility':modelVisibility, 
                'datasetName': selectedDataset,
                'readme': modelReadme
          }
        },
        {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      }).then((response) => {
          console.log(response.data);
          setTrainingLog("Model Training In Progress");
        })
        .catch((error) => console.error(error));
  };

  const handleCollaboratorToggle = (collaborator) => {
    setSelectedCollaborators(prev => {
      const isSelected = prev.some(c => c.username === collaborator.username);
      if (isSelected) {
        return prev.filter(c => c.username !== collaborator.username);
      } else {
        return [...prev, collaborator];
      }
    });
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
      {collaborators.length === 0? (<>
        <h3 style={{ color: 'red', textAlign: 'center' }}>No Collaborators Found</h3>
      </>):
      trainingLog === "" ? (
        <>
      <div className="mb-4 p-4 border rounded-lg bg-white shadow-sm">
        <h3 className="text-lg font-semibold mb-3">Training Parameters</h3>
          <div className='formRow'>
            <label htmlFor="modelSelect" className="labelHeading">
              Model Name: 
            </label>
            <input 
              type='text'
              id='modelName'
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder='Model Name'
            />
          </div>
          <div  className='formRow'>
            <label htmlFor="modelSelect" className="labelHeading">
              Model Type: 
            </label>
            <select
              id="modelSelect"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="selectBox"
            >
              <option value="Dense">Sequential Dense</option>
              <option value="Conv1d">Conv 1D</option>
              <option value="LSTM">LSTM</option>
            </select>
          </div>
          <div  className='formRow'>
            <label htmlFor="modelSelect" className="labelHeading">
              Model Visibility: 
            </label>
            <select
              id="modelSelect"
              value={modelVisibility}
              onChange={(e) => setModelVisibility(e.target.value)}
              className="selectBox"
            >
              <option value="Public">Public</option>
              <option value="Private">Private</option>
            </select>
          </div>
          <div  className='formRow'>
            <label htmlFor="modelSelect" className="labelHeading">
              Readme (Optional): 
            </label>
            <TextField
              id='multiLineField'
              hintText="Readme instruction to use the model."
              placeholder="Readme instruction to use the model."
              multiline
              rows={4}
              variant="outlined"
              fullWidth
              onChange={(e) => setModelReadme(e.target.value)}
            />            
          </div>
          <div className='formRow'>
            <label htmlFor="modelSelect" className="labelHeading">Collaborators:</label>
            <div className='collaborators'>
              {collaborators.map((collaborator, index) => (
                <div key={index} className="collaborator-item">
                  <input
                    type="checkbox"
                    checked={selectedCollaborators.some(c => c.username === collaborator.username)}
                    onChange={() => handleCollaboratorToggle(collaborator)}
                    className="collaborator-checkbox"
                  />
                  <Avatar className='collaborator-avatar' sx={{ width: 30, height: 30, bgcolor: '#008000' }}>
                    <PersonIcon />
                  </Avatar>
                  <label className="collaborator-name">{collaborator.username}</label>
                </div>
              ))}
            </div>
          </div>
      </div>
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 , paddingTop: 2}}>
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
      </>
      ) : (
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
          <Typography>{trainingLog}</Typography>
        </Box>
      )}
      
    </Box>
  );
};

export default ModelTraining;