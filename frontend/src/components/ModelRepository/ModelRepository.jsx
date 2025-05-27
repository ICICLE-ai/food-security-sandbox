import React, { useState, useEffect } from 'react';
import { 
  Typography, 
  Box, 
  Button, 
  List, 
  ListItem, 
  ListItemText, 
  ListItemAvatar,
  Avatar,
  Divider,
  Modal 
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import axios from 'axios';
import InfoIcon from '@mui/icons-material/Info'; // Import InfoIcon
import CloseIcon from '@mui/icons-material/Close';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import PlayCircleIcon from '@mui/icons-material/PlayCircle';
import './ModelRepository.css'

const ModelRepository = ({userName}) => {
  const [models, setModels] = useState([]);
  const [openModal, setOpenModal] = useState(false); // State for modal visibility
  const [selectedModelInfo, setSelectedModelInfo] = useState(null); // State for selected dataset info
  const [openPlaygroundModal, setOpenPlaygroundModal] = useState(false); // State for modal visibility
  const [evalInputData, setEvalInputData] = useState('');

  const handleOpenModal = (model) => {
    console.log('modal open')
    setSelectedModelInfo(model); // Set the selected dataset info
    setOpenModal(true); // Open the modal
  };
  const handleCloseModal = () => {
    setOpenModal(false); // Close the modal
  };

  const handleOpenPlaygroundModal = (model) => {
    console.log('modal open')
    setSelectedModelInfo(model); // Set the selected dataset info
    setOpenPlaygroundModal(true); // Open the modal
  };
  const handleClosePlaygroundModal = () => {
    setOpenPlaygroundModal(false); // Close the modal
  };

  const handlePredictButton = () => {
    try {
        if (evalInputData === '') {
          alert('Pleas Enter Test Data!')
        }
        else{
          axios.post(
            `${process.env.REACT_APP_API_URL}/api/predict_model`,
            {
              model_info: selectedModelInfo,
              eval_data: evalInputData
            },
            {
              headers: {
                'Authorization': `Bearer ${localStorage.getItem('tapis_token')}`
              }
            }
          ).then((response) => {
            console.log('Prediction response:', response.data);
          })
        }   

        } catch (error) {
          console.error('Error making prediction:', error);
        }
    
  }

  useEffect(() => {
    // Simply set the mock data

    const fetchModels = async () => {
      try {

        var responsePublic = await axios.get(`${process.env.REACT_APP_API_URL}/api/get_public_models`, {
          headers: {
          'Authorization': `Bearer ${localStorage.getItem('tapis_token')}`
          }
        })
        var responsePrivate = await axios.get(`${process.env.REACT_APP_API_URL}/api/get_user_models`, {
          headers: {
          'Authorization': `Bearer ${localStorage.getItem('tapis_token')}`
          }
        })

        var models = [...JSON.parse(responsePublic.data), ...JSON.parse(responsePrivate.data)]
        console.log(models)
        setModels(models); // Set the datasets state with the fetched data
        
      } catch (error) {
        console.error('Error fetching datasets:', error);
      }
    };
    fetchModels();
  },[]);

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
        Model Repository
      </Typography>

      {models.length > 0 && (
        <List sx={{ width: '100%', bgcolor: 'background.paper' }}>
          {models.map((model, index) => (
            <React.Fragment key={index}>
              <ListItem alignItems="center">
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: '#008000' }}>
                    <PersonIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={model.modelName}
                  secondary={`Model Type: ${model.modelType}`}
                />

                  <ModelVisibilityIcon modelVisibility={model.modelVisibility}/>

                <Button 
                    onClick={(event) => {
                      event.stopPropagation(); // Prevent ListItem onClick
                      handleOpenModal(model);
                    }}
                  >
                    <InfoIcon />
                </Button>
                <Button 
                    onClick={(event) => {
                      event.stopPropagation(); // Prevent ListItem onClick
                      handleOpenPlaygroundModal(model);
                    }}
                  >
                    <PlayCircleIcon/>
                </Button>
              </ListItem>
              {index < models.length - 1 && <Divider variant="inset" component="li" />}
            </React.Fragment>
          ))}
        </List>
      )}
    <Modal open={openModal} onClose={handleCloseModal}>
        <Box 
          sx={{ 
            bgcolor: 'white', 
            border: '1px solid black', 
            borderRadius: '4px', 
            padding: 2, 
            position: 'relative', 
            width: '40%', // Set a width for the modal
            margin: 'auto', // Center the modal
            top: '50%', // Center vertically
            transform: 'translateY(-50%)' // Adjust for vertical centering
          }}
        >
          <Button 
            onClick={handleCloseModal} 
            sx={{ 
              position: 'absolute', 
              top: 8, 
              right: 8 
            }}
          >
            <CloseIcon />
          </Button>
          <Typography variant="h6">{selectedModelInfo?.modelName}</Typography>
          <Typography>Model Type: {selectedModelInfo?.modelType}</Typography>
          <Typography>Metadata: {selectedModelInfo?.metadata.join(', ')}</Typography>
          <Typography>Model Owner: {selectedModelInfo?.modelOwner}</Typography>
          <Typography>Model Visibility: {selectedModelInfo?.modelVisibility}</Typography>
          <Typography>Num Classes: {selectedModelInfo?.num_classes}</Typography>
          <Typography>Classes: {selectedModelInfo?.classes.join(', ')}</Typography>
          <Typography>Training Status: {selectedModelInfo?.status}</Typography>
        </Box>
      </Modal>

      <Modal open={openPlaygroundModal} onClose={handleClosePlaygroundModal}>
        <Box
          sx={{ 
            bgcolor: 'white', 
            border: '1px solid black', 
            borderRadius: '4px', 
            padding: 2, 
            position: 'relative', 
            width: '40%', // Set a width for the modal
            margin: 'auto', // Center the modal
            top: '50%', // Center vertically
            transform: 'translateY(-50%)' // Adjust for vertical centering
          }}
        >
          <Typography  sx={{textAlign: 'center'}} variant="h6">Model Playground</Typography>
          
          <Button 
              onClick={handleClosePlaygroundModal} 
              sx={{ 
                position: 'absolute', 
                top: 8, 
                right: 8 
              }}
            >
              <CloseIcon />
            </Button>
          <Box
            sx={{
              bgcolor: 'white', 
              border: '1px solid black', 
              borderRadius: '4px', 
              padding: 2, 
              position: 'relative', 
            }}
          >
            <Typography variant="h6">{selectedModelInfo?.modelName}</Typography>
            <Typography>Metadata: {selectedModelInfo?.metadata.join(', ')}</Typography>
            <>
            <div className='formRow'>
              <label htmlFor="modelSelect" className="labelHeading">
                Test Data: (Needs to be in 2D format, each row representing a sample.)
              </label>
              <input 
                type='text'
                id='testData'
                value={evalInputData}
                onChange={(e) => setEvalInputData(e.target.value)}
                placeholder='Test Data'
              />
            </div>
            </>
            <Typography className='modalButtonBox'>
            <Button sx={{
                            bgcolor: '#1976d2',
                            color: 'white',
                            px: 6,
                            mt: 2,
                            '&:hover': {
                              bgcolor: '#1565c0'
                            }
                          }}
                          
                          onClick={handlePredictButton}
                          >
                          Predict
            </Button>
            </Typography>
          </Box>
        </Box>
      </Modal>

      {models.length == 0 && (

        <Typography 
        variant="h5" 
        gutterBottom 
        sx={{ 
          fontWeight: 'light',
          textAlign: 'center',
          mb: 3
        }}
        >
        No Models Created.
        </Typography>
      )}
    </Box>
  );
};

const ModelVisibilityIcon = ({modelVisibility}) => {
  if (modelVisibility == 'Public'){
    return (<VisibilityIcon sx={{ color: '#008000' }}/>)
  }
  else{
    return (<VisibilityOffIcon sx={{ color: '#CC0000' }}/>)
  }
    
}

export default ModelRepository; 