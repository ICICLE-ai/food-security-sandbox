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
import GppMaybeIcon from '@mui/icons-material/GppMaybe';
import HelpIcon from '@mui/icons-material/Help';
import './ModelRepository.css'

const ModelRepository = ({userName}) => {
  const [models, setModels] = useState([]);
  const [openModal, setOpenModal] = useState(false); // State for modal visibility
  const [selectedModelInfo, setSelectedModelInfo] = useState(null); // State for selected dataset info
  const [openPlaygroundModal, setOpenPlaygroundModal] = useState(false); // State for modal visibility
  const [openRiskAnalysisModal, setOpenRiskAnalysisModal] = useState(false); // State for modal visibility
  const [evalInputData, setEvalInputData] = useState('');
  const [evalPredictionOutput, setEvalPredictionOutput] = useState('');
  const [expandedCategories, setExpandedCategories] = useState({});

  const toggleCategory = (categoryName) => {
    setExpandedCategories(prevState => ({
      ...prevState,
      [categoryName]: !prevState[categoryName]
    }));
  };

  const handleOpenModal = (model) => {
    console.log('modal open')
    setSelectedModelInfo(model); // Set the selected dataset info
    setOpenModal(true); // Open the modal
  };
  const handleCloseModal = () => {
    setOpenModal(false); // Close the modal
    setSelectedModelInfo(null); // Set the selected model info
  };

  const handleOpenPlaygroundModal = (model) => {
    console.log('playground modal open')
    setSelectedModelInfo(model); // Set the selected model info
    setOpenPlaygroundModal(true); // Open the modal
  };

  const handleClosePlaygroundModal = () => {
    setOpenPlaygroundModal(false); // Close the modal
    setEvalPredictionOutput(''); // Clear the prediction output
    setEvalInputData(''); // Clear the input data
    setSelectedModelInfo(null); // Set the selected model info
  };

  const handleReadmeDownload = (readme) => {
    const blob = new Blob([readme], { type: 'text/plain;charset=utf-8' });

    // 4. Create a temporary URL for the Blob
    // URL.createObjectURL creates a DOMString containing a URL representing the object given in the parameter.
    const url = URL.createObjectURL(blob);

    // 5. Create a temporary <a> (anchor) element
    const a = document.createElement('a');
    a.style.display = 'none'; // Hide the link
    a.href = url; // Set the href to the Blob URL
    a.download = 'readme.txt'; // Set the download attribute to the desired file name

    // 6. Append the <a> element to the body (necessary for some browsers to trigger click)
    document.body.appendChild(a);

    // 7. Programmatically click the <a> element to trigger the download
    a.click();

    // 8. Clean up: Revoke the object URL and remove the <a> element
    // URL.revokeObjectURL releases the object URL, allowing the browser to free up memory.
    window.URL.revokeObjectURL(url); 
    document.body.removeChild(a);
  }

  const handleOpenRiskAnalysisModal = (model) => {
    console.log('risk analysis modal open')
    setSelectedModelInfo(model); // Set the selected model info
    setOpenRiskAnalysisModal(true); // Open the modal
  };

  const handleCloseRiskAnalysisModal = () => {
    setOpenRiskAnalysisModal(false); // Close the modal
    setSelectedModelInfo(null); // Set the selected model info
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
            setEvalPredictionOutput(response.data.predicted_classes_batch);
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

                  <ModelVisibilityIcon 
                    modelVisibility={model.modelVisibility} 
                    sx={{ 
                      maxWidth: '30px', maxHeight: '30px', minWidth: '30px', minHeight: '30px',
                      padding: '8px', 
                      marginRight: '5px' 
                      }}/>
                  
                  <Button 
                      onClick={(event) => {
                        event.stopPropagation(); // Prevent ListItem onClick
                        handleOpenModal(model);
                      }}
                      sx={{ 
                        maxWidth: '30px', maxHeight: '30px', minWidth: '30px', minHeight: '30px',
                        padding: '8px', 
                        margin: '0 2px' 
                      }}
                    >
                      <InfoIcon />
                  </Button>
                  <Button 
                    onClick={(event) => {
                      event.stopPropagation(); // Prevent ListItem onClick
                      handleOpenRiskAnalysisModal(model);
                    }}
                    sx={{ 
                      maxWidth: '30px', maxHeight: '30px', minWidth: '30px', minHeight: '30px',
                      padding: '8px', 
                      margin: '0 2px' 
                    }}
                  >
                      <GppMaybeIcon sx={{ color: 'red' }} />
                  </Button>
                  <Button 
                      onClick={(event) => {
                        event.stopPropagation(); // Prevent ListItem onClick
                        handleOpenPlaygroundModal(model);
                      }}
                      sx={{ 
                        maxWidth: '30px', maxHeight: '30px', minWidth: '30px', minHeight: '30px',
                        padding: '8px', 
                        margin: '0 2px' 
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
  {/* Information Modal */}
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

  {/* Playground Modal */}
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
          
          <Box
            sx={{ 
              position: 'absolute', 
              top: 8, 
              right: 8,
              minWidth: 'auto',
              display: 'flex',
              flexDirection: 'row',
              ":hover": {
              bgcolor: "transparent"
            }
            }}
          >
            { 
              (selectedModelInfo?.readme != "" && selectedModelInfo?.modelReadme != undefined)?            
                <Button 
                  onClick={() => handleReadmeDownload(selectedModelInfo?.modelReadme)} 
                  sx={{ 
                    maxWidth: '30px', maxHeight: '30px', minWidth: '30px', minHeight: '30px',
                  //   "&.MuiButtonBase-root:hover": {
                  //   bgcolor: "transparent"
                  // }
                  }}
                >
                  <HelpIcon />
                </Button>
                :null
              }
            <Button 
                onClick={handleClosePlaygroundModal} 
                sx={{ 
                  maxWidth: '30px', maxHeight: '30px', minWidth: '30px', minHeight: '30px',
                }}
              >
                <CloseIcon />
              </Button>
              
            </Box>
          <Box
            sx={{
              bgcolor: 'white', 
              border: '1px solid black', 
              borderRadius: '4px', 
              padding: 2, 
              position: 'relative', 
            }}
          >
          {evalPredictionOutput == '' ? (
           <>
            <Typography variant="h6">{selectedModelInfo?.modelName}</Typography>
            <Typography>Metadata: {selectedModelInfo?.metadata.join(', ')}</Typography>
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
          </>) : 
          (
            <>
              <Typography variant="h6">Prediction Result:</Typography>
              <Typography>{evalPredictionOutput?.join(', ')}</Typography>
            </>
          )}
          </Box>
        </Box>
      </Modal>

  {/* Risk Analysis Modal */}
      <Modal open={openRiskAnalysisModal} onClose={handleCloseRiskAnalysisModal}>
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
            onClick={handleCloseRiskAnalysisModal} 
            sx={{ 
              position: 'absolute', 
              top: 8, 
              right: 8 
            }}
          >
            <CloseIcon />
          </Button>
          <Typography variant="h6">Risk Analysis Information</Typography>
          <div>
            {/* Model Logs Section */}
            <h5 onClick={() => toggleCategory('model_logs')} style={{ cursor: 'pointer' }}>
              Model Logs {expandedCategories.model_logs ? '▼' : '▶'}
            </h5>
            {expandedCategories.model_logs && (
              <div id= 'modelInfoBox'>
                {selectedModelInfo?selectedModelInfo.model_logs?Object.entries(selectedModelInfo?.model_logs).map(([key, value]) => (
                  <p key={key} id = 'modelInfoItem'>
                    {key}:{value}
                  </p>
                )):<p>No Information Available</p>:<p>No Information Available</p>}
              </div>
            )}

            {/* Model Activity Section */}
            <h5 onClick={() => toggleCategory('model_activity')} style={{ cursor: 'pointer' }}>
              Model Activity {expandedCategories.model_activity ? '▼' : '▶'}
            </h5>
            {expandedCategories?.model_activity && (
              <div id= 'modelInfoBox'>
                {selectedModelInfo?selectedModelInfo.model_activity?Object.entries(selectedModelInfo?.model_activity).map(([key, valueArray]) => {
                  console.log(valueArray[2]['$date'])
                  return(
                    <p key={key} id = 'modelInfoItem'>
                      {key}: user_id, {valueArray[0]}, qureies: {valueArray[1]}, timestamp: {new Date(valueArray[2]['$date']).toLocaleString()}
                    </p>
                )}):<p>No Information Available</p>:<p>No Information Available</p>}
              </div>
            )}

            {/* Model Risk Section */}
            <h5 onClick={() => toggleCategory('model_risk')} style={{ cursor: 'pointer' }}>
              Risk Analysis {expandedCategories.model_risk ? '▼' : '▶'}
            </h5>
            {expandedCategories?.model_risk && (
              <div id= 'modelInfoBox'>
                {selectedModelInfo?selectedModelInfo.mia_attack_acc?
                  <p>membership inference attack accuracy: {selectedModelInfo.mia_attack_acc}</p>:<p>No Information Available</p>:<p>No Information Available</p>}
              </div>
            )}
          </div>
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