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
import DatasetIcon from '@mui/icons-material/Dataset';
import axios from 'axios';
import InfoIcon from '@mui/icons-material/Info'; // Import InfoIcon
import CloseIcon from '@mui/icons-material/Close'; // Import CloseIcon
import DeleteIcon from '@mui/icons-material/Delete'; // Add this import at the top

const UploadedDatasets = ({userName, setSelectedDataset, update}) => {
  const [datasets, setDatasets] = useState([]);
  const [openModal, setOpenModal] = useState(false); // State for modal visibility
  const [selectedDatasetInfo, setSelectedDatasetInfo] = useState(null); // State for selected dataset info

  const handleOpenModal = (dataset) => {
    console.log('modal open')
    setSelectedDatasetInfo(dataset); // Set the selected dataset info
    setOpenModal(true); // Open the modal
  };
  const handleCloseModal = () => {
    setOpenModal(false); // Close the modal
  };
  const handleDelete = async (datasetId) => {
    // Show confirmation dialog
    if (!window.confirm('Are you sure you want to delete this dataset?')) {
      return;
    }
  
    try {
      
      const response = await axios.get(
        `/sandbox/delete_dataset`,
        {
          params: { datasetId },
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('tapis_token')}`
          }
        }
      );
      
      if (response.status === 200) {
        // Update the local state to remove the deleted dataset
        setDatasets(prevDatasets => 
          prevDatasets.filter(dataset => dataset._id !== datasetId)
        );
        // Optional: Show success message
        alert('Dataset deleted successfully');
      }
    } catch (error) {
      console.error('Error deleting dataset:', error);
      // Show error message to user
      alert('Failed to delete dataset. Please try again.');
    }
  };
  useEffect(() => {

    const fetchDatasets = async () => {
      try {

        axios.get(`/sandbox/get_user_datasets`, {
          headers: {
          'Authorization': `Bearer ${localStorage.getItem('tapis_token')}`
          }
        }).then(response => {
          console.log(response)
          const data = response.data;
          console.log(data); // Log the dataset list
          setDatasets(data); // Set the datasets state with the fetched data
        })
        .catch(error => {
          console.error('Error fetching datasets:', error);
        });
        
      } catch (error) {
        console.error('Error fetching datasets:', error);
      }
    };
    if(update){fetchDatasets()}
    fetchDatasets();
  }, []);

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
        Select Dataset
      </Typography>


      {datasets.length > 0 && (
        <List sx={{ width: '100%', bgcolor: 'background.paper' }}>
          {datasets.map((dataset, index) => (
            <React.Fragment key={dataset._id}>
              <ListItem 
                button
                alignItems="center"
                key={index} 
                onClick={() => setSelectedDataset(dataset._id)}
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
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: '#008000' }}>
                    <DatasetIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={dataset.dataset_name}
                  secondary={`Records: ${dataset.num_records}`}
                />
                 <Button 
                    onClick={(event) => {
                      event.stopPropagation(); // Prevent ListItem onClick
                      handleOpenModal(dataset);
                    }}
                  >
                    <InfoIcon />
                  </Button>
                  <Button 
                    onClick={(event) => {
                      event.stopPropagation();
                      // Add your delete handling logic here
                      console.log(dataset._id);
                      handleDelete(dataset._id);
                    }}
                    sx={{ color: 'error.main' }} // Makes the button red
                  >
                    <DeleteIcon />
                  </Button>
              </ListItem>
              {index < datasets.length - 1 && <Divider variant="inset" component="li" />}
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
          <Typography variant="h6">{selectedDatasetInfo?.dataset_name}</Typography>
          <Typography>Records: {selectedDatasetInfo?.num_records}</Typography>
          <Typography>Metadata: {selectedDatasetInfo?.metadata}</Typography>
        </Box>
      </Modal>
      {datasets.length == 0 && (

        <Typography 
        variant="h5" 
        gutterBottom 
        sx={{ 
          fontWeight: 'light',
          textAlign: 'center',
          mb: 3
        }}
        >
        No datasets uploaded.
        </Typography>
      )}
    </Box>
  );
};

export default UploadedDatasets; 