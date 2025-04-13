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
  Divider 
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import axios from 'axios';
import URL from '../../config';

const ModelRepository = ({userName, selectedDataset}) => {
  const [models, setModels] = useState([]);

  useEffect(() => {
    // Simply set the mock data
    setModels([
      { id: 1, name: "Farm Success Prediction", accuracy: "90%" },
      { id: 2, name: "Disease Risk Prediction", accuracy: "85%" },
      { id: 3, name: "Yield Prediction", accuracy: "82%" },
      { id: 4, name: "Crop Recommendation Model", accuracy: "78%" },
    ]);
  });

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
            <React.Fragment key={model.id}>
              <ListItem alignItems="center">
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: '#008000' }}>
                    <PersonIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={model.name}
                  secondary={`Accuracy: ${model.accuracy}`}
                />
              </ListItem>
              {index < models.length - 1 && <Divider variant="inset" component="li" />}
            </React.Fragment>
          ))}
        </List>
      )}
    </Box>
  );
};

export default ModelRepository; 