// import React, { useState } from 'react';
// import {
//   Container, TextField, Button, List, ListItem, ListItemText, IconButton, Typography, Box, Paper, Divider,
//   Snackbar, Alert, CircularProgress, Grid, Card, CardContent, Chip
// } from '@mui/material';
// import { Add, Delete, Upload, Info } from '@mui/icons-material';
// import SearchPage from '../Search/Search';
// import axios from 'axios';
// import URL from '../../config';
// //we need to make changes here
// const StartCollaboration = () => {
//   const [collabName, setCollabName] = useState('');
//   const [experimentName, setExperimentName] = useState('');
//   const [experimentList, setExperimentList] = useState([]);
//   const [phenoType, setPhenotype] = useState('');
//   const [samples, setSamples] = useState('');
//   const [rawData, setRawData] = useState(null);
//   const [selectedUsers, setSelectedUsers] = useState([]);
//   const [isLoading, setIsLoading] = useState(false);
//   const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
//   const [resetSearch, setResetSearch] = useState(false);

//   const handleAddExperiment = () => {
//     if (experimentName.trim()) {
//       setExperimentList([...experimentList, experimentName]);
//       setExperimentName('');
//     }
//   };

//   const handleDeleteExperiment = (index) => {
//     const updatedList = experimentList.filter((_, i) => i !== index);
//     setExperimentList(updatedList);
//   };

//   const handleFileUpload = (e, setData) => {
//     setData(e.target.files[0]);
//   };

//   const handleCreateCollaboration = async () => {
//     if (selectedUsers.length === 0) {
//       setSnackbar({ open: true, message: "Please select users for invitation.", severity: 'error' });
//       return;
//     }

//     setIsLoading(true);
//     const collaborationData = {
//       collabName,
//       experiments: experimentList,
//       phenoType,
//       samples,
//       rawData: rawData ? rawData.name : null,
//       invitedUsers: selectedUsers.map(user => ({
//         _id: user._id,
//         phenotype: user.phenotype
//       })),
//     };

//     try {
//       const createCollabResponse = await axios.post(`${URL}/api/start_collaboration`, collaborationData, {
//         headers: {
//           'Content-Type': 'application/json',
//           'Authorization': `Bearer ${localStorage.getItem('token')}`
//         }
//       });
//       const { collaboration_id } = createCollabResponse.data;

//       const invitationPromises = selectedUsers.map(user =>
//         axios.post(`${URL}/api/sendinvitation`, {
//           receiver_id: user._id,
//           phenotype: user.phenotype,
//           collaboration_id: collaboration_id
//         }, {
//           headers: {
//             'Content-Type': 'application/json',
//             'Authorization': `Bearer ${localStorage.getItem('token')}`
//           }
//         })
//       );

//       await Promise.all(invitationPromises);

//       setSnackbar({
//         open: true,
//         message: 'Collaboration created and invitations sent successfully!',
//         severity: 'success'
//       });

//       setCollabName('');
//       setExperimentList([]);
//       setPhenotype('');
//       setSamples('');
//       setRawData(null);
//       setSelectedUsers([]);
//       setResetSearch(prev => !prev);

//     } catch (error) {
//       console.error("Error creating collaboration or sending invitations:", error);
//       setSnackbar({
//         open: true,
//         message: 'Failed to create collaboration or send invitations. Please try again.',
//         severity: 'error'
//       });
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const handleCloseSnackbar = (event, reason) => {
//     if (reason === 'clickaway') {
//       return;
//     }
//     setSnackbar({ ...snackbar, open: false });
//   };

//   const handleUserSelect = (users) => {
//     setSelectedUsers(users);
//   };



//   return (
//     <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
//       <Typography variant="h4" align="left" gutterBottom sx={{ fontWeight: 'light' }}>
//         Start a New Collaboration
//       </Typography>
//       <Card sx={{ height: '100%', marginBottom: '20px', border: '1px solid #ccc', borderRadius: 2, boxShadow: 'none' }}>
//         <CardContent>
//           <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
//             <Box
//               sx={{
//                 width: 40,
//                 height: 40,
//                 borderRadius: '50%',
//                 bgcolor: 'primary.main',
//                 color: 'white',
//                 display: 'flex',
//                 justifyContent: 'center',
//                 alignItems: 'center',
//                 mr: 2,
//                 fontWeight: 'bold'
//               }}
//             >
//               1
//             </Box>
//             <Typography variant="h6" sx={{ flexGrow: 1 }}>
//               Collaboration Details
//             </Typography>
//             <Divider sx={{ flexGrow: 30, borderColor: 'black' }} />
//           </Box>

//           <TextField
//             label="Collaboration Name"
//             variant="outlined"
//             fullWidth
//             sx={{ mt: 2 }}
//             value={collabName}
//             onChange={(e) => setCollabName(e.target.value)}
//           />

//           <Box mt={2}>
//             <TextField
//               label="Experiment Name"
//               variant="outlined"
//               fullWidth
//               value={experimentName}
//               onChange={(e) => setExperimentName(e.target.value)}
//               sx={{ mb: 1 }}
//             />
//             <Button
//               variant="contained"
//               color="primary"
//               startIcon={<Add />}
//               onClick={handleAddExperiment}
//               sx={{ mb: 2 }}
//             >
//               Add Experiment
//             </Button>
//             <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
//               {experimentList.map((experiment, index) => (
//                 <Chip
//                   key={index}
//                   label={experiment}
//                   onDelete={() => handleDeleteExperiment(index)}
//                   color="primary"
//                   variant="outlined"
//                 />
//               ))}
//             </Box>
//           </Box>
//         </CardContent>
//       </Card>

//       <Grid item xs={12} md={6}>
//         <Card sx={{ height: '100%', marginBottom: '20px', border: '1px solid #ccc', borderRadius: 2, boxShadow: 'none' }}>
//           <CardContent>
//             <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
//               <Box
//                 sx={{
//                   width: 40,
//                   height: 40,
//                   borderRadius: '50%',
//                   bgcolor: 'primary.main',
//                   color: 'white',
//                   display: 'flex',
//                   justifyContent: 'center',
//                   alignItems: 'center',
//                   mr: 2,
//                   fontWeight: 'bold'
//                 }}
//               >
//                 2
//               </Box>
//               <Typography variant="h6" sx={{ flexGrow: 1 }}>
//                 Upload
//               </Typography>
//               <Divider sx={{ flexGrow: 30, borderColor: 'black' }} />
//             </Box>
//             <Box my={2}>
//               <TextField
//                 label="Phenotype"
//                 variant="outlined"
//                 fullWidth
//                 sx={{ mt: 2 }}
//                 value={phenoType}
//                 onChange={(e) => setPhenotype(e.target.value)}
//               />
//               <TextField
//                 label="Number of Samples"
//                 variant="outlined"
//                 type="number"
//                 fullWidth
//                 sx={{ mt: 2 }}
//                 value={samples}
//                 onChange={(e) => setSamples(e.target.value)}
//               />
//               <Button
//                 variant="outlined"
//                 component="label"
//                 fullWidth
//                 sx={{ mt: 2 }}
//               >
//                 Upload Dataset
//                 <input type="file" hidden onChange={(e) => handleFileUpload(e, setRawData)} />
//               </Button>
//               {rawData && <Typography variant="body2" sx={{ mt: 1 }}>Selected: {rawData.name}</Typography>}
//               <Box display="flex" alignItems="center" sx={{bgcolor:'#e3f2fd', color: '#1876D1' ,border: '1px solid #e3f2fd', borderRadius: 2, p: '5px', mt: 1}}>
//                 <Info sx={{fontSize:'18px'}}/>
//                 <Typography variant="body1" sx={{ marginLeft: 1, fontSize:'14px' }}>
//                   Upload the Noisy version of Dataset
//                 </Typography>
//               </Box>
//             </Box>
//           </CardContent>
//         </Card>

//         <Grid item xs={12}>
//           <Card sx={{ height: '100%', marginBottom: '20px', border: '1px solid #ccc', borderRadius: 2, boxShadow: 'none' }}>
//             <CardContent>
//               <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
//                 <Box
//                   sx={{
//                     width: 40,
//                     height: 40,
//                     borderRadius: '50%',
//                     bgcolor: 'primary.main',
//                     color: 'white',
//                     display: 'flex',
//                     justifyContent: 'center',
//                     alignItems: 'center',
//                     mr: 2,
//                     fontWeight: 'bold'
//                   }}
//                 >
//                   3
//                 </Box>
//                 <Typography variant="h6" sx={{ flexGrow: 1 }}>
//                   Find Collaborators
//                 </Typography>
//                 <Divider sx={{ flexGrow: 30, borderColor: 'black' }} />
//               </Box>
//               <SearchPage onUserSelect={handleUserSelect} resetTrigger={resetSearch} />
//               {/* {selectedUsers.length > 0 && (
//                 <Box mt={3}>
//                   <Grid container spacing={2}>
//                     {selectedUsers.map((user) => (
//                       <Grid item xs={12} sm={6} md={4} key={user._id}>
//                         <Card variant="outlined">
//                           <CardContent>
//                             <Typography variant="subtitle2">{user.name}</Typography>
//                             <Typography variant="body2" color="textSecondary">
//                               Phenotype: {user.phenotype}
//                             </Typography>
//                             <Typography variant="body2" color="textSecondary">
//                               Samples: {user.numberOfSamples}
//                             </Typography>
//                           </CardContent>
//                         </Card>
//                       </Grid>
//                     ))}
//                   </Grid>
//                 </Box>
//               )} */}
//             </CardContent>
//           </Card>
//         </Grid>
//       </Grid>

//       <Box mt={4}>
//         <Button
//           variant="contained"
//           color="primary"
//           fullWidth
//           size="large"
//           onClick={handleCreateCollaboration}
//           disabled={isLoading}
//           sx={{ py: 2, fontSize: '1.1rem' }}
//         >
//           {isLoading ? <CircularProgress size={24} color="inherit" /> : 'Create Collaboration'}
//         </Button>
//       </Box>

//       <Snackbar open={snackbar.open} autoHideDuration={6000} onClose={handleCloseSnackbar}>
//         <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
//           {snackbar.message}
//         </Alert>
//       </Snackbar>
//     </Container>
//   );
// };

// export default StartCollaboration;



import React, { useState, useEffect } from 'react';
import {
  Container, TextField, Button, List, ListItem, ListItemText, IconButton, Typography, Box, Paper, Divider,
  Snackbar, Alert, CircularProgress, Grid, Card, CardContent, Chip, FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import { Add, Delete, Upload, Info } from '@mui/icons-material';
import SearchPage from '../Search/Search';
import axios from 'axios';
import URL from '../../config';

//we need to make changes here
const StartCollaboration = () => {
  const [collabName, setCollabName] = useState('');
  const [experimentName, setExperimentName] = useState('');
  const [experimentList, setExperimentList] = useState([]);
  // const [phenoType, setPhenotype] = useState('');
  // const [samples, setSamples] = useState('');
  // const [rawData, setRawData] = useState(null);
  const [selectedUsers, setSelectedUsers] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [resetSearch, setResetSearch] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');

  // **Fetch Datasets on Component Mount**
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await axios.get(`${URL}/api/start_collaboration`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        console.log(response.data);
        setDatasets(response.data.datasets);
      } catch (error) {
        console.error("Error fetching datasets:", error);
        setSnackbar({
          open: true,
          message: 'Failed to fetch datasets. Please try again.',
          severity: 'error'
        });
      }
    };

    fetchDatasets();
  }, []);

  const handleAddExperiment = () => {
    if (experimentName.trim()) {
      setExperimentList([...experimentList, experimentName]);
      setExperimentName('');
    }
  };

  const handleDeleteExperiment = (index) => {
    const updatedList = experimentList.filter((_, i) => i !== index);
    setExperimentList(updatedList);
  };

  // const handleFileUpload = (e, setData) => {
  //   setData(e.target.files[0]);
  // };

  const handleCreateCollaboration = async () => {
    console.log("Selected Users:", selectedUsers); // Add this line for debugging

    if (selectedUsers.length === 0) {
      setSnackbar({ open: true, message: "Please select users for invitation.", severity: 'error' });
      return;
    }

    setIsLoading(true);
    const collaborationData = {
      collabName,
      experiments: experimentList,
      // phenoType,
      // samples,
      // rawData: rawData ? rawData.name : null,
      creatorDatasetId: selectedDataset.dataset_id,
      invitedUsers: selectedUsers.map(user => ({
        _id: user._id,
        dataset_id: user.dataset_id,
        phenotype: user.phenotype,
      })),
    };

    try {
      const createCollabResponse = await axios.post(`${URL}/api/start_collaboration`, collaborationData, {
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const { collaboration_id } = createCollabResponse.data;

      const invitationPromises = selectedUsers.map(user =>
        axios.post(`${URL}/api/sendinvitation`, {
          receiver_id: user._id,
          phenotype: user.phenotype,
          collaboration_id: collaboration_id
        }, {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        })
      );

      await Promise.all(invitationPromises);

      setSnackbar({
        open: true,
        message: 'Collaboration created and invitations sent successfully!',
        severity: 'success'
      });

      setCollabName('');
      setExperimentList([]);
      // setPhenotype('');
      // setSamples('');
      // setRawData(null);
      setSelectedUsers([]);
      setSelectedDataset(''); 
      setResetSearch(prev => !prev);

    } catch (error) {
      console.error("Error creating collaboration or sending invitations:", error);
      setSnackbar({
        open: true,
        message: 'Failed to create collaboration or send invitations. Please try again.',
        severity: 'error'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCloseSnackbar = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbar({ ...snackbar, open: false });
  };

  const handleUserSelect = (users) => {
    setSelectedUsers(users);
  };

  const handleDatasetChange = (event) => {
    const selectedDataset = event.target.value; 
    setSelectedDataset(selectedDataset);
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" align="left" gutterBottom sx={{ fontWeight: 'light' }}>
        Start a New Collaboration
      </Typography>
      <Card sx={{ height: '100%', marginBottom: '20px', border: '1px solid #ccc', borderRadius: 2, boxShadow: 'none' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Box
              sx={{
                width: 40,
                height: 40,
                borderRadius: '50%',
                bgcolor: 'primary.main',
                color: 'white',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                mr: 2,
                fontWeight: 'bold'
              }}
            >
              1
            </Box>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Collaboration Details
            </Typography>
            <Divider sx={{ flexGrow: 30, borderColor: 'black' }} />
          </Box>

          <TextField
            label="Collaboration Name"
            variant="outlined"
            fullWidth
            sx={{ mt: 2 }}
            value={collabName}
            onChange={(e) => setCollabName(e.target.value)}
          />

          <Box mt={2}>
            <TextField
              label="Experiment Name"
              variant="outlined"
              fullWidth
              value={experimentName}
              onChange={(e) => setExperimentName(e.target.value)}
              sx={{ mb: 1 }}
            />
            <Button
              variant="contained"
              color="primary"
              startIcon={<Add />}
              onClick={handleAddExperiment}
              sx={{ mb: 2 }}
            >
              Add Experiment
            </Button>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {experimentList.map((experiment, index) => (
                <Chip
                  key={index}
                  label={experiment}
                  onDelete={() => handleDeleteExperiment(index)}
                  color="primary"
                  variant="outlined"
                />
              ))}
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Grid item xs={12} md={6}>
        <Card sx={{ height: '100%', marginBottom: '20px', border: '1px solid #ccc', borderRadius: 2, boxShadow: 'none' }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Box
                sx={{
                  width: 40,
                  height: 40,
                  borderRadius: '50%',
                  bgcolor: 'primary.main',
                  color: 'white',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  mr: 2,
                  fontWeight: 'bold'
                }}
              >
                2
              </Box>
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Upload
              </Typography>
              <Divider sx={{ flexGrow: 30, borderColor: 'black' }} />
            </Box>
            <Box my={2}>
              {/* <TextField
                label="Phenotype"
                variant="outlined"
                fullWidth
                sx={{ mt: 2 }}
                value={phenoType}
                onChange={(e) => setPhenotype(e.target.value)}
              />
              <TextField
                label="Number of Samples"
                variant="outlined"
                type="number"
                fullWidth
                sx={{ mt: 2 }}
                value={samples}
                onChange={(e) => setSamples(e.target.value)}
              /> */}

              {/* **New Dropdown for Selecting Datasets** */}
              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel id="dataset-select-label">Select Dataset</InputLabel>
                <Select
                  labelId="dataset-select-label"
                  id="dataset-select"
                  value={selectedDataset}
                  label="Select Dataset"
                  onChange={handleDatasetChange}
                >
                  {datasets.length > 0 ? (
                    datasets.map((dataset, index) => (
                      <MenuItem key={index} value={dataset}>
                        {`${dataset.phenotype} | ${dataset.number_of_samples}`}
                      </MenuItem>
                    ))
                  ) : (
                    <MenuItem value="" disabled>
                      No Datasets Available
                    </MenuItem>
                  )}
                </Select>
              </FormControl>

              {/* <Button
                variant="outlined"
                component="label"
                fullWidth
                sx={{ mt: 2 }}
              >
                Upload Dataset
                <input type="file" hidden onChange={(e) => handleFileUpload(e, setRawData)} />
              </Button> */}
              {/* {rawData && <Typography variant="body2" sx={{ mt: 1 }}>Selected: {rawData.name}</Typography>}
              <Box display="flex" alignItems="center" sx={{ bgcolor: '#e3f2fd', color: '#1876D1', border: '1px solid #e3f2fd', borderRadius: 2, p: '5px', mt: 1 }}>
                <Info sx={{ fontSize: '18px' }} />
                <Typography variant="body1" sx={{ marginLeft: 1, fontSize: '14px' }}>
                  Upload the Noisy version of Dataset
                </Typography>
              </Box> */}
            </Box>
          </CardContent>
        </Card>

        <Grid item xs={12}>
          <Card sx={{ height: '100%', marginBottom: '20px', border: '1px solid #ccc', borderRadius: 2, boxShadow: 'none' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: '50%',
                    bgcolor: 'primary.main',
                    color: 'white',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    mr: 2,
                    fontWeight: 'bold'
                  }}
                >
                  3
                </Box>
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  Find Collaborators
                </Typography>
                <Divider sx={{ flexGrow: 30, borderColor: 'black' }} />
              </Box>
              <SearchPage onUserSelect={handleUserSelect} resetTrigger={resetSearch} />
              {/* {selectedUsers.length > 0 && (
                <Box mt={3}>
                  <Grid container spacing={2}>
                    {selectedUsers.map((user) => (
                      <Grid item xs={12} sm={6} md={4} key={user._id}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="subtitle2">{user.name}</Typography>
                            <Typography variant="body2" color="textSecondary">
                              Phenotype: {user.phenotype}
                            </Typography>
                            <Typography variant="body2" color="textSecondary">
                              Samples: {user.numberOfSamples}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )} */}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box mt={4}>
        <Button
          variant="contained"
          color="primary"
          fullWidth
          size="large"
          onClick={handleCreateCollaboration}
          disabled={isLoading}
          sx={{ py: 2, fontSize: '1.1rem' }}
        >
          {isLoading ? <CircularProgress size={24} color="inherit" /> : 'Create Collaboration'}
        </Button>
      </Box>

      <Snackbar open={snackbar.open} autoHideDuration={6000} onClose={handleCloseSnackbar}>
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default StartCollaboration;
