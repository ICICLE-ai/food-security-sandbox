import React, { useEffect, useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box, Typography, alpha, Divider, Button, Slider, TextField, Chip, Grid, Snackbar, Alert, CircularProgress, Container, Card, CardContent, List, ListItem, ListItemText, IconButton, Tooltip, TableContainer, Table, TableBody, TableCell, TableHead, TableRow, LinearProgress
} from '@mui/material';
import { Add, Edit as EditIcon, Save as SaveIcon, Cancel as CancelIcon } from '@mui/icons-material';
import axios from 'axios';
import URL from '../../config';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';


const CollaborationDetails = () => {
  const { uuid } = useParams();
  const [collaboration, setCollaboration] = useState(null);
  const [collabName, setCollabName] = useState('');
  const [experimentName, setExperimentName] = useState('');
  const [experimentList, setExperimentList] = useState([]);
  const [phenoType, setPhenotype] = useState('');
  const [samples, setSamples] = useState('');
  const [statData, setStatData] = useState(null);
  const [role, setRole] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [senderInfo, setSenderInfo] = useState({ id: null, name: '' });
  const [invitedUsers, setInvitedUsers] = useState([]);
  const [collaborationUuid, setCollaborationUuid] = useState('');
  const [threshold, setThreshold] = useState(null);
  const [thresholdDefined, setThresholdDefined] = useState();
  const [activeStep, setActiveStep] = useState(0);
  const [isQcInitiateLoading, setIsQcInitiateLoading] = useState(false);
  const [qcResultsAvailable, setQcResultsAvailable] = useState(false);
  const [qcResults, setQcResults] = useState(null);
  const [isQcResultsLoading, setIsQcResultsLoading] = useState(false);
  const [displayQcResults, setdisplayQcResults] = useState(false);
  const [qcInitiated, setQcInitiated] = useState(false); // New state to track QC initiation
  const [filteredResults, setFilteredResults] = useState([]);
  const prevFilteredResultsRef = useRef();
  // const [matrix, setMatrix] = useState([]);
  const [isEditing, setIsEditing] = useState(false);
  const [originalCollabName, setOriginalCollabName] = useState('');
  const [originalExperimentList, setOriginalExperimentList] = useState([]);
  const [originalPhenoType, setOriginalPhenoType] = useState('');
  const [originalSamples, setOriginalSamples] = useState('');

  const determineUserRole = (data) => {
    if (data.sender_id) {
      setRole('sender');
    } else {
      setRole('receiver');
    }
  };
  const steps = ['QC Calculation', 'Stat Data Upload'];

  useEffect(() => {
    const fetchCollaborationDetails = async () => {
      try {
        const response = await axios.get(`${URL}/api/collaboration/${uuid}`, {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`,
          },
        });
        // console.log(response.data);
        setCollaboration(response.data);
        setCollabName(response.data.name);
        setExperimentList(response.data.experiments || []);
        setPhenotype(response.data.datasets.phenotype || []);
        setSamples(response.data.datasets.samples || '');
        setCollaborationUuid(response.data.uuid);
        setSenderInfo({ id: response.data.sender_id, name: response.data.sender_name });
        setInvitedUsers(response.data.invited_users || []);
        determineUserRole(response.data);
      } catch (error) {
        console.error('Error fetching collaboration details:', error);
        setSnackbar({
          open: true,
          message: 'Failed to fetch collaboration details. Please try again.',
          severity: 'error',
        });
      }
    };

    fetchCollaborationDetails();
  }, [uuid]);

  const handleAddExperiment = () => {
    if (experimentName.trim()) {
      setExperimentList([...experimentList, experimentName.trim()]);
      setExperimentName('');
    } else {
      setSnackbar({ open: true, message: 'Experiment name cannot be empty.', severity: 'error' });
    }
  };

  const handleDeleteExperiment = (index) => {
    const updatedList = experimentList.filter((_, i) => i !== index);
    setExperimentList(updatedList);
  };
  // Stat upload handlers
  const handleFileUpload = (e) => {
    setStatData(e.target.files[0]);
  };

  const handleSubmitStat = async () => {
    if (!statData) {
      console.error('No file selected!');
      return;
    }

    const collabStatData = new CollabStatData();
    collabStatData.append('file', statData);

    try {
      //waiting for the endpoint to be created
      const response = await axios.post(`${URL}/api/upload_csv_stats`, collabStatData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('File uploaded successfully:', response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };
  const handleUpdateCollaboration = async () => {
    setIsLoading(true);
    try {
      const updateData = {
        name: collabName,
        experiments: experimentList,
        phenotype: phenoType,
        samples: samples,
      };
      if (role === 'receiver') {
        updateData.stat_data = statData ? statData.name : null;
      }
      await axios.put(`${URL}/api/collaboration/${uuid}`, updateData, {
        headers: { Authorization: `Bearer ${localStorage.getItem('token')}` },
      });
      setSnackbar({
        open: true,
        message: 'Collaboration updated successfully!',
        severity: 'success',
      });
      setIsEditing(false);
    } catch (error) {
      console.error('Error updating collaboration:', error);
      setSnackbar({
        open: true,
        message: 'Failed to update collaboration. Please try again.',
        severity: 'error',
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

  const handleAccept = async (userId) => {
    try {
      const response = await axios.post(
        `${URL}/api/acceptinvitation`,
        {
          uuid: collaborationUuid,
          receiver_id: userId,
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`,
          },
        }
      );

      if (response.status === 200) {
        setInvitedUsers((prevUsers) =>
          prevUsers.map((user) =>
            user.user_id === userId ? { ...user, status: 'accepted' } : user
          )
        );
        setSnackbar({ open: true, message: 'Invitation accepted successfully!', severity: 'success' });
      }
    } catch (error) {
      console.error('Error accepting invitation:', error);
      setSnackbar({ open: true, message: 'Failed to accept invitation. Please try again.', severity: 'error' });
    }
  };

  const handleReject = async (userId) => {
    try {
      const response = await axios.post(
        `${URL}/api/rejectinvitation`,
        {
          uuid: collaborationUuid,
          receiver_id: userId,
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`,
          },
        }
      );

      if (response.status === 200) {
        setInvitedUsers((prevUsers) => prevUsers.filter((user) => user.user_id !== userId));
        setSnackbar({ open: true, message: 'Invitation rejected successfully!', severity: 'success' });
      }
    } catch (error) {
      console.error('Error rejecting invitation:', error);
      setSnackbar({ open: true, message: 'Failed to reject invitation. Please try again.', severity: 'error' });
    }
  };

  const handleWithdraw = async (userId) => {
    try {
      const response = await axios.post(
        `${URL}/api/withdrawinvitation`,
        {
          uuid: collaborationUuid,
          receiver_id: userId,
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`,
          },
        }
      );

      if (response.status === 200) {
        setInvitedUsers((prevUsers) => prevUsers.filter((user) => user.user_id !== userId));
        setSnackbar({ open: true, message: 'Invitation withdrawn successfully!', severity: 'success' });
      }
    } catch (error) {
      console.error('Error withdrawing invitation:', error);
      setSnackbar({ open: true, message: 'Failed to withdraw invitation. Please try again.', severity: 'error' });
    }
  };


  // Send the threshold value to the backend
  const handleSubmitThreshold = async () => {
    try {
      const response = await axios.post(`${URL}/api/datasets/${uuid}/qc-results`, {
        threshold: threshold,
      },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`,
          },
        }
      );

      console.log('Filtered QC Results:', response.data);

      setSnackbar({
        open: true,
        message: 'Threshold applied and QC results filtered successfully!',
        severity: 'success',
      });
    } catch (error) {
      console.error('Error submitting threshold and filtering:', error);

      setSnackbar({
        open: true,
        message: 'Failed to apply threshold. Please try again.',
        severity: 'error',
      });
    }
  };


  const checkQcStatus = async () => {
    try {
      const response = await axios.get(`${URL}/api/datasets/${uuid}/qc-results`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
      });
      // console.log("results", response);
      if (response.status === 200) {
        setQcResultsAvailable(true);
        setdisplayQcResults(true);
        setQcResults(response.data.full_qc_results); // Store QC results
        // below condition ensures if the threshold already defined by user earlier, it shall be used when user interacts with the UI again.
        if (response.data.threshold !== null) {
          setThreshold(response.data.threshold);
          setThresholdDefined(true);
        } else {
          setThresholdDefined(false)
        }
        return true;
      } else {
        // console.error(`Unexpected response status: ${response.status}`);
        setQcResultsAvailable(false);
        return false;
      }
    } catch (error) {
      // console.error('Error checking QC status:', error);
      setQcResultsAvailable(false); // Default to unavailable in case of error
      return false;
    } finally {
      setIsQcResultsLoading(false);
      setdisplayQcResults(true);
    }
  };


  const handleQcInitiate = async () => {
    setIsQcInitiateLoading(true);
    try {
      const resultsAvailable = await checkQcStatus();

      if (resultsAvailable) {
        setSnackbar({
          open: true,
          message: 'QC calculations are already completed.',
          severity: 'info',
        });
        return; // Exit early if results are available
      }

      // Proceed with QC initiation if no results are available
      const response = await axios.post(
        `${URL}/api/datasets/${uuid}`,
        {},
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('token')}`,
          },
        }
      );

      setSnackbar({
        open: true,
        message: 'QC Calculations Initiated.',
        severity: 'success',
      });
      setQcInitiated(true);
    } catch (error) {
      console.error('Error initiating QC calculations:', error);
      setSnackbar({
        open: true,
        message: 'Failed to initiate QC calculations. Please try again.',
        severity: 'error',
      });
    } finally {
      setIsQcInitiateLoading(false);
      setQcResultsAvailable(true);
    }
  };
// handleQcResults function might not be necessary since we are already checking the results in checkQcStatus function --- verify later
  const handleQcResults = async () => {
    if (qcResults) {
      setSnackbar({ open: true, message: 'Results are already available.', severity: 'info' });
      return;
    }

    setIsQcResultsLoading(true);
    try {
      const response = await axios.get(`${URL}/api/datasets/${uuid}/qc-results`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('token')}`,
        },
      });

      console.log(response.data);

      if (response.status === 200) {
        setSnackbar({ open: true, message: 'Results are available.', severity: 'success' });
        setQcResultsAvailable(true);
        setQcResults(response.data.full_qc_results); // Store QC results
        // below condition ensures if the threshold already defined by user earlier, it shall be used when user interacts with the UI again.
        if (response.data.threshold !== null) {
          setThreshold(response.data.threshold);
          setThresholdDefined(true);
        } else {
          setThresholdDefined(false)
        }
      } else {
        setSnackbar({ open: true, message: 'Results are not available.', severity: 'info' });
      }
    } catch (error) {
      console.error('Error fetching QC results:', error);
      setSnackbar({ open: true, message: 'Error fetching results.', severity: 'error' });
    } finally {
      setIsQcResultsLoading(false);
      setdisplayQcResults(true);
    }
  };
  const getMatrix = () => {
    if (!qcResults) return [];
    return qcResults.map((item) => [
      item.phi_value,
      item.sample1,
      item.sample2
    ]);
  };
  const matrix = getMatrix();

  const handleSliderChange = (event, newValue) => {
    setThresholdDefined(false);
    setThreshold(newValue);
  };

  const filterData = (threshold) => {
    const filteredData = matrix.filter(row => row.phi_value >= threshold);
    setFilteredResults(filteredData);
  };

  // Re-filter the data when threshold changes
  useEffect(() => {
    const filteredResults = matrix.filter(row => row[0] < threshold);
    // console.log('Filtered Results:', filteredResults.length);
    // Only set state if filtered results have changed (compare with previous state)
    if (JSON.stringify(filteredResults) !== JSON.stringify(prevFilteredResultsRef.current)) {
      setFilteredResults(filteredResults);
      prevFilteredResultsRef.current = filteredResults; // Update the ref with the current filtered results
    }
  }, [threshold, matrix]);


  const isQcInitiateEnabled =
    role === 'sender' && invitedUsers.length > 0 && invitedUsers.every(user => user.status === 'accepted' && !qcResultsAvailable);

  const isQcResultsEnabled = !isQcInitiateEnabled && qcResultsAvailable;

  useEffect(() => {
    if (isQcResultsEnabled) {
      setActiveStep(1); // Automatically move to next step when QC calculation is ready
    }
  }, [isQcResultsEnabled]);


  useEffect(() => {
    checkQcStatus(); // Update QC results availability on component load
  }, []);

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" align="left" gutterBottom sx={{ fontWeight: 'light' }}>
        Collaboration Details
      </Typography>

      <Grid container spacing={4}>
        {/* Left Column */}
        <Grid item xs={12} md={8}>
          {/* Collaboration Details Card */}
          <Card sx={{ height: '100%', marginBottom: '20px', border: '1px solid #ccc', borderRadius: 2, boxShadow: 'none' }}>
            <CardContent>
              {/* Header */}
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  <strong>About Collaboration</strong>
                </Typography>
                <Divider sx={{ flexGrow: 30, borderColor: 'black' }} />
                {role === 'sender' && !isEditing && (
                  <Tooltip title="Edit Collaboration Details">
                    <IconButton
                      onClick={() => {
                        // Store original values before editing
                        setOriginalCollabName(collabName);
                        setOriginalExperimentList([...experimentList]);
                        setOriginalPhenoType(phenoType);
                        setOriginalSamples(samples);
                        setIsEditing(true);
                      }}
                    >
                      <EditIcon />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>

              {/* Collaboration Name */}
              {role === 'sender' ? (
                isEditing ? (
                  <TextField
                    label="Collaboration Name"
                    variant="outlined"
                    fullWidth
                    sx={{ mt: 2 }}
                    value={collabName}
                    onChange={(e) => setCollabName(e.target.value)}
                  />
                ) : (
                  <Typography variant="body1" sx={{ mt: 1 }}>
                    <strong>Title:</strong> {collabName}
                  </Typography>
                )
              ) : (
                <Typography variant="body1" sx={{ mt: 1 }}>
                  <strong>Title:</strong> {collabName}
                </Typography>
              )}

              {/* List of Details */}
              <List>
                {/* Experiments */}
                <ListItem disableGutters>
                  <ListItemText
                    primary={<strong>Experiments</strong>}
                    secondary={
                      role === 'sender' ? (
                        isEditing ? (
                          <>
                            <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                              <TextField
                                variant="outlined"
                                value={experimentName}
                                onChange={(e) => setExperimentName(e.target.value)}
                                size="small"
                                sx={{ mr: 2, flexGrow: 1 }}
                              />
                              <Tooltip title="Add a new experiment">
                                <Button variant="contained" color="primary" startIcon={<Add />} onClick={handleAddExperiment}>
                                  Add
                                </Button>
                              </Tooltip>
                            </Box>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
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
                          </>
                        ) : (
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                            {experimentList.map((experiment, index) => (
                              <Chip key={index} label={experiment} color="primary" variant="outlined" />
                            ))}
                          </Box>
                        )
                      ) : (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                          {experimentList.map((experiment, index) => (
                            <Chip key={index} label={experiment} color="primary" variant="outlined" />
                          ))}
                        </Box>
                      )
                    }
                  />
                </ListItem>
                <Divider />

                {/* Phenotype */}
                <ListItem disableGutters>
                  <ListItemText
                    primary={<strong>Phenotype</strong>}
                    secondary={
                      role === 'sender' ? (
                        isEditing ? (
                          <TextField
                            variant="outlined"
                            fullWidth
                            value={phenoType}
                            onChange={(e) => setPhenotype(e.target.value)}
                          />
                        ) : (
                          <Typography variant="body2">{phenoType}</Typography>
                        )
                      ) : (
                        <Typography variant="body2">{phenoType}</Typography>
                      )
                    }
                  />
                </ListItem>
                <Divider />

                <ListItem disableGutters>
                  <ListItemText
                    primary={<strong>Number of Samples</strong>}
                    secondary={
                      role === 'sender' ? (
                        isEditing ? (
                          <TextField
                            variant="outlined"
                            fullWidth
                            value={samples}
                            onChange={(e) => setSamples(e.target.value)}
                          />
                        ) : (
                          <Typography variant="body2">{samples}</Typography>
                        )
                      ) : (
                        <Typography variant="body2">{samples}</Typography>
                      )
                    }
                  />
                </ListItem>
                <Divider />

                <ListItem disableGutters sx={{ mt: 2, display: 'block' }}>
                  {role !== 'receiver' && (
                    <ListItemText
                      sx={{ width: '100%', display: 'block' }}
                      primary={<strong>Quality Control</strong>}
                      secondary={
                        <Box
                          sx={{
                            display: 'flex',
                            gap: 2,
                            mt: 1,
                            width: '100%', // Ensure the Box takes full width of its parent
                          }}
                        >
                          {/* QC Initiate Button */}
                          <Box sx={{ flex: 1 }}>
                            <Tooltip
                              title={
                                !isQcInitiateEnabled
                                  ? 'QC calculations already initiated or waiting for Collaborators'
                                  : role === 'receiver'
                                    ? 'Receiver cannot initiate QC calculations'
                                    : 'Initiate QC Calculations'
                              }
                              placement="bottom"
                            >
                              <span>
                                <Button
                                  variant="contained"
                                  color="primary"
                                  onClick={handleQcInitiate}
                                  disabled={!isQcInitiateEnabled || isQcInitiateLoading}
                                  startIcon={
                                    isQcInitiateLoading && (
                                      <CircularProgress size={20} color="inherit" />
                                    )
                                  }
                                  fullWidth
                                  sx={{ borderRadius: 20 }}
                                >
                                  Initiate QC Calculation
                                </Button>
                              </span>
                            </Tooltip>
                          </Box>

                          {/* QC Results Button */}
                          {/* <Box sx={{ flex: 1 }}>
                            <Tooltip
                              title={
                                !isQcResultsEnabled
                                  ? 'Results not available to preview'
                                  : role === 'receiver'
                                    ? 'Receiver cannot initiate QC results'
                                    : 'Get QC Results'
                              }
                              placement="bottom"
                            >
                              <span>
                                <Button
                                  variant="outlined"
                                  color="secondary"
                                  onClick={handleQcResults}
                                  disabled={!isQcResultsEnabled || isQcResultsLoading || role === 'receiver'}
                                  startIcon={
                                    isQcResultsLoading && (
                                      <CircularProgress size={20} color="inherit" />
                                    )
                                  }
                                  fullWidth
                                  sx={{ borderRadius: 20 }}
                                >
                                  QC Results
                                </Button>
                              </span>
                            </Tooltip>
                          </Box> */}
                        </Box>
                      }
                    />
                  )}
                  {role === 'receiver' && (
                    <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', mt: 2 }}>
                      <Box
                        sx={{
                          display: 'flex',
                          width: '100%',
                          backgroundColor: '#E3F2E4', // Light green background for incomplete steps
                          borderRadius: 50,
                          height: 40,
                          overflow: 'hidden',
                          position: 'relative',
                        }}
                      >
                        {/* Left section: Initiator Pending */}
                        <Tooltip
                          title={activeStep === 0 ? 'Initiator yet to do perform Quality Control Calculation' : 'Quality Control results avilable, contact Initiator for more information'}
                          placement="top"
                        >

                          <Box
                            sx={{
                              width: '50%',
                              backgroundColor:
                                activeStep === 0
                                  ? 'success.main' // Active step color
                                  : activeStep > 0
                                    ? '#E3F2E4' // Completed step background
                                    : 'grey.300', // Incomplete step background (grey)
                              display: 'flex',
                              justifyContent: 'center',
                              alignItems: 'center',
                              color: activeStep === 0 ? 'white' : activeStep > 0 ? 'success.main' : 'text.secondary',
                              borderRadius: '50px 0px 0px 50px',
                              transition: 'background-color 0.3s ease',
                            }}
                          >
                            {/* If step is completed, show checkmark icon */}
                            {activeStep > 0 ? (
                              <CheckCircleIcon sx={{ color: 'success.main', marginRight: 1, fontSize: 15 }} />
                            ) : null}
                            <Typography variant="body2">{steps[0]}</Typography>
                          </Box>
                        </Tooltip>
                        {/* Right section: Stat Data Upload */}
                        <Tooltip
                          title={activeStep === 1 ? 'You can now upload the Stat for GWAS Experiment' : 'Waiting for QC results'}
                          placement="top"
                        >

                          <Box
                            sx={{
                              width: '50%',
                              backgroundColor:
                                activeStep === 1
                                  ? 'success.main' // Active step color
                                  : activeStep > 1
                                    ? '#E3F2E4' // Completed step background
                                    : 'grey.300', // Incomplete step background (grey)
                              display: 'flex',
                              justifyContent: 'center',
                              alignItems: 'center',
                              color: activeStep === 1 || activeStep > 1 ? 'white' : 'text.secondary',
                              borderRadius: '0px 50px 50px 0px',
                              transition: 'background-color 0.3s ease',
                            }}
                          >
                            {/* If step is completed, show checkmark icon */}
                            {activeStep > 1 ? (
                              <CheckCircleIcon sx={{ color: 'success.main', marginRight: 1 }} />
                            ) : null}
                            <Typography variant="body2">{steps[1]}</Typography>
                          </Box>
                        </Tooltip>
                      </Box>
                    </Box>

                  )}
                </ListItem>
                {(displayQcResults && role === 'sender')&& (
                  <>
                    <ListItem disableGutters sx={{ mt: 2 }}>
                      <ListItemText
                        primary={<strong>Select Threshold Value</strong>}
                        secondary={
                          <Box sx={{ mt: 1 }}>
                            <Slider
                              value={threshold}
                              onChange={handleSliderChange}
                              aria-labelledby="threshold-slider"
                              valueLabelDisplay="auto"
                              step={0.01}
                              min={0}
                              max={1}
                              sx={{
                                color: "primary",
                                height: 8,
                                "& .MuiSlider-track": { border: "none" },
                                "& .MuiSlider-thumb": {
                                  height: 24,
                                  width: 24,
                                  backgroundColor: "#fff",
                                  border: "2px solid currentColor",
                                  "&:focus, &:hover, &.Mui-active, &.Mui-focusVisible": { boxShadow: "inherit" },
                                  "&::before": { display: "none" },
                                },
                                "& .MuiSlider-valueLabel": {
                                  lineHeight: 1.2,
                                  fontSize: 12,
                                  background: "unset",
                                  padding: 0,
                                  width: 32,
                                  height: 32,
                                  borderRadius: "50% 50% 50% 0",
                                  backgroundColor: "#1876D1",
                                  transformOrigin: "bottom left",
                                  transform: "translate(50%, -100%) rotate(-45deg) scale(0)",
                                  "&::before": { display: "none" },
                                  "&.MuiSlider-valueLabelOpen": { transform: "translate(50%, -100%) rotate(-45deg) scale(1)" },
                                  "& > *": { transform: "rotate(45deg)" },
                                },
                              }}
                            />

                            <Typography variant="body2" gutterBottom>
                              Current Threshold: {threshold}
                            </Typography>
                            <Tooltip title={thresholdDefined ? 'Threshold previously defined' : 'Confirm your selected threshold value'} placement="bottom">
                              <Button variant="contained" color="primary" onClick={handleSubmitThreshold} sx={{ borderRadius: 10 }} disabled={thresholdDefined}>
                                {thresholdDefined ? 'Threshold previously defined' : 'Confirm Threshold Value'}
                              </Button>
                            </Tooltip>
                          </Box>
                        }
                      />
                    </ListItem>
                    <Divider />

                    {/* QC Results Section */}
                    <ListItem disableGutters sx={{ mt: 2 }}>
                      <ListItemText
                        primary={<strong>QC Results</strong>}
                        secondary={
                          qcResults ? (
                            <Box sx={{ mt: 4, maxHeight: 400, border: '1px solid #ccc', borderRadius: 2 }}>
                              <TableContainer sx={{ maxHeight: 400, overflow: "auto", borderRadius: 2 }}>
                                <Table stickyHeader>
                                  <TableHead>
                                    <TableRow>
                                      <TableCell sx={{ backgroundColor: "#3B6CC7", color: "white", fontWeight: "bold", position: "sticky", top: 0, zIndex: 1 }}>
                                        Phi Value
                                      </TableCell>
                                      <TableCell sx={{ backgroundColor: "#3B6CC7", color: "white", fontWeight: "bold", position: "sticky", top: 0, zIndex: 1 }}>
                                        Sample 1
                                      </TableCell>
                                      <TableCell sx={{ backgroundColor: "#3B6CC7", color: "white", fontWeight: "bold", position: "sticky", top: 0, zIndex: 1 }}>
                                        Sample 2
                                      </TableCell>
                                    </TableRow>
                                  </TableHead>
                                  <TableBody>
                                    {filteredResults.map((row, rowIndex) => (
                                      <TableRow key={rowIndex}>
                                        {row.map((cell, cellIndex) => (
                                          <TableCell key={cellIndex}>{cell}</TableCell>
                                        ))}
                                      </TableRow>
                                    ))}
                                  </TableBody>

                                </Table>
                              </TableContainer>

                            </Box>

                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              No QC results to display.
                            </Typography>
                          )
                        }
                      />
                    </ListItem>
                    <Divider />
                  </>
                )}

                {/* Upload Dataset */}
                <>
                  <ListItem disableGutters>
                    {isQcResultsEnabled && (
                      <ListItemText
                        primary={<strong>Upload Stat</strong>}
                        secondary={
                          <>
                            {/* File Upload Button */}
                            <Tooltip placement='top' title={statData ? "File already selected" : "Upload file must be in CSV or JSON formatted."}>
                              <span>
                                <Button variant="outlined" component="label" fullWidth sx={{ mt: 1, borderRadius: 10, }}>
                                  Select Stat File
                                  <input type="file" hidden onChange={handleFileUpload} />
                                </Button>
                                {statData && (
                                  <Typography variant="body2" sx={{ mt: 1 }}>
                                    Selected: {statData.name}
                                  </Typography>
                                )}
                              </span>
                            </Tooltip>
                            {/* Submit Button */}
                            {statData && (
                              <Button variant="contained" color="primary" fullWidth sx={{ mt: 2, borderRadius: 10 }} onClick={handleSubmitStat} >
                                Submit Stat Data
                              </Button>
                            )}
                          </>
                        }
                      />
                    )}
                  </ListItem>

                  <Divider />
                </>
                {/*GWAS Calculation Trigger */}
                <>
                  {role === 'sender' && (
                    <ListItem disableGutters sx={{ mt: 2 }}>

                      <ListItemText
                        primary={<strong>GWAS Calculation</strong>}
                        secondary={
                          <Box
                            sx={{
                              display: 'flex',
                              gap: 2,
                              mt: 1,
                              width: '100%', // Ensure the Box takes full width of its parent
                            }}
                          >
                            {/* GWAS Initiate Button */}
                            <Box sx={{ flex: 1 }}>
                              <Tooltip
                                title={
                                  !isQcInitiateEnabled
                                    ? 'Cannot perform action at the moment'
                                    : role === 'receiver'
                                      ? 'Receiver cannot initiate QC calculations'
                                      : 'Initiate QC Calculations'
                                }
                                placement="bottom"
                              >
                                <span>
                                  <Button
                                    variant="contained"
                                    color="primary"
                                    onClick={handleQcInitiate}
                                    // disabled={!isQcInitiateEnabled || isQcInitiateLoading}
                                    startIcon={
                                      isQcInitiateLoading && (
                                        <CircularProgress size={20} color="inherit" />
                                      )
                                    }
                                    fullWidth
                                    sx={{ borderRadius: 20 }}
                                  >
                                    Initiate GWAS Calculation
                                  </Button>
                                </span>
                              </Tooltip>
                            </Box>

                            {/* GWAS Results Button */}
                            <Box sx={{ flex: 1 }}>
                              <Tooltip
                                title={
                                  !isQcResultsEnabled
                                    ? 'Results not available to preview' : role === 'receiver'
                                      ? 'Receiver cannot initiate QC results'
                                      : 'Get QC Results'
                                }
                                placement="bottom"
                              >
                                <span>
                                  <Button
                                    variant="outlined"
                                    color="secondary"
                                    onClick={handleQcResults}
                                    // disabled={!isQcResultsEnabled || isQcResultsLoading || role === 'receiver'}
                                    startIcon={
                                      isQcResultsLoading && (
                                        <CircularProgress size={20} color="inherit" />
                                      )
                                    }
                                    fullWidth
                                    sx={{ borderRadius: 20 }}
                                  >
                                    Get Results
                                  </Button>
                                </span>
                              </Tooltip>
                            </Box>
                          </Box>
                        }
                      />
                    </ListItem>
                  )}
                </>


              </List>

              {/* Save and Cancel Buttons */}
              {isEditing && (
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                  <Tooltip title="Save your changes" placement="top">
                    <Button
                      variant="contained"
                      color="primary"
                      startIcon={<SaveIcon />}
                      onClick={handleUpdateCollaboration}
                      sx={{ mr: 2 }}
                      disabled={isLoading}
                    >
                      {isLoading ? <CircularProgress size={24} color="inherit" /> : 'Save'}
                    </Button>
                  </Tooltip>
                  <Tooltip title="Cancel editing" placement="top">
                    <Button
                      variant="outlined"
                      color="secondary"
                      startIcon={<CancelIcon />}
                      onClick={() => {
                        // Revert to original values
                        setCollabName(originalCollabName);
                        setExperimentList([...originalExperimentList]);
                        setPhenotype(originalPhenoType);
                        setSamples(originalSamples);
                        setIsEditing(false);
                      }}
                      disabled={isLoading}
                    >
                      Cancel
                    </Button>
                  </Tooltip>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', marginBottom: '20px', border: '1px solid #ccc', borderRadius: 2, boxShadow: 'none' }}>
            <CardContent>
              {/* Header */}
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  <strong>Collaborators</strong>
                </Typography>
                <Divider sx={{ flexGrow: 30, borderColor: 'black' }} />
              </Box>
              <List>
                <ListItem disableGutters>
                  <ListItemText
                    primary={`${senderInfo.name} (Initiator)`}
                  />
                </ListItem>
                <Divider />

                {invitedUsers.map((user, index) => (
                  <ListItem key={index} disableGutters sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <ListItemText
                      primary={user.name}
                      secondary={`Phenotype: ${user.phenotype}`}
                    />
                    <Box>
                      {role === 'sender' && user.status === 'pending' ? (
                        <Tooltip title="Withdraw this invitation" placement="top">
                          <Button
                            variant="outlined"
                            color="secondary"
                            size="small"
                            onClick={() => handleWithdraw(user.user_id)}
                          >
                            Withdraw
                          </Button>
                        </Tooltip>
                      ) : role === 'receiver' && user.status === 'pending' ? (
                        <>
                          <Tooltip title="Accept this invitation" placement="top">
                            <Button
                              variant="outlined"
                              color="primary"
                              size="small"
                              onClick={() => handleAccept(user.user_id)}
                              sx={{ mr: 1 }}
                            >
                              Accept
                            </Button>
                          </Tooltip>
                          <Tooltip title="Reject this invitation" placement="top">
                            <Button
                              variant="outlined"
                              color="secondary"
                              size="small"
                              onClick={() => handleReject(user.user_id)}
                            >
                              Reject
                            </Button>
                          </Tooltip>
                        </>
                      ) : (
                        <StatusChip status={user.status} />
                      )}
                    </Box>
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box mt={4}>
        <Tooltip title="Save your collaboration details" placement="top">
          <Button
            variant="contained"
            color="primary"
            fullWidth
            size="large"
            onClick={handleUpdateCollaboration}
            disabled={isLoading || !isEditing} // Disable if not in edit mode
            sx={{ py: 2, fontSize: '1.1rem', borderRadius: 100 }}
          >
            {isLoading ? <CircularProgress size={24} color="inherit" /> : 'Update Collaboration'}
          </Button>
        </Tooltip>
      </Box>

      <Snackbar open={snackbar.open} autoHideDuration={6000} onClose={handleCloseSnackbar}>
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container >
  );
};

const StatusChip = ({ status }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'accepted':
        return { dot: alpha('#4caf50', 1.0), bg: alpha('#c8e6c9', 0.5) }; // Green
      case 'withdrawn':
        return { dot: alpha('#ff9800', 1.0), bg: alpha('#ffe0b2', 0.5) }; // Orange
      case 'rejected':
        return { dot: alpha('#f44336', 1.0), bg: alpha('#ffcdd2', 0.5) }; // Red
      default:
        return { dot: alpha('#1976d2', 1.0), bg: alpha('#bbdefb', 0.5) }; // Blue
    }
  };

  const { dot, bg } = getStatusColor(status);

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        backgroundColor: bg,
        borderRadius: 10,
        padding: '4px 8px',
        marginLeft: 1,
      }}
    >
      <Box
        sx={{
          width: 10,
          height: 10,
          borderRadius: '100%',
          backgroundColor: dot,
          marginRight: 1,
        }}
      />
      <Typography variant="body2" color={dot}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Typography>
    </Box>
  );
};

export default CollaborationDetails;
