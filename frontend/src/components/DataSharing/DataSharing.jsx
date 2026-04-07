import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Avatar,
  Button,
  Checkbox,
  FormControlLabel,
  Snackbar,
  Alert,
  CircularProgress,
  Modal,
  FormGroup
} from '@mui/material';
import DatasetIcon from '@mui/icons-material/Dataset';
import ShareIcon from '@mui/icons-material/Share';
import InfoIcon from '@mui/icons-material/Info';
import CloseIcon from '@mui/icons-material/Close';

const DataSharing = () => {
  const [datasets, setDatasets] = useState([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [privacyEnabled, setPrivacyEnabled] = useState(true);
  const [selectedDirectIds, setSelectedDirectIds] = useState([]);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [openModal, setOpenModal] = useState(false);
  const [selectedDatasetInfo, setSelectedDatasetInfo] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await axios.get(
          `${process.env.REACT_APP_FARMER_API_URL}/api/get_user_datasets`,
          {
            headers: {
              Authorization: `Bearer ${localStorage.getItem('tapis_token')}`
            }
          }
        );

        setDatasets(response.data || []);
      } catch (error) {
        console.error('Error fetching datasets:', error);
        setSnackbar({
          open: true,
          message: 'Unable to load datasets. Please try again.',
          severity: 'error'
        });
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, []);

  useEffect(() => {
    setSelectedDirectIds([]);
  }, [selectedDatasetId, privacyEnabled]);

  const parseMetadata = (metadataValue) => {
    if (Array.isArray(metadataValue)) return metadataValue;
    if (typeof metadataValue !== 'string') return [];

    try {
      const parsed = JSON.parse(metadataValue);
      return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
      try {
        const normalized = metadataValue.replace(/'/g, '"');
        const parsed = JSON.parse(normalized);
        return Array.isArray(parsed) ? parsed : [];
      } catch (e) {
        return [];
      }
    }
  };

  const selectedDataset = datasets.find((dataset) => dataset._id === selectedDatasetId);
  const selectedDatasetMetadata = parseMetadata(selectedDataset?.metadata);

  const handleExportToFeast = async () => {
    if (!selectedDatasetId) {
      setSnackbar({
        open: true,
        message: 'Please select a dataset first.',
        severity: 'warning'
      });
      return;
    }

    try {
      setExporting(true);
      const response = await axios.post(
        `${process.env.REACT_APP_FARMER_API_URL}/api/export_dataset_to_feast`,
        {
          datasetId: selectedDatasetId,
          privacyEnabled,
          direct_ids: selectedDirectIds
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('tapis_token')}`
          }
        }
      );
      console.log('Export dataset response:', response.data);

      setSnackbar({
        open: true,
        message: 'Dataset export to FEAST was triggered successfully.',
        severity: 'success'
      });
    } catch (error) {
      console.error('Error exporting dataset to FEAST:', error);
      setSnackbar({
        open: true,
        message:
          error?.response?.data?.message ||
          'Failed to export dataset to FEAST. Please try again.',
        severity: 'error'
      });
    } finally {
      setExporting(false);
    }
  };

  const handleOpenModal = (dataset) => {
    setSelectedDatasetInfo(dataset);
    setOpenModal(true);
  };

  const handleCloseModal = () => {
    setOpenModal(false);
  };

  const handleToggleDirectId = (columnName) => {
    setSelectedDirectIds((prev) =>
      prev.includes(columnName)
        ? prev.filter((item) => item !== columnName)
        : [...prev, columnName]
    );
  };

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
          mb: 3
        }}
      >
        Data Sharing to Other Projects
      </Typography>

      <Box
        sx={{
          maxWidth: '900px',
          margin: '0 auto',
          border: '1px solid #ccc',
          borderRadius: 2,
          p: 3,
          backgroundColor: '#fff'
        }}
      >
        <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
          Select one of your uploaded datasets
        </Typography>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
            <CircularProgress sx={{ color: '#008000' }} />
          </Box>
        ) : datasets.length === 0 ? (
          <Typography variant="body1">No uploaded datasets found.</Typography>
        ) : (
          <List sx={{ width: '100%', bgcolor: 'background.paper', mb: 2 }}>
            {datasets.map((dataset) => (
              <ListItem
                key={dataset._id}
                onClick={() => setSelectedDatasetId(dataset._id)}
                sx={{
                  border: selectedDatasetId === dataset._id ? '2px solid #008000' : '1px solid #ccc',
                  borderRadius: 1,
                  mb: 1,
                  cursor: 'pointer',
                  backgroundColor: selectedDatasetId === dataset._id ? '#f3faf3' : '#fff',
                  '&:hover': {
                    backgroundColor: '#f0f0f0'
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
                  secondary={`Records: ${dataset.num_records ?? 0}`}
                />
                <Button
                  onClick={(event) => {
                    event.stopPropagation();
                    handleOpenModal(dataset);
                  }}
                >
                  <InfoIcon />
                </Button>
              </ListItem>
            ))}
          </List>
        )}

        <FormControlLabel
          control={
            <Checkbox
              checked={privacyEnabled}
              onChange={(event) => setPrivacyEnabled(event.target.checked)}
              sx={{
                color: '#008000',
                '&.Mui-checked': {
                  color: '#008000'
                }
              }}
            />
          }
          label="Privacy Enabled"
          sx={{ mb: 2 }}
        />

        {privacyEnabled && selectedDatasetId && selectedDatasetMetadata.length > 0 && (
          <Box
            sx={{
              border: '1px solid #ddd',
              borderRadius: 1,
              p: 2,
              mb: 2,
              backgroundColor: '#fafafa'
            }}
          >
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Select direct identifier columns
            </Typography>
            <Typography variant="body2" sx={{ mb: 1.5, color: 'text.secondary' }}>
              Choose one or more columns to pass as direct identifiers for privacy checks.
            </Typography>
            <FormGroup>
              {selectedDatasetMetadata.map((columnName) => (
                <FormControlLabel
                  key={columnName}
                  control={
                    <Checkbox
                      checked={selectedDirectIds.includes(columnName)}
                      onChange={() => handleToggleDirectId(columnName)}
                      sx={{
                        color: '#008000',
                        '&.Mui-checked': {
                          color: '#008000'
                        }
                      }}
                    />
                  }
                  label={columnName}
                />
              ))}
            </FormGroup>
          </Box>
        )}

        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            startIcon={<ShareIcon />}
            onClick={handleExportToFeast}
            disabled={loading || datasets.length === 0 || exporting}
            sx={{
              backgroundColor: '#008000',
              '&:hover': { backgroundColor: '#009900' },
              borderRadius: 2,
              px: 3
            }}
          >
            {exporting ? 'Exporting...' : 'Export Data to FEAST'}
          </Button>
        </Box>
      </Box>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
      >
        <Alert
          onClose={() => setSnackbar((prev) => ({ ...prev, open: false }))}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

      <Modal open={openModal} onClose={handleCloseModal}>
        <Box
          sx={{
            bgcolor: 'white',
            border: '1px solid black',
            borderRadius: '4px',
            padding: 2,
            position: 'relative',
            width: { xs: '90%', sm: '60%', md: '40%' },
            margin: 'auto',
            top: '50%',
            transform: 'translateY(-50%)'
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
          <Typography>
            Metadata:{' '}
            {Array.isArray(selectedDatasetInfo?.metadata)
              ? selectedDatasetInfo.metadata.join(', ')
              : selectedDatasetInfo?.metadata ?? 'N/A'}
          </Typography>
        </Box>
      </Modal>
    </Box>
  );
};

export default DataSharing;
