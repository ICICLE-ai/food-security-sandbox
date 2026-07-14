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
  FormGroup,
  Switch,
  Slider,
  Divider
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
  const [checkingKAnonymity, setCheckingKAnonymity] = useState(false);
  const [kAnonymityResult, setKAnonymityResult] = useState(null);
  const [columnInfo, setColumnInfo] = useState({ numeric_columns: [], latitude_column: null, longitude_column: null });
  const [dpEnabled, setDpEnabled] = useState(false);
  const [excludedDpColumns, setExcludedDpColumns] = useState([]);
  const [locationPrivacyEnabled, setLocationPrivacyEnabled] = useState(false);
  const [epsilon, setEpsilon] = useState(5);
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
    setKAnonymityResult(null);
    setColumnInfo({ numeric_columns: [], latitude_column: null, longitude_column: null });
    setDpEnabled(false);
    setExcludedDpColumns([]);
    setLocationPrivacyEnabled(false);
    setEpsilon(5);
  }, [selectedDatasetId, privacyEnabled]);

  useEffect(() => {
    if (!privacyEnabled || !selectedDatasetId) return;

    const fetchColumnInfo = async () => {
      try {
        const response = await axios.post(
          `${process.env.REACT_APP_FARMER_API_URL}/api/get_dataset_column_info`,
          { datasetId: selectedDatasetId },
          {
            headers: {
              Authorization: `Bearer ${localStorage.getItem('tapis_token')}`
            }
          }
        );
        setColumnInfo({
          numeric_columns: response.data.numeric_columns || [],
          latitude_column: response.data.latitude_column || null,
          longitude_column: response.data.longitude_column || null
        });
      } catch (error) {
        console.error('Error fetching column info:', error);
      }
    };

    fetchColumnInfo();
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

  const numericColumnsInSelection = columnInfo.numeric_columns.filter((col) =>
    selectedDirectIds.includes(col)
  );
  const dpColumnsToApply = numericColumnsInSelection.filter(
    (col) => !excludedDpColumns.includes(col)
  );
  const hasLocationColumnsInSelection =
    !!columnInfo.latitude_column &&
    !!columnInfo.longitude_column &&
    selectedDirectIds.includes(columnInfo.latitude_column) &&
    selectedDirectIds.includes(columnInfo.longitude_column);

  const handleCheckKAnonymity = async () => {
    if (!selectedDatasetId) {
      setSnackbar({ open: true, message: 'Please select a dataset first.', severity: 'warning' });
      return;
    }
    if (selectedDirectIds.length === 0) {
      setSnackbar({
        open: true,
        message: 'Please select at least one direct identifier column.',
        severity: 'warning'
      });
      return;
    }

    try {
      setCheckingKAnonymity(true);
      const response = await axios.post(
        `${process.env.REACT_APP_FARMER_API_URL}/api/check_k_anonymity`,
        {
          datasetId: selectedDatasetId,
          direct_ids: selectedDirectIds
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('tapis_token')}`
          }
        }
      );

      setKAnonymityResult(response.data.k_anonymity);
      setSnackbar({
        open: true,
        message: 'K-anonymity evaluation complete.',
        severity: 'success'
      });
    } catch (error) {
      console.error('Error checking k-anonymity:', error);
      setKAnonymityResult(null);
      setSnackbar({
        open: true,
        message:
          error?.response?.data?.message ||
          'Failed to evaluate k-anonymity. Please try again.',
        severity: 'error'
      });
    } finally {
      setCheckingKAnonymity(false);
    }
  };

  const handleExportSelectedColumns = async () => {
    if (!selectedDatasetId) {
      setSnackbar({ open: true, message: 'Please select a dataset first.', severity: 'warning' });
      return;
    }
    if (selectedDirectIds.length === 0) {
      setSnackbar({
        open: true,
        message: 'Please select at least one column to export.',
        severity: 'warning'
      });
      return;
    }

    try {
      setExporting(true);
      const response = await axios.post(
        `${process.env.REACT_APP_FARMER_API_URL}/api/export_dataset_columns`,
        {
          datasetId: selectedDatasetId,
          columns: selectedDirectIds,
          differential_privacy: {
            enabled: dpEnabled && dpColumnsToApply.length > 0,
            epsilon,
            columns: dpColumnsToApply
          },
          location_privacy: {
            enabled: locationPrivacyEnabled && hasLocationColumnsInSelection,
            epsilon
          }
        },
        {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('tapis_token')}`
          },
          responseType: 'blob'
        }
      );

      const downloadUrl = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', `${selectedDataset?.dataset_name || 'dataset'}_selected_columns.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);

      setSnackbar({
        open: true,
        message: 'Selected columns exported successfully.',
        severity: 'success'
      });
    } catch (error) {
      console.error('Error exporting selected columns:', error);
      setSnackbar({
        open: true,
        message:
          error?.response?.data?.message ||
          'Failed to export selected columns. Please try again.',
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

  const handleToggleDpColumn = (columnName) => {
    setExcludedDpColumns((prev) =>
      prev.includes(columnName)
        ? prev.filter((item) => item !== columnName)
        : [...prev, columnName]
    );
  };

  const handleToggleDirectId = (columnName) => {
    setSelectedDirectIds((prev) =>
      prev.includes(columnName)
        ? prev.filter((item) => item !== columnName)
        : [...prev, columnName]
    );
    setKAnonymityResult(null);
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

            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
              <Button
                variant="outlined"
                onClick={handleCheckKAnonymity}
                disabled={loading || checkingKAnonymity || selectedDirectIds.length === 0}
                sx={{
                  color: '#008000',
                  borderColor: '#008000',
                  '&:hover': { borderColor: '#009900', backgroundColor: '#f3faf3' },
                  borderRadius: 2,
                  px: 3
                }}
              >
                {checkingKAnonymity ? 'Evaluating...' : 'Evaluate K-Anonymity'}
              </Button>
            </Box>

            {kAnonymityResult && (
              <Box
                sx={{
                  mt: 2,
                  p: 1.5,
                  borderRadius: 1,
                  border: '1px solid #ddd',
                  backgroundColor: kAnonymityResult.is_k_anonymous ? '#f3faf3' : '#fdf3f3'
                }}
              >
                <Typography variant="body2">
                  {kAnonymityResult.is_k_anonymous
                    ? `Dataset satisfies k-anonymity (k=${kAnonymityResult.target_k}).`
                    : `Dataset does NOT satisfy k-anonymity (k=${kAnonymityResult.target_k}).`}
                </Typography>
                <Typography variant="body2">
                  Smallest group size found: {kAnonymityResult.actual_min_k}
                </Typography>
                <Typography variant="body2">
                  Vulnerable rows: {kAnonymityResult.vulnerable_rows_count}
                </Typography>
              </Box>
            )}

            {(numericColumnsInSelection.length > 0 || hasLocationColumnsInSelection) && (
              <>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1" sx={{ mb: 1 }}>
                  Privacy-Preserving Export
                </Typography>

                {numericColumnsInSelection.length > 0 && (
                  <>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={dpEnabled}
                          onChange={(event) => setDpEnabled(event.target.checked)}
                          sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#008000' }, '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { backgroundColor: '#008000' } }}
                        />
                      }
                      label="Differential Privacy (add noise to numeric columns)"
                    />

                    {dpEnabled && (
                      <Box sx={{ pl: 4, mb: 1 }}>
                        <Typography variant="body2" sx={{ color: 'text.secondary', mb: 0.5 }}>
                          Columns to noise (uncheck any you want to keep exact, e.g. label/target columns):
                        </Typography>
                        <FormGroup row>
                          {numericColumnsInSelection.map((columnName) => (
                            <FormControlLabel
                              key={columnName}
                              control={
                                <Checkbox
                                  size="small"
                                  checked={!excludedDpColumns.includes(columnName)}
                                  onChange={() => handleToggleDpColumn(columnName)}
                                  sx={{
                                    color: '#008000',
                                    '&.Mui-checked': { color: '#008000' }
                                  }}
                                />
                              }
                              label={columnName}
                            />
                          ))}
                        </FormGroup>
                      </Box>
                    )}
                  </>
                )}

                {hasLocationColumnsInSelection && (
                  <FormControlLabel
                    sx={{ display: 'block' }}
                    control={
                      <Switch
                        checked={locationPrivacyEnabled}
                        onChange={(event) => setLocationPrivacyEnabled(event.target.checked)}
                        sx={{ '& .MuiSwitch-switchBase.Mui-checked': { color: '#008000' }, '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { backgroundColor: '#008000' } }}
                      />
                    }
                    label="Location Privacy (anonymize GPS coordinates)"
                  />
                )}

                {(dpEnabled || locationPrivacyEnabled) && (
                  <Box sx={{ px: 1, mt: 1 }}>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      Privacy budget (epsilon): {epsilon} — lower means more noise/privacy, higher means less noise/more accuracy
                    </Typography>
                    <Slider
                      value={epsilon}
                      onChange={(event, value) => setEpsilon(value)}
                      min={0.1}
                      max={10}
                      step={0.1}
                      valueLabelDisplay="auto"
                      sx={{ color: '#008000', maxWidth: 400 }}
                    />
                  </Box>
                )}
              </>
            )}
          </Box>
        )}

        <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            variant="contained"
            startIcon={<ShareIcon />}
            onClick={handleExportSelectedColumns}
            disabled={loading || datasets.length === 0 || exporting}
            sx={{
              backgroundColor: '#008000',
              '&:hover': { backgroundColor: '#009900' },
              borderRadius: 2,
              px: 3
            }}
          >
            {exporting ? 'Exporting...' : 'Export Selected Columns'}
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
