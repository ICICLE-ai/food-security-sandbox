import React, { useState } from 'react';
import { TextField, Button, Container, Typography, Box, Paper, IconButton } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import axios from 'axios';
import URL from '../../config';
import { CircularProgress } from '@mui/material';

const UploadForm = () => {
    const [field1, setField1] = useState('');
    const [csvFile, setCSVFile] = useState(null);
    const [fileName, setFileName] = useState('');
    const getToken = () => {
        return localStorage.getItem('token');
    };

    const [loading, setLoading] = useState(false);

    const handleField1Change = (event) => {
        setField1(event.target.value);
    };


    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setCSVFile(file);
        setFileName(file.name);
    };

    const handleDrop = (event) => {
        event.preventDefault();
        const file = event.dataTransfer.files[0];
        setCSVFile(file);
        setFileName(file.name);
    };

    const handleDragOver = (event) => {
        event.preventDefault();
    };
    const token = getToken();
            
    const handleSubmit = async (event) => {
        event.preventDefault();
        
        // Check if Dataset Name is empty
        if (!field1.trim()) {
            alert('Please enter a Dataset Name');
            return;
        }

        // Check if file is selected
        if (!csvFile) {
            alert('No file selected.');
            return;
        }

        setLoading(true);

        try {
            const formData = new FormData();
            formData.append('field1', field1);
            formData.append('file', csvFile);

            const response = await axios.post(`${URL}/api/upload_csv`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                    'Authorization': `Bearer ${token}`
                },
                withCredentials: true
            });

            if (response.status === 200) {
                setField1('');
                setCSVFile(null);
                setFileName('');
                alert(response.data.message)
            } else {
                throw new Error('Upload failed.');
            }
        } catch (error) {
            alert('Error uploading dataset: ' + error);
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container component="div" maxWidth="sm" sx={{ padding: 4, borderRadius: 2 }}>
            <Typography variant="h4" align="center" gutterBottom sx={{ fontWeight: 'light' }}>
                Farm Dataset Upload
            </Typography>
            <form onSubmit={handleSubmit}>
                <TextField
                    fullWidth
                    label="Dataset Name"
                    id="field1"
                    name="field1"
                    value={field1}
                    onChange={handleField1Change}
                    variant="outlined"
                    margin="normal"
                />
                <Box
                    sx={{
                        border: '2px dashed #ccc',
                        borderRadius: '8px',
                        padding: 2,
                        textAlign: 'center',
                        marginTop: 2,
                        backgroundColor: '#fafafa',
                        cursor: 'pointer',
                        transition: 'background-color 0.3s',
                        '&:hover': {
                            backgroundColor: '#f0f0f0',
                        },
                    }}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                >
                    <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        style={{ display: 'none' }}
                        id="csv-file-input"
                    />
                    <label htmlFor="csv-file-input" style={{ display: 'block', cursor: 'pointer' }}>
                        <IconButton 
                            component="span"
                            sx={{ 
                                color: '#008000',
                                '&:hover': {
                                    color: '#009900',
                                }
                            }}
                        >
                            <CloudUploadIcon sx={{ fontSize: 40 }} />
                        </IconButton>
                        <Typography variant="body1">
                            {fileName || 'Upload Farm Dataset'}
                        </Typography>
                    </label>
                </Box>
                {loading ? (
                    <Box display="flex" justifyContent="center" alignItems="center" mt={2} sx={{
                        backgroundColor: '#fafafa',
                    }}>
                        <CircularProgress />
                    </Box>
                ) : (
                    <Button
                        type="submit"
                        variant="contained"
                        fullWidth
                        sx={{ 
                            marginTop: 2, 
                            padding: 2, 
                            fontSize: '1rem', 
                            borderRadius: 100,
                            backgroundColor: '#008000',
                            '&:hover': {
                                backgroundColor: '#009900',
                            }
                        }}
                    >
                        Upload Dataset
                    </Button>
                )}
            </form>
        </Container>
    );
};

export default UploadForm;