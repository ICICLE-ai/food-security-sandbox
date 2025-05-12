import React, { useState } from 'react';
import { Container, Typography, TextField, Button, Box, CircularProgress, Divider, styled, IconButton, InputAdornment } from '@mui/material';
import { Visibility, VisibilityOff } from '@mui/icons-material';
import { Link as RouterLink } from 'react-router-dom';

const WhiteButton = styled(Button)({
  background: '#fff',
  color: '#1976d2',
  border: '1px solid #1976d2',
  '&:hover': {
    background: '#1976d2',
    color: '#fff',
  },
});

const ForgotPassword = () => {
  const [loading, setLoading] = useState(false);

  const handleSubmit = () => {
    // Add your submission logic here
  };

  return (
    <Container maxWidth="xs">
      <Box sx={{ marginTop: 3, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Box sx={{ border: '1px dashed #ccc', borderRadius: 5, padding: 3, width: '100%', boxSizing: 'border-box' }}>
          <Typography variant="h5" gutterBottom align="center">
            Forgot Password
          </Typography>
          <TextField
            label="Email"
            fullWidth
            margin="normal"
            variant="outlined"
            // Add onChange and value props to handle input changes
          />
          <Button type="submit" fullWidth variant="contained" color="primary" onClick={handleSubmit} sx={{ mt: 2 }}>
            {loading ? <CircularProgress size={24} /> : 'Submit'}
          </Button>
          <Divider sx={{ mt: 2, mb: 1 }}>or</Divider>
          <Button component={RouterLink} to="/login" fullWidth variant="text" color="primary" sx={{ mb: 1 }}>
            Back to Login
          </Button>
          <Divider sx={{ mt: 2, mb: 1 }}>Don't have an account?</Divider>
          <WhiteButton component={RouterLink} to="/register" fullWidth variant="outlined">
            Register New Account
          </WhiteButton>
        </Box>
      </Box>
    </Container>
  );
};

export default ForgotPassword;
