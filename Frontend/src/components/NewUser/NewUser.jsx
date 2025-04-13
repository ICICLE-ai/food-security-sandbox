import { useState } from 'react';
import { TextField, Button, Container, Typography, Box, CircularProgress, styled, Divider, InputAdornment, IconButton } from '@mui/material';
import axios from 'axios';
import { Link as RouterLink } from 'react-router-dom';
import { Visibility, VisibilityOff } from '@mui/icons-material';
import URL from '../../config';
import loginBG from "../../assets/loginBG.jpg";

const WhiteButton = styled(Button)({
  background: '#fff',
  color: '#008000',
  border: '1px solid #008000',
  '&:hover': {
    background: '#008000',
    color: '#fff',
  },
});

const StyledTextField = styled(TextField)({
  '& .MuiFilledInput-root': {
    backgroundColor: '#f2f2f2',
  },
  '& .MuiFilledInput-underline:after': {
    borderBottomColor: '#1976d2',
  },
});

const NewUser = () => {
  const [username, setUsername] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      alert('Passwords do not match!');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${URL}/api/register`, {
        name,
        email: username,
        password,
      });

      console.log('Response:', response.data);
      setError('');
      setSuccess('User created successfully');
      alert('Registration successful! Please login to continue.');
      setUsername('');
      setName('');
      setPassword('');
      setConfirmPassword('');
    } catch (err) {
      console.error('Error:', err.response?.data?.message || 'An error occurred');
      console.error('Error headers:', err.response?.headers);
      const errorMessage = err.response?.data?.message || 'Registration failed. Please try again.';
      setError(errorMessage);
      setSuccess('');
      alert(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleTogglePasswordVisibility = () => {
    setShowPassword((prevShowPassword) => !prevShowPassword);
  };

  const handleConfirmPasswordChange = (e) => {
    setConfirmPassword(e.target.value);
  };

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 64,
        left: 0,
        right: 0,
        bottom: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundImage: `url(${loginBG})`,
        backgroundRepeat: 'no-repeat',
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        zIndex: 0,
      }}
    >
      <Container component="main" maxWidth="xs">
        <Box sx={{ 
          border: '1px dashed #ccc', 
          borderRadius: 5, 
          padding: 3, 
          width: '100%', 
          boxSizing: 'border-box',
          backgroundColor: "white",
          maxHeight: 'calc(100vh - 96px)',
          overflowY: 'auto',
          my: 2,
        }}>
          <Typography variant="h5" align="center" gutterBottom>Registration</Typography>
          <form onSubmit={handleSubmit}>
            <StyledTextField
              fullWidth
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              id="name"
              label="Name"
              margin="normal"
            />
            <StyledTextField
              fullWidth
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              id="username"
              label="Email"
              margin="normal"
            />
            <StyledTextField
              fullWidth
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              id="password"
              label="Password"
              type={showPassword ? 'text' : 'password'}
              margin="normal"
              InputProps={{
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton onClick={handleTogglePasswordVisibility}>
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
            />
            <StyledTextField
              fullWidth
              required
              value={confirmPassword}
              onChange={handleConfirmPasswordChange}
              id="confirm-password"
              label="Confirm Password"
              type={showPassword ? 'text' : 'password'}
              margin="normal"
            />
            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={loading}
              sx={{ 
                mt: 3, 
                mb: 2,
                backgroundColor: '#008000',
                '&:hover': {
                  backgroundColor: '#009900',
                }
              }}
            >
              {loading ? <CircularProgress size={24} /> : 'Submit'}
            </Button>
          </form>
          <Divider sx={{ mb: 1 }}>Already have an account?</Divider>
          <WhiteButton component={RouterLink} to="/login" fullWidth variant="outlined">
            <Typography variant="body1" align="center">Login here</Typography>
          </WhiteButton>
        </Box>
      </Container>
    </Box>
  );
};

export default NewUser;
