import { useState, useEffect } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { TextField, Button, Container, Typography, Box, CircularProgress, Divider, styled, IconButton, InputAdornment } from '@mui/material';
import { GoogleLogin } from '@react-oauth/google';
import axios from 'axios';
import { Visibility, VisibilityOff } from '@mui/icons-material';
import './Login.css';
import URL from '../../config';
import { jwtDecode } from "jwt-decode";
import loginBG from "../../assets/loginBG.jpg"
import PropTypes from 'prop-types';

const WhiteButton = styled(Button)({
  background: '#fff',
  color: '#008000',
  border: '1px solid #008000',
  '&:hover': {
    background: '#008000',
    color: '#fff',
  },
});

const Login = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is already logged in, if yes, redirect to home page
    const token = localStorage.getItem('token');
    if (token) {
      navigate('/');
    }
  }, [navigate]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post(`${URL}/api/login`, {
        email: username,
        password,
      });

      if (response.data.message === 'Login successful' && response.data.token) {
        localStorage.setItem('token', response.data.token); // Store token in localStorage
        onLogin(); // Set isLoggedIn to true in App component
        navigate('/'); // Redirect to the home page upon successful login
      }
    } catch (err) {
      setError(err.response?.data?.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleTogglePasswordVisibility = () => {
    setShowPassword((prevShowPassword) => !prevShowPassword);
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
      {loading ? (
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center',
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          borderRadius: '50%',
          padding: 2
        }}>
          <CircularProgress size={60} sx={{ color: '#008000' }} />
        </Box>
      ) : (
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
            <Typography variant="h5" align="center" gutterBottom>Welcome back, Login</Typography>
            <form onSubmit={handleLogin}>
              <TextField
                margin="normal"
                required
                fullWidth
                id="username"
                label="Email"
                name="username"
                autoComplete="username"
                autoFocus
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
              <TextField
                margin="normal"
                required
                fullWidth
                name="password"
                label="Password"
                type={showPassword ? 'text' : 'password'}
                id="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
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
              {error && (
                <Typography variant="body2" color="error" sx={{ mt: 1, mb: 1 }}>
                  {error}
                </Typography>
              )}
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
                {loading ? <CircularProgress size={24} /> : 'Login'}
              </Button>
              <Box sx={{ display: 'flex', justifyContent: 'center', width: '100%', mb: 2 }}>
                <GoogleLogin
                  onSuccess={credentialResponse => {
                    console.log(jwtDecode(credentialResponse['credential']));
                  }}
                  onError={() => {
                    console.log('Login Failed');
                  }}
                />
              </Box>
              <Divider sx={{ mb: 1 }}>or</Divider>
              <Button 
                component={RouterLink} 
                to="/forgot/password" 
                fullWidth 
                variant="text" 
                sx={{ 
                  mb: 1,
                  color: '#008000',
                  '&:hover': {
                    backgroundColor: 'rgba(0, 128, 0, 0.04)', // Light green background on hover
                  }
                }}
              >
                Forgot Password?
              </Button>
            </form>
            <Divider sx={{ mt: 2, mb: 1 }}>Don&apos;t have an account?</Divider>
            <WhiteButton component={RouterLink} to="/register" fullWidth variant="outlined">
              Register New Account
            </WhiteButton>
          </Box>
        </Container>
      )}
    </Box>
  );
};

Login.propTypes = {
  onLogin: PropTypes.func.isRequired
};

export default Login;