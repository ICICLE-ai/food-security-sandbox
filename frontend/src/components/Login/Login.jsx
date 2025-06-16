import { useState, useEffect } from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { TextField, Button, Container, Typography, Box, CircularProgress, Divider, styled, IconButton, InputAdornment } from '@mui/material';
import axios from 'axios';
import { Visibility, VisibilityOff } from '@mui/icons-material';
import './Login.css';
import tacc_logo from '../../tacc-black.png'
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

const Login = ({ setIsAuthenticated }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const navigate = useNavigate();

 
   const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      console.log(process.env.REACT_APP_API_URL);
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/auth/login`, {
        username,
        password
      });

      console.log(response.data.token)
      
      if (response.data.token) {
        localStorage.setItem('tapis_token', response.data.token);
        localStorage.setItem('tapis_username', response.data.username);
        setIsAuthenticated(true);
        setError('');
        navigate('/');
      }
    } catch (err) {
      console.log(err)
      setError(err.response?.data?.message || 'Login failed');
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
            <Typography align="center" gutterBottom>
              <img src={tacc_logo} alt="My Logo" />
            </Typography>
            {/* <Typography variant="h6" align="center" gutterBottom>Log In</Typography> */}
            <form onSubmit={handleLogin}>
              <TextField
                margin="normal"
                required
                fullWidth
                id="username"
                label="TACC Username"
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
              <Divider sx={{ mb: 1 }}>or</Divider>
              <Button 
                component={RouterLink}
                onClick={()=>{window.open("https://accounts.tacc.utexas.edu/forgot_password", '_blank')}}
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
            <WhiteButton component={RouterLink} onClick={()=>{window.open("https://accounts.tacc.utexas.edu/register", '_blank')}} fullWidth variant="outlined">
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