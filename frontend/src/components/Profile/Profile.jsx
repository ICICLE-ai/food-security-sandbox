import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, TextField, Button, Typography, Box } from '@mui/material';




const Profile = () => {
  const getToken = () => {
    return localStorage.getItem('tapis_token');
  };
  const token = getToken();
  
  const [user, setUser] = useState({
    name: '',
    email: '',
    currentPassword: '',
    newPassword: '',
    confirmNewPassword: '',
  });

  useEffect(() => {
    axios.get(`${process.env.REACT_APP_API_URL}/api/profile`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
      .then(response => {
        setUser(prevState => ({
          ...prevState,
          email: response.data.email,
          name: response.data.name
        }));
      })
      .catch(error => {
        console.error('There was an error fetching the user data!', error);
      });
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setUser(prevState => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    axios.put(`${process.env.REACT_APP_API_URL}/api/profile`, user, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
      .then(response => {
        if (response.status === 200) {
          alert('Profile updated successfully!');
          setUser({
            currentPassword: '',
            newPassword: '',
            confirmNewPassword: '',
          });
        }
      })
      .catch(error => {
        console.error('There was an error updating the profile!', error);
        alert(error.response.data.message);
      });
  };


  return (
    <Container maxWidth="sm" sx={{ marginTop: 5 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        Edit Profile
      </Typography>
      <form onSubmit={handleSubmit}>
        <Box mb={2}>
          <TextField
            fullWidth
            label="Name"
            variant="outlined"
            id="name"
            name="name"
            value={user.name}
            onChange={handleChange}
          />
        </Box>
        <Box mb={2}>
          <TextField
            fullWidth
            label="Email"
            variant="outlined"
            id="email"
            name="email"
            value={user.email}
            InputProps={{
              readOnly: true,
            }}
          />
        </Box>
        <Box mb={2}>
          <TextField
            fullWidth
            label="Current Password"
            variant="outlined"
            type="password"
            id="currentPassword"
            name="currentPassword"
            value={user.currentPassword}
            onChange={handleChange}
          />
        </Box>
        <Box mb={2}>
          <TextField
            fullWidth
            label="New Password"
            variant="outlined"
            type="password"
            id="newPassword"
            name="newPassword"
            value={user.newPassword}
            onChange={handleChange}
          />
        </Box>
        <Box mb={2}>
          <TextField
            fullWidth
            label="Confirm New Password"
            variant="outlined"
            type="password"
            id="confirmNewPassword"
            name="confirmNewPassword"
            value={user.confirmNewPassword}
            onChange={handleChange}
          />
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
          <Button
            type="submit"
            variant="contained"
            sx={{ 
              mt: 3,
              mb: 2,
              px: 4,
              backgroundColor: '#008000',
              '&:hover': {
                backgroundColor: '#009900',
              }
            }}
          >
            Update Profile
          </Button>
        </Box>
      </form>
    </Container>
  );
};

export default Profile;
