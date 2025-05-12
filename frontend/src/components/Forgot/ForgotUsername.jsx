// ForgotUsernamePage.js
import React from 'react';
import { Container, Typography, TextField, Button } from '@mui/material';

const ForgotUsername = () => {
  return (
    <Container maxWidth="sm">
      <Typography variant="h4" gutterBottom>
        Forgot Username
      </Typography>
      <TextField
        label="Email"
        fullWidth
        margin="normal"
        variant="outlined"
        // Add onChange and value props to handle input changes
      />
      <Button variant="contained" color="primary">
        Submit
      </Button>
    </Container>
  );
};

export default ForgotUsername;
