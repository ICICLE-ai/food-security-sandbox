import React from 'react';
import { Button } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

const LoggedOut = () => {
  return (
    <>
      <Button color="inherit" component={RouterLink} to="/">
        Home
      </Button>
      <Button color="inherit" component={RouterLink} to="/login">
        Login
      </Button>
      <Button color="inherit" component={RouterLink} to="/register">
        Register
      </Button>
    </>
  );
};

export default LoggedOut;
