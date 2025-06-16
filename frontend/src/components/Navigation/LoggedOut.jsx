import React from 'react';
import { Button } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";

const LoggedOut = () => {
  return (
    <>
      <Button color="inherit" component={RouterLink} to="/">
        Home
      </Button>
      {/* <Button color="inherit" component={RouterLink} to="/">
        Login
      </Button> */}
      <Button color="inherit" component={RouterLink} onClick={()=>{window.open("https://accounts.tacc.utexas.edu/register", '_blank')}}>
        Register
      </Button>
    </>
  );
};

export default LoggedOut;
