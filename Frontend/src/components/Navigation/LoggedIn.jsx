import React, { useEffect, useState } from 'react';
import { Button, IconButton, Tooltip, Avatar, Menu, MenuItem, Badge } from "@mui/material";
import { Link as RouterLink, useNavigate } from "react-router-dom";
import axios from 'axios';
import URL from '../../config';

export default function LoggedIn({ onLogout }) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [collabAnchorEl, setCollabAnchorEl] = useState(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [userName, setUserName] = useState('');
  const open = Boolean(anchorEl);
  const collabOpen = Boolean(collabAnchorEl);
  const navigate = useNavigate();

  useEffect(() => {

    const fetchUserName = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await axios.get(`${URL}/api/profile`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });
        setUserName(response.data.name);
      } catch (error) {
        console.error('There was an error fetching the user name!', error);
      }
    };

    fetchUserName();
  }, []);

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
    setCollabAnchorEl(null);
  };

  const handleLogout = async () => {
    try {
      const token = localStorage.getItem('token');
      await axios.post(`${URL}/api/logout`, {}, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      localStorage.removeItem('token');
      onLogout();
      navigate('/login');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      handleClose();
    }
  };

  const getInitials = (name) => name ? name.charAt(0).toUpperCase() : '';

  return (

    <>
      <Button color="inherit" component={RouterLink} to="/">Home</Button>
      <Button color="inherit" component={RouterLink} to="/training">Collaborative Machine Learning</Button>
      <Button color="inherit" component={RouterLink} to="/chat">Chat</Button>
      <Button color="inherit" component={RouterLink} to="/dataSharing">Data Sharing</Button>
      

      <Tooltip title="Account settings">
        <IconButton
          onClick={handleClick}
          size="small"
          sx={{ ml: 2 }}
          aria-controls={open ? 'account-menu' : undefined}
          aria-haspopup="true"
          aria-expanded={open ? 'true' : undefined}
        >
          <Avatar sx={{ bgcolor: 'white', color:'blue', width: 32, height: 32 }}>
            {getInitials(userName)}
          </Avatar>
        </IconButton>
      </Tooltip>

      <Menu
        id="account-menu"
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
      >
        <MenuItem component={RouterLink} to="/profile" onClick={handleClose}>Edit Profile</MenuItem>
        <MenuItem onClick={handleLogout}>Logout</MenuItem>
      </Menu>
    </>

    // <>
    //   <Button color="inherit" component={RouterLink} to="/">Home</Button>
    //   <Button color="inherit" component={RouterLink} to="/upload">Upload</Button>
      
    //   <Badge badgeContent={pendingCount} color="error">
    //     <Button
    //       color="inherit"
    //       onMouseEnter={(e) => setCollabAnchorEl(e.currentTarget)}
    //     >
    //       Collaboration
    //     </Button>
    //   </Badge>
      
    //   <Menu
    //     anchorEl={collabAnchorEl}
    //     open={collabOpen}
    //     onClose={handleClose}
    //     MenuListProps={{ onMouseLeave: handleClose }}
    //   >
    //     <MenuItem component={RouterLink} to="/collaboration" onClick={handleClose}>
    //       All Collaborations
    //     </MenuItem>
    //     <MenuItem component={RouterLink} to="/start-collaboration" onClick={handleClose}>
    //       Start Collaboration
    //     </MenuItem>
    //   </Menu>

    //   <Button color="inherit" component={RouterLink} to="/session">Session</Button>

    //   <Tooltip title="Account settings">
    //     <IconButton
    //       onClick={handleClick}
    //       size="small"
    //       sx={{ ml: 2 }}
    //       aria-controls={open ? 'account-menu' : undefined}
    //       aria-haspopup="true"
    //       aria-expanded={open ? 'true' : undefined}
    //     >
    //       <Avatar sx={{ bgcolor: 'white', color:'blue', width: 32, height: 32 }}>
    //         {getInitials(userName)}
    //       </Avatar>
    //     </IconButton>
    //   </Tooltip>

    //   <Menu
    //     id="account-menu"
    //     anchorEl={anchorEl}
    //     open={open}
    //     onClose={handleClose}
    //   >
    //     <MenuItem component={RouterLink} to="/profile" onClick={handleClose}>Edit Profile</MenuItem>
    //     <MenuItem component={RouterLink} to="/" onClick={handleClose}>Change Password</MenuItem>
    //     <MenuItem onClick={handleLogout}>Logout</MenuItem>
    //   </Menu>
    // </>
  );
}