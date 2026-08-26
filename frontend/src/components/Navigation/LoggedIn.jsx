import React, { useEffect, useRef, useState } from 'react';
import { Button, IconButton, Tooltip, Avatar, Menu, MenuItem, Badge } from "@mui/material";
import { Link as RouterLink, useNavigate, useLocation } from "react-router-dom";
import axios from 'axios';
import socket from '../../socket';


export default function LoggedIn({ onLogout }) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [collabAnchorEl, setCollabAnchorEl] = useState(null);
  const [pendingCount, setPendingCount] = useState(0);
  const [userName, setUserName] = useState(localStorage.getItem('tapis_username'));
  const open = Boolean(anchorEl);
  const collabOpen = Boolean(collabAnchorEl);
  const navigate = useNavigate();
  const location = useLocation();
  const token = localStorage.getItem('tapis_token');

  const pathnameRef = useRef(location.pathname);
  useEffect(() => {
    pathnameRef.current = location.pathname;
    if (location.pathname === '/chat') {
      setPendingCount(0);
    }
  }, [location.pathname]);

  useEffect(() => {
    const handleNewMessage = (msg) => {
      if (msg.receiverID === userName && pathnameRef.current !== '/chat') {
        setPendingCount((prev) => prev + 1);
      }
    };
    socket.on('new_message', handleNewMessage);
    return () => socket.off('new_message', handleNewMessage);
  }, [userName]);


  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
    setCollabAnchorEl(null);
  };

  const handleLogout = async () => {
    try {
      onLogout();
      navigate('/');
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
      <Badge badgeContent={pendingCount} color="error" overlap="rectangular">
        <Button color="inherit" component={RouterLink} to="/chat">Chat</Button>
      </Badge>
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