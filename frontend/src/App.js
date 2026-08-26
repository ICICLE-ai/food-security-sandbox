import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AppBar, Toolbar, Typography, Box } from "@mui/material";
import { Home} from "./components";
import LoggedIn from './components/Navigation/LoggedIn';
import LoggedOut from './components/Navigation/LoggedOut';
import CollaborativeML from './components/CollaborativeML/CollaborativeML';
import Chat from "./components/Chat/Chat";
import DataSharing from "./components/DataSharing/DataSharing";
import './App.css';
import axios from 'axios';
import AppLogo from "./assets/AppLogo";
import icicleLogo from "./assets/icicleLogo.png"
import taccLogo from "./tacc-black.png"
import Loader from './components/Loader/Loader';
import { connectSocket, disconnectSocket } from './socket';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);


  useEffect(()=>{
    const params = new URLSearchParams(window.location.search);
      if (params.get('tapis_token')){
        const tapis_token = params.get('tapis_token');
        const username = params.get('username');
        localStorage.setItem('tapis_token', tapis_token);
        localStorage.setItem('tapis_username', username);
        setIsAuthenticated(true);
      }
      else{
        if(localStorage.getItem('tapis_token') == null){
          localStorage.removeItem('tapis_token');
          localStorage.removeItem('tapis_username');
          window.location.href = `${process.env.REACT_APP_API_URL}/api/auth/login`;
        }else{
          axios.get(`${process.env.REACT_APP_API_URL}/api/auth/verify`, {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('tapis_token')}`
            }
          }).then(response => {
            console.log(response)
            setIsAuthenticated(true);
          })
          .catch(error => {
            console.error(error);
            localStorage.removeItem('tapis_token');
            localStorage.removeItem('tapis_username');
            window.location.href = `${process.env.REACT_APP_API_URL}/api/auth/login`;
          });
        }
      }
  })



  useEffect(() => {
    if (isAuthenticated) {
      connectSocket();
    } else {
      disconnectSocket();
    }
  }, [isAuthenticated]);

  const handleLogout = () => {
    localStorage.removeItem('tapis_token');
    localStorage.removeItem('tapis_username');
    disconnectSocket();
    setIsAuthenticated(false);
  };

  return (
    
    <Router>
      <AppBar position="static">
        <Toolbar>
          <AppLogo size={36} color="#ffffff" className="app-bar-logo" />
          <Typography variant="h6" sx={{ flexGrow: 1, ml: 1.5, fontWeight: 600 }}>
            Collaborative Research Sandbox
          </Typography>
          {isAuthenticated ? (
            <LoggedIn onLogout={handleLogout} />
          ) : (
            <LoggedOut />
          )}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              ml: 2,
              pl: 2,
              borderLeft: '1px solid rgba(255,255,255,0.3)',
            }}
            title="Powered by ICICLE and TACC"
          >
            <img src={icicleLogo} alt="ICICLE" style={{ height: 22, borderRadius: '50%', backgroundColor: 'white', padding: 1 }} />
            <img src={taccLogo} alt="TACC" style={{ height: 16, backgroundColor: 'white', borderRadius: 3, padding: '2px 4px' }} />
          </Box>
        </Toolbar>
      </AppBar>

      <Routes>
        <Route path="/" element={isAuthenticated ? <Home /> : <Loader></Loader>} />
        <Route path="/training" element={isAuthenticated ? <CollaborativeML /> : <Navigate to="/" />} />
        <Route path="/chat" element={isAuthenticated ? <Chat /> : <Navigate to="/" />} />
        <Route path="/dataSharing" element={isAuthenticated ? <DataSharing /> : <Navigate to="/" />} />
      </Routes>
    </Router>
    
  );
}

export default App;
