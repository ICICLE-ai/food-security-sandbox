import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AppBar, Toolbar, Typography } from "@mui/material";
import { Home} from "./components";
import LoggedIn from './components/Navigation/LoggedIn';
import LoggedOut from './components/Navigation/LoggedOut';
import CollaborativeML from './components/CollaborativeML/CollaborativeML';
import Chat from "./components/Chat/Chat";
import './App.css';
import axios from 'axios';
import icicleLogo from "./assets/icicleLogo.png"
import Loader from './components/Loader/Loader';

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
          window.location.href = 'http://localhost:5003/api/auth/login';
        });
      }
  })



  const handleLogout = () => {
    localStorage.removeItem('tapis_token');
    localStorage.removeItem('tapis_username');
    setIsAuthenticated(false);
  };

  return (
    
    <Router>
      <AppBar position="static" sx={{ bgcolor: "green" }}> 
        <Toolbar>
          <img src={icicleLogo} alt="ICICLE Logo" style={{
                      'width': '50px',   
                      'height': '50px',
                      'backgroundColor' : 'white',
                      'borderRadius' : '25px',
                      'marginRight' : '10px'
                    } }
            />
          {/* <BrightnessHighIcon sx={{ display: { xs: 'none', md: 'flex' }, mr: 1 }} /> */}
          <Typography variant="h6" style={{ flexGrow: 1 }}>
            Food System Sandbox
          </Typography>
          {isAuthenticated ? (
            <LoggedIn onLogout={handleLogout} />
          ) : (
            <LoggedOut />
          )}
        </Toolbar>
      </AppBar>

      <Routes>
        <Route path="/" element={isAuthenticated ? <Home /> : <Loader></Loader>} />
        <Route path="/training" element={isAuthenticated ? <CollaborativeML /> : <Navigate to="/" />} />
        <Route path="/chat" element={isAuthenticated ? <Chat /> : <Navigate to="/" />} />
      </Routes>
    </Router>
    
  );
}

export default App;
