import React, { useState, useEffect, useCallback } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AppBar, Toolbar, Typography } from "@mui/material";
import { Home, Upload, Search, Session, NewUser, Login } from "./components";
import BrightnessHighIcon from '@mui/icons-material/BrightnessHigh';
import LoggedIn from './components/Navigation/LoggedIn';
import LoggedOut from './components/Navigation/LoggedOut';
import ForgotUsername from './components/Forgot/ForgotUsername'; 
import ForgotPassword from './components/Forgot/ForgotPassword'; 
import SessionsResults from "./components/Session/SessionsResults";
import Profile from "./components/Profile/Profile";
import CollaborativeML from './components/CollaborativeML/CollaborativeML';
import Chat from "./components/Chat/Chat";
import './App.css';
import axios from 'axios';
import icicleLogo from "./assets/icicleLogo.png"

function App() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);


  useEffect(()=>{
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
      });
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
        <Route path="/" element={isAuthenticated ? <Home /> : <Login setIsAuthenticated={setIsAuthenticated}  />} />
        <Route path="/register" element={<NewUser />} />
        <Route path="/training" element={isAuthenticated ? <CollaborativeML /> : <Navigate to="/" />} />
        <Route path="/chat" element={isAuthenticated ? <Chat /> : <Navigate to="/" />} />
        <Route path="/search" element={isAuthenticated ? <Search /> : <Navigate to="/" />} />
        <Route path="/session" element={isAuthenticated ? <Session /> : <Navigate to="/" />} />
        <Route path="/profile" element={isAuthenticated ? <Profile /> : <Navigate to="/" />} />
        <Route path="/session/viewresults" element={ <SessionsResults />} />
        <Route path="/forgot/username" element={<ForgotUsername />} />
        <Route path="/forgot/password" element={<ForgotPassword />} />
      </Routes>
    </Router>
    
  );
}

export default App;
