// frontend/src/components/Loader/Loader.jsx
import React from 'react';
import './Loader.css';
import logo from '../../assets/icicleLogo.png'; // Change to your logo file name

const Loader = () => (
    <div className="loader-background">
      <div className="loader-logo-wrapper">
        <img src={logo} alt="Logo" className="loader-logo" />
      </div>
    </div>
  );
  
  export default Loader;