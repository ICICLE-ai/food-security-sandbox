import React from 'react';
import './Loader.css';
import AppLogo from '../../assets/AppLogo';

const Loader = () => (
    <div className="loader-background">
      <div className="loader-logo-wrapper">
        <AppLogo size={80} color="var(--color-primary)" className="loader-logo" />
      </div>
    </div>
  );

  export default Loader;
