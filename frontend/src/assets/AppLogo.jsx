import React from 'react';

// Four outer "participant" nodes connecting into one shared center —
// stands in for any collaborative-research domain, not a specific one.
const AppLogo = ({ size = 32, color = 'currentColor', className = '' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 100 100"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={`app-logo ${className}`}
    role="img"
    aria-label="Collaborative Research Sandbox"
  >
    <g stroke={color} strokeWidth="4" strokeLinecap="round" opacity="0.55">
      <path className="logo-line logo-line-1" d="M24 24 Q40 40 50 50" />
      <path className="logo-line logo-line-2" d="M76 24 Q60 40 50 50" />
      <path className="logo-line logo-line-3" d="M24 76 Q40 60 50 50" />
      <path className="logo-line logo-line-4" d="M76 76 Q60 60 50 50" />
    </g>
    <circle className="logo-node logo-node-1" cx="24" cy="24" r="10" fill={color} />
    <circle className="logo-node logo-node-2" cx="76" cy="24" r="10" fill={color} />
    <circle className="logo-node logo-node-3" cx="24" cy="76" r="10" fill={color} />
    <circle className="logo-node logo-node-4" cx="76" cy="76" r="10" fill={color} />
    <circle className="logo-center" cx="50" cy="50" r="14" fill={color} />
  </svg>
);

export default AppLogo;
