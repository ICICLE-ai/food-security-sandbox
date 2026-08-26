import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2E3A54',
      dark: '#1E2740',
      light: '#E3E7EE',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#B8863B',
      dark: '#8F6526',
      contrastText: '#ffffff',
    },
    background: {
      default: '#FAF8F4',
      paper: '#FFFFFF',
    },
    text: {
      primary: '#22242B',
      secondary: '#6B6875',
    },
    error: {
      main: '#C0392B',
    },
    divider: '#E7E2D8',
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: { fontWeight: 600, letterSpacing: '-0.02em' },
    h5: { fontWeight: 600, letterSpacing: '-0.01em' },
    h6: { fontWeight: 600 },
  },
  shape: {
    borderRadius: 10,
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          elevation: 0,
          boxShadow: 'none',
        },
      },
      defaultProps: {
        elevation: 0,
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          transition: 'transform 0.16s ease, box-shadow 0.16s ease, background-color 0.16s ease',
          '&:hover': {
            transform: 'translateY(-2px)',
          },
        },
        containedPrimary: {
          '&:hover': {
            boxShadow: '0 6px 14px rgba(46, 58, 84, 0.35)',
          },
        },
        containedSecondary: {
          '&:hover': {
            boxShadow: '0 6px 14px rgba(184, 134, 59, 0.35)',
          },
        },
      },
    },
  },
});

export default theme;
