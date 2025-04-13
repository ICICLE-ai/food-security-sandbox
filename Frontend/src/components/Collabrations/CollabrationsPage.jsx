import React, { useState, useEffect } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import {
  Container, Typography, Button, Box, Snackbar, Paper, Alert, Card, CardContent, Avatar, Tabs, Tab, Tooltip, Grid,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import UndoIcon from '@mui/icons-material/Undo';
import CancelPresentationIcon from '@mui/icons-material/CancelPresentation';
import axios from 'axios';
import URL from '../../config';

const getToken = () => {
  return localStorage.getItem('token');
};

const TabPanel = (props) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`invitation-tabpanel-${index}`}
      aria-labelledby={`invitation-tab-${index}`}
      {...other}
    >
      {value === index && <Box>{children}</Box>}
    </div>
  );
};

const a11yProps = (index) => {
  return {
    id: `invitation-tab-${index}`,
    'aria-controls': `invitation-tabpanel-${index}`,
  };
};

const UserInvitation = ({
  invitation,
  onAccept,
  onReject,
  onWithdraw,
  onRevoke,
  currentUserId,
  type, // 'pending' or 'sent'
}) => {
  const collaborationUuid = invitation.collab_uuid;

  const handleAccept = () => {
    onAccept(invitation._id, invitation.receiver_id, invitation.sender_id, collaborationUuid);
  };

  const handleReject = () => {
    onReject(invitation._id, invitation.receiver_id, invitation.sender_id, collaborationUuid);
  };

  const handleWithdraw = () => {
    onWithdraw(invitation._id, invitation.receiver_id, invitation.sender_id, collaborationUuid);
  };

  const handleRevoke = () => {
    onRevoke(invitation._id, invitation.receiver_id, invitation.sender_id, collaborationUuid);
  };

  const displayUser = () => {
    if (type === 'pending') {
      return invitation.sender_name;
    } else if (type === 'sent') {
      return invitation.receiver_name;
    }
    return '';
  };

  const displayCollabName = () => {
    return invitation.collab_name;
  };

  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Grid container alignItems="center">
          {/* Information Section */}
          <Grid item xs={12} sm={8}>
            <Box display="flex" alignItems="center">
              <Avatar sx={{ mr: 2 }}>
                {displayUser().charAt(0).toUpperCase()}
              </Avatar>
              <Box>
                <Typography
                  variant="h6"
                  component={RouterLink}
                  to={`/collaboration/${collaborationUuid}`}
                  sx={{ textDecoration: 'none', color: 'primary.main' }}
                >
                  {displayUser()} • {displayCollabName()}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  {type === 'pending' ? 'Incoming Invitation' : 'Sent Invitation'}
                </Typography>
              </Box>
            </Box>
          </Grid>

          {/* Actions Section */}
          <Grid item xs={12} sm={4} sx={{ textAlign: { xs: 'left', sm: 'right' }, mt: { xs: 2, sm: 0 } }}>
            {type === 'pending' && (
              <Box>
                <Tooltip title="Accept Invitation">
                  <Button
                    variant="contained"
                    color="primary"
                    size="small"
                    startIcon={<CheckCircleIcon />}
                    onClick={handleAccept}
                    sx={{ mr: 1 }}
                  >
                    Accept
                  </Button>
                </Tooltip>
                <Tooltip title="Reject Invitation">
                  <Button
                    variant="outlined"
                    color="secondary"
                    size="small"
                    startIcon={<CancelIcon />}
                    onClick={handleReject}
                  >
                    Reject
                  </Button>
                </Tooltip>
              </Box>
            )}
            {type === 'sent' && (
              invitation.status === 'accepted' ? (
                <Tooltip title="Revoke Invitation">
                  <Button
                    variant="contained"
                    color="secondary"
                    size="small"
                    startIcon={<CancelPresentationIcon />}
                    onClick={handleRevoke}
                  >
                    Revoke
                  </Button>
                </Tooltip>
              ) : (
                <Tooltip title="Withdraw Invitation">
                  <Button
                    variant="outlined"
                    color="primary"
                    size="small"
                    startIcon={<UndoIcon />}
                    onClick={handleWithdraw}
                  >
                    Withdraw
                  </Button>
                </Tooltip>
              )
            )}
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

const CollaborationsPage = () => {
  const [pendingInvitations, setPendingInvitations] = useState([]);
  const [sentInvitations, setSentInvitations] = useState([]);
  const [acceptedInvitations, setAcceptedInvitations] = useState([]);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState('');
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [currentUserId, setCurrentUserId] = useState('');
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    const fetchInvitations = async () => {
      const token = getToken();

      try {
        const response = await axios.get(`${URL}/api/invitations`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        console.log('invitation data: ',response.data);

        const { user_id, invitations } = response.data;

        const pending = invitations.filter(
          (invitation) => invitation.status === 'pending' && invitation.receiver_id === user_id
        );
        const sent = invitations.filter(
          (invitation) => invitation.sender_id === user_id
        );
        const accepted = invitations.filter(
          (invitation) => invitation.status === 'accepted' && invitation.sender_id === user_id || invitation.receiver_id === user_id
        );

        setCurrentUserId(user_id);
        setPendingInvitations(pending);
        setSentInvitations(sent);
        setAcceptedInvitations(accepted);
      } catch (error) {
        console.error('Error fetching invitations:', error);
        setError('Failed to fetch invitations');
      }
    };

    fetchInvitations();
  }, []);

  // Handlers for various invitation actions
  const handleWithdraw = async (invitationId, receiverId, senderId, collaborationUuid) => {
    const token = getToken();
    try {
      const response = await axios.post(
        `${URL}/api/withdrawinvitation`,
        {
          uuid: collaborationUuid,
          receiver_id: receiverId,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      if (response.status === 200) {
        setMessage('Invitation Withdrawn');
        setOpenSnackbar(true);
        setSentInvitations((prev) => prev.filter((invitation) => invitation._id !== invitationId));
      }
    } catch (error) {
      console.error('Error withdrawing invitation:', error.response?.data?.message || error.message);
      setMessage('Failed to withdraw invitation');
      setOpenSnackbar(true);
    }
  };

  const handleAccept = async (invitationId, receiverId, senderId, collaborationUuid) => {
    const token = getToken();
    try {
      const response = await axios.post(
        `${URL}/api/acceptinvitation`,
        {
          uuid: collaborationUuid,
          receiver_id: receiverId,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      if (response.status === 200) {
        setMessage('Invitation Accepted');
        setOpenSnackbar(true);
        setPendingInvitations((prev) => prev.filter((invitation) => invitation._id !== invitationId));
        // Optionally, add to acceptedInvitations if needed
        const acceptedInvitation = pendingInvitations.find((inv) => inv._id === invitationId);
        setAcceptedInvitations((prev) => [...prev, { ...acceptedInvitation, status: 'accepted' }]);
      }
    } catch (error) {
      console.error('Error accepting invitation:', error);
      setMessage('Failed to accept invitation');
      setOpenSnackbar(true);
    }
  };

  const handleReject = async (invitationId, receiverId, senderId, collaborationUuid) => {
    const token = getToken();
    try {
      const response = await axios.post(
        `${URL}/api/rejectinvitation`,
        {
          uuid: collaborationUuid,
          receiver_id: receiverId,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.status === 200) {
        setMessage('Invitation Rejected');
        setOpenSnackbar(true);
        setPendingInvitations((prev) => prev.filter((invitation) => invitation._id !== invitationId));
      }
    } catch (error) {
      console.error('Error rejecting invitation:', error);
      setMessage('Failed to reject invitation');
      setOpenSnackbar(true);
    }
  };

  const handleRevoke = async (invitationId, receiverId, senderId, collaborationUuid) => {
    const token = getToken();
    try {
      const response = await axios.post(
        `${URL}/api/revoke_invitation`,
        {
          uuid: collaborationUuid,
          receiver_id: receiverId,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.status === 200) {
        setMessage('Invitation Revoked');
        setOpenSnackbar(true);
        setSentInvitations((prev) => prev.filter((invitation) => invitation._id !== invitationId));
        setAcceptedInvitations((prev) => prev.filter((invitation) => invitation._id !== invitationId));
      }
    } catch (error) {
      console.error('Error revoking invitation:', error);
      setMessage('Failed to revoke invitation');
      setOpenSnackbar(true);
    }
  };

  const handleCloseSnackbar = () => {
    setOpenSnackbar(false);
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  if (error) {
    return (
      <Container component="div" maxWidth="md" sx={{ marginTop: 4 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );
  }

  return (
    <Container component="div" maxWidth="md" sx={{ marginTop: 4, marginBottom: 4 }}>
      <Typography variant="h4" align="center" gutterBottom sx={{ fontWeight: 300 }}>
        Collaborations
      </Typography>

      <Paper sx={{ mb: 4, border: '1px solid #dddddd', boxShadow: 'none' }}>
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          variant="fullWidth"
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label={`Pending (${pendingInvitations.length})`} {...a11yProps(0)} />
          <Tab label={`Accepted (${acceptedInvitations.length})`} {...a11yProps(1)} />
          <Tab label={`Sent (${sentInvitations.length})`} {...a11yProps(2)} />
        </Tabs>
      </Paper>

      <TabPanel value={tabValue} index={0}>
        {pendingInvitations.length === 0 ? (
          <Typography variant="body1">
            No pending invitations.{' '}
            <Box component="span">
              <RouterLink
                to="/start-collaboration"
                style={{ textDecoration: 'none', color: '#1976d2', fontWeight: 'bold' }}
              >
                Start new collaboration
              </RouterLink>
            </Box>{' '}
            or{' '}
            <Box component="span">
              <RouterLink
                to="#accepted-tab"
                onClick={() => setTabValue(1)}
                style={{ textDecoration: 'none', color: '#1976d2', fontWeight: 'bold' }}
              >
                check existing
              </RouterLink>
            </Box>.
          </Typography>
        ) : (
          pendingInvitations.map((invitation) => (
            <UserInvitation
              key={invitation._id}
              invitation={invitation}
              onAccept={handleAccept}
              onReject={handleReject}
              onWithdraw={handleWithdraw}
              onRevoke={handleRevoke}
              currentUserId={currentUserId}
              type="pending"
            />
          ))
        )}
      </TabPanel>


      <TabPanel value={tabValue} index={1}>
        {acceptedInvitations.length === 0 ? (
          <Typography variant="body1">No accepted invitations.</Typography>
        ) : (
          acceptedInvitations.map((invitation) => (
            <Card variant="outlined" sx={{ mb: 2 }} key={invitation._id}>
              <CardContent>
                <Grid container alignItems="center">
                  {/* Information Section */}
                  <Grid item xs={12} sm={8}>
                    <Box display="flex" alignItems="center">
                      <Avatar sx={{ mr: 2 }}>
                        {invitation.sender_name.charAt(0).toUpperCase()}
                      </Avatar>
                      
                      <Box>
                      <Tooltip title={`${invitation.collab_name} initiated by ${invitation.sender_name}`} placement='right'>
                        <Typography
                          variant="h6"
                          component={RouterLink}
                          to={`/collaboration/${invitation.collab_uuid}`}
                          sx={{ textDecoration: 'none', color: 'primary.main' }}
                        >
                          {invitation.collab_name}
                        </Typography>
                        </Tooltip>
                        <Typography variant="body2" color="textSecondary">
                          Collaboration between {invitation.sender_name} & {invitation.receiver_name}
                        </Typography>
                      </Box>
                      
                    </Box>
                  </Grid>

                  {/* Actions Section */}
                  <Grid item xs={12} sm={4} sx={{ textAlign: { xs: 'left', sm: 'right' }, mt: { xs: 2, sm: 0 } }}>
                    {invitation.sender_id === currentUserId && (
                      <Button
                        variant="contained"
                        color="primary"
                        size="small"
                        onClick={() => navigate('/session')}
                      >
                        Create a Session
                      </Button>
                    )}
                    {invitation.receiver_id === currentUserId && (
                      <Tooltip title="Revoke Invitation">
                        <Button
                          variant="outlined"
                          color="secondary"
                          size="small"
                          startIcon={<CancelPresentationIcon />}
                          onClick={() =>
                            handleRevoke(invitation._id, invitation.receiver_id, invitation.sender_id, invitation.collab_uuid)
                          }
                        >
                          Revoke
                        </Button>
                      </Tooltip>
                    )}
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          ))
        )}
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        {sentInvitations.length === 0 ? (
          <Typography variant="body1">
          No invitations sent.{' '}
          <Box component="span">
            <RouterLink
              to="/start-collaboration"
              style={{ textDecoration: 'none', color: '#1976d2', fontWeight: 'bold' }}
            >
              Start new collaboration
            </RouterLink>
          </Box>{' '}
          ?
        </Typography>
        ) : (
          sentInvitations.map((invitation) => (
            <UserInvitation
              key={invitation._id}
              invitation={invitation}
              onAccept={handleAccept}
              onReject={handleReject}
              onWithdraw={handleWithdraw}
              onRevoke={handleRevoke}
              currentUserId={currentUserId}
              type="sent"
            />
          ))
        )}
      </TabPanel>

      <Snackbar open={openSnackbar} autoHideDuration={3000} onClose={handleCloseSnackbar}>
        <Alert onClose={handleCloseSnackbar} severity="success" sx={{ width: '100%' }}>
          {message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default CollaborationsPage;
