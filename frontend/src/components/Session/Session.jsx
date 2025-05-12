import React, { useState } from 'react';
import { Container, Grid, FormGroup, FormControlLabel, Button, Typography, Checkbox, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle } from '@mui/material';

function CreateSession() {
  const [checked1, setChecked1] = useState(true);
  const [checked2, setChecked2] = useState(true);
  const [checked3, setChecked3] = useState(true);
  const [checked4, setChecked4] = useState(true);
  const [checked5, setChecked5] = useState(true);

  const [sessionCreated, setSessionCreated] = useState(false);

  const handleCreateSession = () => {
    setSessionCreated(true);
  };

  const handleCloseDialog = () => {
    setSessionCreated(false);
  };

  return (
    <Container component="div" maxWidth="md" sx={{ marginTop: 4, padding: 4, borderRadius: 2 }}>
      <Grid container spacing={3} alignItems="center" justifyContent="center">
        <Grid item xs={12}>
          <Typography variant="h4" align="center" gutterBottom sx={{ fontWeight: 'bold' }}>
            Select Experiments
          </Typography>
        </Grid>
        <Grid item xs={12} md={6}>
          <FormGroup>
            <FormControlLabel
              control={<Checkbox checked={checked1} onChange={(e) => setChecked1(e.target.checked)} />}
              label="Sample Relatedness"
            />
            <FormControlLabel
              control={<Checkbox checked={checked2} onChange={(e) => setChecked2(e.target.checked)} />}
              label="Gene Expression"
            />
            <FormControlLabel
              control={<Checkbox checked={checked3} onChange={(e) => setChecked3(e.target.checked)} />}
              label="Rare Variants"
            />
            <FormControlLabel
              control={<Checkbox checked={checked4} onChange={(e) => setChecked4(e.target.checked)} />}
              label="Variant Annotation"
            />
            <FormControlLabel
              control={<Checkbox checked={checked5} onChange={(e) => setChecked5(e.target.checked)} />}
              label="Population Genetics"
            />
          </FormGroup>
          <Button
            variant="contained"
            color="primary"
            fullWidth
            onClick={handleCreateSession}
            sx={{ marginTop: 2, padding: 2, fontSize: '1rem' }}
          >
            Perform Selected Experiments
          </Button>
          <Dialog
            open={sessionCreated}
            onClose={handleCloseDialog}
            aria-labelledby="alert-dialog-title"
            aria-describedby="alert-dialog-description"
          >
            <DialogTitle id="alert-dialog-title">{"Experiments Scheduled"}</DialogTitle>
            <DialogContent>
              <DialogContentText id="alert-dialog-description">
                The selected experiments have been successfully scheduled. Updates will be sent to you via email. Please check back later for experiment results.
              </DialogContentText>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseDialog} color="primary" autoFocus>
                Acknowledge
              </Button>
            </DialogActions>
          </Dialog>
        </Grid>
      </Grid>
    </Container>
  );
}

export default CreateSession;
