import React from 'react';
import { Grid, Typography, Paper, Button, TableContainer, Table, TableHead, TableRow, TableCell, TableBody } from '@mui/material';

const ResultsPage = () => {
  const resultsData = [
    {
      title: 'Sample Relatedness',
      tableData: [
        { experiment: 'Dataset Relatedness Coefficient', result: '0.8' }
      ],
      downloadUrl: '/session/viewresults', // link for download button
    },
    {
      title: 'GWAS Experiments',
      tableData: [
        { metric: 'χ² Test', result: 'Found 20 significant SNPs' },
        { metric: 'Odds Ratio Test', result: 'Found 25 significant SNPs' }
      ],
      downloadUrl: '/session/viewresults', // link for download button
    },
    // Add data for additional results here
  ];

  const boxStyle = {
    width: '100%', // Make the boxes take up full width
    height: 300, // Set a fixed height
    borderRadius: 12,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    alignItems: 'center',
    border: '1px solid #ccc',
    padding: '20px',
    // Remove default shadow
    boxShadow: 'none', 
  };

  return (
    <Grid container justifyContent="center" spacing={2} style={{ maxWidth: 1000, margin: '0 auto' }}>
      <Grid item xs={12} sx={{marginTop:2}}>
      <Typography variant="h4" align="center" gutterBottom sx={{ fontWeight: 'bold' }}>
          Results
        </Typography>
      </Grid>
      {resultsData.map((result, index) => (
        <Grid item xs={12} sm={6} key={index}>
          <Paper style={boxStyle}>
            <Typography variant="h6" align="center" style={{ marginBottom: '20px' }}>
              {result.title}
            </Typography>
            <TableContainer style={{ maxHeight: 180, overflow: 'auto' }}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    <TableCell>Result</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {result.tableData.map((row, rowIndex) => (
                    <TableRow key={rowIndex}>
                      <TableCell component="th" scope="row">
                        {row.experiment || row.metric}
                      </TableCell>
                      <TableCell>{row.result}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            <Button variant="outlined" fullWidth href={result.downloadUrl} style={{ marginTop: '20px' }}>
              Download Full Result
            </Button>
          </Paper>
        </Grid>
      ))}
    </Grid>
  );
};

export default ResultsPage;
