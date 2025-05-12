// import React, { useState } from 'react';
// import { TextField, Button, Grid } from '@mui/material';
// import SearchIcon from '@mui/icons-material/Search';

// const SearchInput = ({ onSearch }) => {
//   const [phenotypes, setPhenotypes] = useState('');
//   const [minSamples, setMinSamples] = useState('');

//   const handleSearch = () => {
//     // Prepare search criteria
//     const searchCriteria = {
//       phenotypes,
//       minSamples,
//     };

//     // Trigger search
//     onSearch(searchCriteria);
//   };

//   return (
//     <Grid container spacing={2} alignItems="center" justifyContent="center">
//       <Grid item xs={12}>
//         <TextField
//           fullWidth
//           label="Phenotype(s)"
//           variant="outlined"
//           margin="normal"
//           value={phenotypes}
//           onChange={(e) => setPhenotypes(e.target.value)}
//         />
//       </Grid>
//       <Grid item xs={12}>
//         <TextField
//           fullWidth
//           label="Minimum # of Samples"
//           variant="outlined"
//           margin="normal"
//           value={minSamples}
//           onChange={(e) => setMinSamples(e.target.value)}
//         />
//       </Grid>
//       <Grid item xs={12}>
//         <Button
//           variant="contained"
//           color="primary"
//           startIcon={<SearchIcon />}
//           fullWidth
//           onClick={handleSearch}
//         >
//           Search For Collaborator
//         </Button>
//       </Grid>
//     </Grid>
//   );
// };

// export default SearchInput;
