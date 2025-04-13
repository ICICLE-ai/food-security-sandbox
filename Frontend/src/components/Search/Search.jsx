// import React, { useState, useEffect } from 'react';
// import { Container, Grid, TextField, Button, Typography, Card, CardContent, CardActions, Checkbox, FormControlLabel } from '@mui/material';
// import SearchIcon from '@mui/icons-material/Search';
// import axios from 'axios';
// import URL from '../../config';

// function SearchPage({ onUserSelect }) {
//   const [phenotype, setPhenotype] = useState('');
//   const [minSamples, setMinSamples] = useState('');
//   const [searchResults, setSearchResults] = useState([]);
//   const [selectedUsers, setSelectedUsers] = useState({});

//   const getToken = () => {
//     return localStorage.getItem('token');
//   };

//   const handleSearch = async () => {
//     try {
//       const token = getToken();
//       if (!token) {
//         return;
//       }

//       const config = {
//         headers: {
//           Authorization: `Bearer ${token}`,
//         },
//         params: {
//           phenotype: phenotype,
//           minSamples: minSamples,
//         },
//       };

//       const response = await axios.get(`${URL}/api/invite/users`, config);
//       const results = response.data;
//       setSearchResults(results);
//       setSelectedUsers({}); // Reset selected users
//       console.log(response.data);
//     } catch (error) {
//       console.error('Error fetching data:', error);
//     }
//   };

//   const handleUserSelect = (userId) => {
//     setSelectedUsers((prev) => ({
//       ...prev,
//       [userId]: !prev[userId], // Toggle selection
//     }));
//   };

//   // Pass the selected users to the parent only when selectedUsers or searchResults change.
//   useEffect(() => {
//     if (searchResults.length > 0) {
//       const selectedUsersList = searchResults.filter(user => selectedUsers[user._id]);
//       onUserSelect(selectedUsersList);
//     }
//   }, [selectedUsers, searchResults, onUserSelect]);

//   return (
//     <Container component="div" maxWidth="lg" sx={{ borderRadius: 2 }}>
//       <Grid container spacing={3} alignItems="center" justifyContent="center">
//         <Grid item xs={12}>
//           {/* <Typography variant="h4" align="center" gutterBottom sx={{ fontWeight: 'light' }}>
//             Search Collaborators
//           </Typography> */}
//         </Grid>
//         <Grid item xs={4} sm={4}>
//           <TextField
//             fullWidth
//             label="Phenotype(s)"
//             variant="outlined"
//             value={phenotype}
//             onChange={(e) => setPhenotype(e.target.value)}
//             sx={{ marginBottom: 2 }}
//           />
//         </Grid>
//         <Grid item xs={4} sm={4}>
//           <TextField
//             fullWidth
//             label="Minimum # of Samples"
//             variant="outlined"
//             value={minSamples}
//             onChange={(e) => setMinSamples(e.target.value)}
//             sx={{ marginBottom: 2 }}
//           />
//         </Grid>
//         <Grid item xs={4} sm={4} md={4}>
//           <Button
//             variant="contained"
//             color="primary"
//             startIcon={<SearchIcon />}
//             onClick={handleSearch}
//             fullWidth
//             sx={{ padding: 2, fontSize: '1rem', marginBottom:2 }}
//           >
//             Search
//           </Button>
//         </Grid>
//         <Grid item xs={12}>
//           <Grid container spacing={3}>
//             {searchResults.length > 0 ? (
//               searchResults.map((result) => (
//                 <Grid item xs={12} sm={6} md={4} key={result._id}>
//                   <Card sx={{ borderRadius: 2, boxShadow: 'none', border: '1px solid #ddd' }}>
//                     <CardContent>
//                       <Typography variant="h6" component="div">
//                         {result.name}
//                       </Typography>
//                       <Typography variant="body2" color="textSecondary">
//                         Phenotype: {result.phenotype}
//                       </Typography>
//                       <Typography variant="body2" color="textSecondary">
//                         Number of samples: {result.number_of_samples}
//                       </Typography>
//                     </CardContent>
//                     <CardActions>
//                       <FormControlLabel
//                         control={
//                           <Checkbox
//                             checked={!!selectedUsers[result._id]}
//                             onChange={() => handleUserSelect(result._id)}
//                           />
//                         }
//                         label="Select for Collaboration"
//                       />
//                     </CardActions>
//                   </Card>
//                 </Grid>
//               ))
//             ) : (
//               <Typography variant="body1" color="textSecondary" sx={{ padding: 4 }}>
//                 No results found. Please refine your search.
//               </Typography>
//             )}
//           </Grid>
//         </Grid>
//       </Grid>
//     </Container>
//   );
// }

// export default SearchPage;


import React, { useState, useEffect } from 'react';
import { Container, Grid, TextField, Button, Typography, Card, CardContent, CardActions, Checkbox, FormControlLabel } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import axios from 'axios';
import URL from '../../config';

function SearchPage({ onUserSelect, resetTrigger }) {
  const [phenotype, setPhenotype] = useState('');
  const [minSamples, setMinSamples] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedDatasets, setSelectedDatasets] = useState({}); // Renamed to reflect dataset selection

  const getToken = () => {
    return localStorage.getItem('token');
  };

  const handleSearch = async () => {
    try {
      const token = getToken();
      if (!token) {
        return;
      }

      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        params: {
          phenotype: phenotype,
          minSamples: minSamples,
        },
      };

      const response = await axios.get(`${URL}/api/invite/users`, config);
      const results = response.data;
      setSearchResults(results);
      setSelectedDatasets({}); 
      console.log(response.data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const handleDatasetSelect = (datasetId) => { 
    setSelectedDatasets((prev) => ({
      ...prev,
      [datasetId]: !prev[datasetId], 
    }));
  };

  useEffect(() => {
    // Clear search results and selected datasets whenever resetTrigger changes
    setSearchResults([]);
    setSelectedDatasets({});
    setPhenotype(''); // Optionally reset search input fields
    setMinSamples('');
  }, [resetTrigger]);


  // Pass the selected datasets to the parent only when selectedDatasets or searchResults change.
  useEffect(() => {
    if (searchResults.length > 0) {
      // Filter selected results and map to ensure required fields
      const selectedDatasetsList = searchResults
        .filter(dataset => selectedDatasets[dataset.dataset_id])
        .map(dataset => ({
          _id: dataset._id,  // Ensure each entry has _id
          dataset_id: dataset.dataset_id,  // Ensure dataset_id is included
          phenotype: dataset.phenotype,  // Include phenotype if required
          // Include any other fields the parent component expects
        }));
  
      onUserSelect(selectedDatasetsList);
    }
  }, [selectedDatasets, searchResults, onUserSelect]);
  

  return (
    <Container component="div" maxWidth="lg" sx={{ borderRadius: 2 }}>
      <Grid container spacing={3} alignItems="center" justifyContent="center">
        <Grid item xs={12}>
          {/* <Typography variant="h4" align="center" gutterBottom sx={{ fontWeight: 'light' }}>
            Search Collaborators
          </Typography> */}
        </Grid>
        <Grid item xs={12} sm={4}>
          <TextField
            fullWidth
            label="Phenotype(s)"
            variant="outlined"
            value={phenotype}
            onChange={(e) => setPhenotype(e.target.value)}
            sx={{ marginBottom: 2 }}
          />
        </Grid>
        <Grid item xs={12} sm={4}>
          <TextField
            fullWidth
            label="Minimum # of Samples"
            variant="outlined"
            value={minSamples}
            onChange={(e) => setMinSamples(e.target.value)}
            sx={{ marginBottom: 2 }}
            type="number" // Added type for better UX
            inputProps={{ min: 0 }}
          />
        </Grid>
        <Grid item xs={12} sm={4} md={4}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<SearchIcon />}
            onClick={handleSearch}
            fullWidth
            sx={{ padding: 2, fontSize: '1rem', marginBottom: 2 }}
          >
            Search
          </Button>
        </Grid>
        <Grid item xs={12}>
          <Grid container spacing={3}>
            {searchResults.length > 0 ? (
              searchResults.map((result) => (
                <Grid item xs={12} sm={6} md={4} key={result.dataset_id}>
                  <Card sx={{ borderRadius: 2, boxShadow: 'none', border: '1px solid #ddd' }}>
                    <CardContent>
                      <Typography variant="h6" component="div">
                        {result.phenotype}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                      Number of Samples: {result.number_of_samples}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        By: {result.name}
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={!!selectedDatasets[result.dataset_id]}
                            onChange={() => handleDatasetSelect(result.dataset_id)}
                          />
                        }
                        label="Select for Collaboration"
                      />
                    </CardActions>
                  </Card>
                </Grid>
              ))
            ) : (
              <Typography variant="body1" color="textSecondary" sx={{ padding: 4 }}>
                No results found. Please refine your search.
              </Typography>
            )}
          </Grid>
        </Grid>
      </Grid>
    </Container>
  );
}

export default SearchPage;
