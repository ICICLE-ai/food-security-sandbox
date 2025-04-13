// import React, { useState } from 'react';
// import { ListItem, ListItemText, Button, Grid, Snackbar } from '@mui/material';
// import CollaborationRequestButton from '../CollaborationRequestButton';

// const CollaboratorItem = ({ collaborator }) => {
//   const { id, email } = collaborator;
  
//   const [isRequestSent, setIsRequestSent] = useState(false); // State to manage visibility of the message

//   const handleSendRequest = () => {
//     // Implement the functionality to send a collaboration request
//     console.log(`Sending collaboration request to ${email}`);
    
//     // Update state to show the message
//     setIsRequestSent(true);
//   };

//   const handleCloseSnackbar = () => {
//     // Close the Snackbar
//     setIsRequestSent(false);
//   };
// // Need to change to specific metadata of the user profile
//   return (
//     <ListItem key={_id}>
//       <Grid container spacing={2} alignItems="center">
//         <Grid item xs={12} sm={6}>
//           <ListItemText 
//             primary={id} 
//             secondary={`Id: ${skills.join(', ')}`}
//           />
//         </Grid>
//         <Grid item xs={12} sm={4}>
//           <ListItemText 
//             primary={`email: ${metadata}`} 
//           />
//         </Grid>
//         <Grid item xs={12} sm={2}>
//           <CollaborationRequestButton onClick={handleSendRequest} />
//         </Grid>
//       </Grid>
//       <Snackbar 
//         open={isRequestSent} 
//         autoHideDuration={3000} 
//         onClose={handleCloseSnackbar}
//         message="Request sent successfully" 
//       />
//     </ListItem>
//   );
// };

// export default CollaboratorItem;
