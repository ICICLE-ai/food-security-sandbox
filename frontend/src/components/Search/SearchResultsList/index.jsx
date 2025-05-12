// import React, { useState, useEffect } from 'react';
// import { List } from '@mui/material';
// import CollaboratorItem from "../CollaboratorItem"; 

// const SearchResultList = () => {
//   const [collaborators, setCollaborators] = useState([]);
//   const [isLoading, setIsLoading] = useState(true);

//   useEffect(() => {
//     const fetchData = async () => {
//       try {
//         const response = await fetch('https://ubiquitous-space-potato-q77r667x74qx29vvr-5000.app.github.dev/api/collaborations');
//         const data = await response.json();
//         setCollaborators(data);
//         setIsLoading(false);
//       } catch (error) {
//         console.error('Error fetching data:', error);
//         setIsLoading(false);
//       }
//     };

//     fetchData();

//     // Cleanup function to abort fetch if component unmounts
//     return () => {
//       // Abort the fetch if it's still ongoing
//     };
//   }, []); // Empty dependency array to run effect only once when component mounts

//   if (isLoading) {
//     return <p>Loading...</p>;
//   }

//   if (!collaborators || collaborators.length === 0) {
//     return <p>No collaborators found.</p>;
//   }

//   return (
//     <List>
//       {collaborators.map((collaborator) => (
//         <CollaboratorItem key={collaborator.id} collaborator={collaborator} />
//       ))}
//     </List>
//   );
// };

// export default SearchResultList;
