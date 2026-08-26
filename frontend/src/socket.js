import { io } from 'socket.io-client';

const socket = io(process.env.REACT_APP_API_URL, {
  autoConnect: false,
  auth: (cb) => cb({ token: localStorage.getItem('tapis_token') }),
});

export const connectSocket = () => {
  if (!socket.connected) {
    socket.connect();
  }
};

export const disconnectSocket = () => {
  if (socket.connected) {
    socket.disconnect();
  }
};

export default socket;
