# # Step 1: Use an official Node.js image as the base image
# FROM node:16-alpine

# # Step 2: Set the working directory inside the container
# WORKDIR /app

# # Step 3: Copy package.json and package-lock.json for dependencies installation
# COPY package*.json ./

# # Step 4: Install the dependencies
# RUN npm install

# # Step 5: Copy the rest of the app files
# COPY . .

# # Step 6: Build the React app
# RUN npm run build

# # Step 7: Install a web server to serve the build
# RUN npm install -g serve

# # Step 8: Expose the port that the app will run on
# EXPOSE 3000

# # Step 9: Start the web server to serve the app
# CMD ["serve", "-s", "build", "-l", "3000"]


FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy the package.json and package-lock.json
COPY package*.json ./

# Install all dependencies (including development dependencies)
RUN npm install --legacy-peer-deps

# Copy the rest of the application code
COPY . .

EXPOSE 3000

CMD ["npm", "start"] 