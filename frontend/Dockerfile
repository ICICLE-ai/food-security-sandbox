FROM node:lts-slim AS build-stage
WORKDIR /app
RUN mkdir src public
COPY package*.json ./
RUN npm install
COPY ./public ./public/
COPY ./src ./src/
RUN npm run build

FROM nginx AS production-stage
RUN mkdir /app
COPY --from=build-stage /app/build /app
COPY ./nginx/nginx.conf /etc/nginx/nginx.conf.template
COPY ./nginx/replace_env_vars.sh /docker-entrypoint.d/10-replace-env-vars.sh
RUN chmod +x /docker-entrypoint.d/10-replace-env-vars.sh

# Ensure nginx starts properly
CMD ["nginx", "-g", "daemon off;"]
