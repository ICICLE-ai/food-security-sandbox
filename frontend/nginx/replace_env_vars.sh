#!/bin/bash

set -euo pipefail

: "${BACKEND_URL:?BACKEND_URL must be set, e.g. http://app-backend:5001}"
: "${SANDBOX_URL:=}"

echo "[entrypoint] Using BACKEND_URL=${BACKEND_URL}"
if [ -n "${SANDBOX_URL}" ]; then
  echo "[entrypoint] Using SANDBOX_URL=${SANDBOX_URL}"
else
  echo "[entrypoint] SANDBOX_URL not set (sandbox routes will still be templated if present)."
fi

# Replace placeholders in nginx config template with actual environment variables
envsubst '${BACKEND_URL} ${SANDBOX_URL}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf
echo "[entrypoint] Rendered Nginx config to /etc/nginx/nginx.conf"
# Inside the nginx container
cat /etc/nginx/nginx.conf | grep proxy_pass

# Continue with nginx execution
exec "$@"