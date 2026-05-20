#!/usr/bin/env bash
# Start JupyterLab + standalone Jupyter MCP server for Cursor.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv/bin"
JUPYTER_TOKEN="${JUPYTER_TOKEN:-odev-jupyter-2026}"
MCP_TOKEN="${MCP_TOKEN:-odev-mcp-2026}"

cd "$ROOT"

if ! ss -tln | grep -q ':8888 '; then
  echo "Starting JupyterLab on :8888 ..."
  "$VENV/jupyter" lab \
    --port 8888 \
    --IdentityProvider.token="$JUPYTER_TOKEN" \
    --ip 127.0.0.1 \
    --no-browser &
fi

if ! ss -tln | grep -q ':4040 '; then
  echo "Starting Jupyter MCP server on :4040 ..."
  "$VENV/jupyter-mcp-server" start \
    --transport streamable-http \
    --jupyter-url http://127.0.0.1:8888 \
    --jupyter-token "$JUPYTER_TOKEN" \
    --mcp-token "$MCP_TOKEN" \
    --port 4040 \
    --document-id notebook.ipynb &
fi

echo "JupyterLab: http://127.0.0.1:8888/lab?token=$JUPYTER_TOKEN"
echo "MCP endpoint: http://127.0.0.1:4040/mcp (Bearer $MCP_TOKEN)"
