#!/usr/bin/env bash
# Build the plugin and copy it into a vault for local testing.
# Usage: scripts/deploy.sh "/path/to/vault"   (or set OBSIDIAN_VAULT)
set -euo pipefail

VAULT="${1:-${OBSIDIAN_VAULT:-}}"
if [[ -z "$VAULT" ]]; then
  echo "Usage: $0 \"/path/to/vault\"  (or set OBSIDIAN_VAULT)" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$VAULT/.obsidian/plugins/meeting-transcriber"

cd "$ROOT"
npm run build
mkdir -p "$DEST"
cp main.js manifest.json styles.css "$DEST"/
echo "Deployed to $DEST — reload Obsidian (or toggle the plugin) to pick it up."
