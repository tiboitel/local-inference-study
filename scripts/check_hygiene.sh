#!/usr/bin/env bash
# Release gate: no secrets and no absolute host paths in tracked files.
#
#   bash scripts/check_hygiene.sh   # exit 0 == clean
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0

scan() {
  local label="$1" pattern="$2"
  local matches
  matches=$(git grep -InE "$pattern" -- . ':(exclude)scripts/check_hygiene.sh' || true)
  if [ -n "$matches" ]; then
    echo "HYGIENE FAIL ($label):"
    echo "$matches"
    fail=1
  fi
}

scan "secrets" 'api[-_]?key[[:space:]]*[:=]|BEGIN [A-Z ]*PRIVATE KEY|password[[:space:]]*[:=]'
scan "abs host paths" '(^|[^[:alnum:]])/(home|Users)/[A-Za-z0-9._-]+'

if [ "$fail" -ne 0 ]; then
  exit 1
fi
echo "hygiene OK"
