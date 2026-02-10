#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

curl -s -X POST "${BASE_URL%/}/documents/index" \
  -H 'Content-Type: application/json' \
  -H 'X-User: amy_analyst' \
  -d '{
    "visibility": "public",
    "paths": [
      "data/docs/retention_policy.txt",
      "data/docs/payroll_sop.txt",
      "data/docs/security_controls.txt"
    ]
  }'
