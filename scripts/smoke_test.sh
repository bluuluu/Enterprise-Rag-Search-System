#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
START_SERVER="${START_SERVER:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SERVER_PID=""

log() {
  printf '[smoke] %s\n' "$1"
}

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    log "Stopping local API process ($SERVER_PID)"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

assert_json_field() {
  local json="$1"
  local expr="$2"
  local expected="$3"

  "$PYTHON_BIN" -c 'import json,sys
obj=json.loads(sys.argv[1])
expr=sys.argv[2].split(".")
val=obj
for p in expr:
    if p:
        val=val[p]
print(val)
' "$json" "$expr" | {
    read -r actual
    if [[ "$actual" != "$expected" ]]; then
      echo "Expected $expr=$expected but got $actual" >&2
      exit 1
    fi
  }
}

if [[ "$START_SERVER" == "1" ]]; then
  log "Starting API server"
  "$PYTHON_BIN" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >/tmp/enterprise-rag-smoke.log 2>&1 &
  SERVER_PID=$!
fi

log "Waiting for /health"
for _ in {1..40}; do
  if [[ "$START_SERVER" == "1" ]] && ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "API failed to start. Server log:" >&2
    cat /tmp/enterprise-rag-smoke.log >&2 || true
    exit 1
  fi
  if curl -fsS "$BASE_URL/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
curl -fsS "$BASE_URL/health" >/dev/null

log "Checking /users/me as viewer"
ME_JSON=$(curl -fsS "$BASE_URL/users/me" -H 'X-User: victor_viewer')
assert_json_field "$ME_JSON" "username" "victor_viewer"
assert_json_field "$ME_JSON" "role" "viewer"

log "Indexing sample docs"
BASE_URL="$BASE_URL" bash scripts/index_sample_docs.sh >/tmp/enterprise-rag-index.log

log "Listing documents"
DOCS_JSON=$(curl -fsS "$BASE_URL/documents" -H 'X-User: victor_viewer')
"$PYTHON_BIN" -c 'import json,sys
arr=json.loads(sys.argv[1])
assert isinstance(arr,list)
assert len(arr)>=1, "No documents found"
print(len(arr))
' "$DOCS_JSON" >/tmp/enterprise-rag-doc-count.txt

log "Running search"
SEARCH_JSON=$(curl -fsS -X POST "$BASE_URL/search" \
  -H 'Content-Type: application/json' \
  -H 'X-User: victor_viewer' \
  -d '{"query":"What are required SSO controls?","top_k":3}')
"$PYTHON_BIN" -c 'import json,sys
obj=json.loads(sys.argv[1])
assert "answer" in obj and obj["answer"], "Missing answer"
assert "citations" in obj and isinstance(obj["citations"],list), "Missing citations"
assert len(obj["citations"])>=1, "No citations returned"
print(len(obj["citations"]))
' "$SEARCH_JSON" >/tmp/enterprise-rag-citation-count.txt

log "Running retrieval evaluation"
EVAL_OUT=$($PYTHON_BIN scripts/evaluate_retrieval.py --username victor_viewer --top-k 5)
printf '%s\n' "$EVAL_OUT" >/tmp/enterprise-rag-eval.log

echo "$EVAL_OUT" | grep -q 'recall@5='

DOC_COUNT=$(cat /tmp/enterprise-rag-doc-count.txt)
CITATION_COUNT=$(cat /tmp/enterprise-rag-citation-count.txt)

log "PASS: docs=$DOC_COUNT citations=$CITATION_COUNT"
