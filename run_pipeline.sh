#!/bin/bash
set -e

# ======================================================================
# BIRD Evaluation Full Pipeline Script
# ======================================================================
# Please adjust the following paths and variables according to the
# testing environment before running the script.
#
# Point LLM / reranker at the SAME artifacts you validated on BIRD dev
# in the Finetune project (merged LLM, specialist reranker).
# If Finetune USE_EVIDENCE=True, use the evidence-trained reranker and
# keep evidence enabled (default in inference.py).
# No embedding model / FAISS — same as Finetune's BIRD eval path.
# ======================================================================

# Paths to BIRD Test Set Data (override via env, e.g. `docker run -e DB_DIR=...`)
DB_DIR="${DB_DIR:-/path/to/test_databases}"
TEST_JSON="${TEST_JSON:-/path/to/test.json}"
COLUMN_MEANING_JSON="${COLUMN_MEANING_JSON:-}"  # unused for official BIRD eval; leave empty

# Model Weights (public HF Hub repos, or local paths). These repos are public,
# so no HF token is required. (If you later make them private, `export HF_TOKEN=...`.)
LLM_MODEL_PATH="${LLM_MODEL_PATH:-treprepr/qwen3.6-27b-bird-evidence-finetuned}"   # merged, evidence-trained
LLM_MODEL_NAME="${LLM_MODEL_NAME:-qwen-rag}"
RERANKER_MODEL="${RERANKER_MODEL:-treprepr/reranker-bird-evidence-finetuned}"      # bge-reranker-v2-m3 specialist (evidence)

# vLLM lives in its own venv (see README). Point this at that venv's vllm binary;
# defaults to whatever `vllm` is on PATH. The client (inference.py) runs in the
# env built from requirements.txt.
VLLM_BIN="${VLLM_BIN:-vllm}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"   # leave headroom for the reranker in the client process
# Qwen3.6 GDN: default max_num_seqs=256 exceeds Mamba cache blocks on ~80G; 64 fits.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
# Optional overrides (defaults: config.py).
PARALLEL_QUESTIONS="${PARALLEL_QUESTIONS:-}"
NUM_VOTES="${NUM_VOTES:-}"

# Max time (seconds) to wait for vLLM to answer /v1/models before giving up.
VLLM_READY_TIMEOUT="${VLLM_READY_TIMEOUT:-1800}"

# Output Paths (override via env; point at a mounted dir when running in Docker)
CHUNKS_JSON="${CHUNKS_JSON:-schema_chunks_test.json}"
PREDICT_JSON="${PREDICT_JSON:-predict.json}"

# Log paths (siblings of outputs; override via --log-file on each Python step)
CHUNKS_LOG="${CHUNKS_JSON%.json}.log"
PREDICT_LOG="${PREDICT_JSON%.json}.log"
VALIDATE_LOG="${PREDICT_JSON%.json}.validate.log"
VLLM_LOG="${PREDICT_JSON%.json}.vllm.log"

# ======================================================================

echo "=========================================="
echo "1. Preflight (before spending GPU on vLLM)"
echo "=========================================="
# Validate the run can succeed BEFORE we pay the multi-minute model-load cost.
# Everything up to and including chunk prep is CPU-only and does not need vLLM,
# so we fail fast here on bad inputs instead of after loading the model.

# 1a. TEST_JSON must parse and every referenced db_id must have its .sqlite.
python3 - "$TEST_JSON" "$DB_DIR" <<'PY'
import json, sys
from pathlib import Path

test_json, db_dir = sys.argv[1], sys.argv[2]
try:
    with open(test_json, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    sys.exit(f"❌ Preflight: could not load TEST_JSON {test_json!r}: {e}")

if not isinstance(data, list) or not data:
    sys.exit(f"❌ Preflight: TEST_JSON {test_json!r} is not a non-empty JSON array.")

db_ids = sorted({item.get("db_id") for item in data if item.get("db_id")})
if not db_ids:
    sys.exit(f"❌ Preflight: no db_id found in TEST_JSON {test_json!r}.")

missing = [d for d in db_ids if not (Path(db_dir) / d / f"{d}.sqlite").is_file()]
if missing:
    sys.exit(
        f"❌ Preflight: {len(missing)} db_id(s) missing <db>/<db>.sqlite under "
        f"{db_dir!r}: {missing[:10]}"
    )

print(f"✅ Preflight: {len(data)} questions, {len(db_ids)} db_id(s), all .sqlite present.")
PY

echo "=========================================="
echo "2. Preparing Database Schema Chunks"
echo "=========================================="
PREPARE_ARGS=(--db-dir "$DB_DIR" --output "$CHUNKS_JSON" --log-file "$CHUNKS_LOG")
if [[ -n "$COLUMN_MEANING_JSON" && -f "$COLUMN_MEANING_JSON" ]]; then
    PREPARE_ARGS+=(--column-meaning "$COLUMN_MEANING_JSON")
fi
python prepare_data.py "${PREPARE_ARGS[@]}"

# 2a. Chunks must be non-empty AND cover every db_id in the test set — otherwise
# inference would silently run with empty schema context and produce garbage SQL.
python3 - "$TEST_JSON" "$CHUNKS_JSON" <<'PY'
import json, sys

test_json, chunks_json = sys.argv[1], sys.argv[2]
try:
    with open(chunks_json, "r", encoding="utf-8") as f:
        chunks = json.load(f)
except Exception as e:
    sys.exit(f"❌ Preflight: could not load CHUNKS_JSON {chunks_json!r}: {e}")

if not isinstance(chunks, list) or not chunks:
    sys.exit(f"❌ Preflight: CHUNKS_JSON {chunks_json!r} has 0 schema chunks.")

with open(test_json, "r", encoding="utf-8") as f:
    data = json.load(f)

test_db_ids = {item.get("db_id") for item in data if item.get("db_id")}
chunk_db_ids = {c.get("db_id") for c in chunks if c.get("db_id")}
uncovered = sorted(test_db_ids - chunk_db_ids)
if uncovered:
    sys.exit(
        f"❌ Preflight: {len(uncovered)} db_id(s) have no schema chunks: "
        f"{uncovered[:10]}"
    )

print(f"✅ Preflight: {len(chunks)} chunks covering all {len(test_db_ids)} db_id(s).")
PY

echo "=========================================="
echo "3. Starting vLLM Server in the background"
echo "=========================================="
# Refuse to launch if port 8000 is already taken — otherwise our fresh server
# fails to bind and dies, but the readiness curl below would succeed against the
# pre-existing (wrong) server and inference would run against the wrong model.
if (exec 3<>/dev/tcp/localhost/8000) 2>/dev/null; then
    exec 3>&- 3<&-
    echo "❌ Port 8000 is already in use. Stop the process bound to it before running." >&2
    exit 1
fi

# NOTE: Depending on the model size, you may need to adjust tensor-parallel-size.
# --trust-remote-code + --language-model-only are REQUIRED for this Qwen3.6
# multimodal checkpoint (skip vision tower so KV fits on one 80G A100 + reranker).
# Tee startup/crash lines to *.vllm.log as well.
"$VLLM_BIN" serve "$LLM_MODEL_PATH" \
    --served-model-name "$LLM_MODEL_NAME" \
    --trust-remote-code \
    --language-model-only \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --port 8000 \
    > >(tee -a "$VLLM_LOG") 2>&1 &
VLLM_PID=$!

cleanup() {
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM (PID $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready (this may take a few minutes)..."
# Poll until the server responds with a real 2xx (curl -fsS fails on HTTP errors,
# unlike plain `curl -s`), but bail out if the process dies or we exceed
# VLLM_READY_TIMEOUT — otherwise a crashed server would leave us curling forever.
READY_DEADLINE=$(( $(date +%s) + VLLM_READY_TIMEOUT ))
until curl -fsS http://localhost:8000/v1/models > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ vLLM process (PID $VLLM_PID) exited before becoming ready. See $VLLM_LOG" >&2
        exit 1
    fi
    if [[ $(date +%s) -ge $READY_DEADLINE ]]; then
        echo "❌ vLLM did not become ready within ${VLLM_READY_TIMEOUT}s. See $VLLM_LOG" >&2
        exit 1
    fi
    sleep 10
done

# Now that it answers, assert it is serving the model we expect. A stale/other
# server on :8000 (or a --served-model-name typo) = wrong-model garbage for every
# question, so fail hard here rather than silently mispredicting the whole set.
if ! curl -fsS http://localhost:8000/v1/models \
        | python3 -c "import json,sys; ids={m.get('id') for m in json.load(sys.stdin).get('data',[])}; sys.exit(0 if '$LLM_MODEL_NAME' in ids else 1)"; then
    echo "❌ vLLM is up but does not serve model '$LLM_MODEL_NAME'. Check --served-model-name / port 8000. See $VLLM_LOG" >&2
    curl -fsS http://localhost:8000/v1/models >&2 || true
    exit 1
fi
echo "vLLM server is up and serving '$LLM_MODEL_NAME'!"

echo "=========================================="
echo "4. Running NL-to-SQL Inference"
echo "=========================================="
INFER_ARGS=(
    --input "$TEST_JSON"
    --db-dir "$DB_DIR"
    --chunks "$CHUNKS_JSON"
    --output "$PREDICT_JSON"
    --log-file "$PREDICT_LOG"
    --llm-model "$LLM_MODEL_NAME"
    --reranker-model "$RERANKER_MODEL"
)
if [[ -n "$PARALLEL_QUESTIONS" ]]; then
    INFER_ARGS+=(--parallel-questions "$PARALLEL_QUESTIONS")
fi
if [[ -n "$NUM_VOTES" ]]; then
    INFER_ARGS+=(--num-votes "$NUM_VOTES")
fi
python inference.py "${INFER_ARGS[@]}"

echo "=========================================="
echo "5. Validating predict.json"
echo "=========================================="
python validate_predict.py \
    --input "$TEST_JSON" \
    --predict "$PREDICT_JSON" \
    --log-file "$VALIDATE_LOG"

echo "Evaluation Pipeline Complete! Predictions saved to $PREDICT_JSON"
echo "Logs: $CHUNKS_LOG | $PREDICT_LOG | $VALIDATE_LOG | $VLLM_LOG"
