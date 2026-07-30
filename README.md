# BIRD Submission — Qwen RAG (Category 1)

NL-to-SQL inference for the hidden BIRD **test** set. Validated on BIRD **dev**
(local EX **71.64%**, 1099/1534; see `dev_predict.json`).

| | |
|---|---|
| **Category** | 1 — single **A100 80G** |
| **CUDA** | 12.2 / 12.3 (`Dockerfile` uses `nvidia/cuda:12.3.2-…-devel`) |
| **Models (public HF)** | `treprepr/qwen3.6-27b-bird-evidence-finetuned`, `treprepr/reranker-bird-evidence-finetuned` |
| **`column_meaning.json`** | **Not used** (schema from BIRD `database_description/*.csv` only) |
| **Dev runtime** | ~6 h on H200 MIG ~143 GB; ~1 day on A100 80G (Category 1) |

## Quick start (Exp Team)

Mount `test.json` and `test_databases/`. Output is `predict.json`. Resume-safe:
re-run the same command to continue after failures.

```bash
# 1) Build (once; needs network for pip + base image)
docker build -t bird-qwen-rag .

# 2) Run (set the host paths on the left of -v to your test set)
docker run --gpus all \
  -v /path/to/test_databases:/data/test_databases \
  -v /path/to/test.json:/data/test.json \
  -v "$PWD/out":/data/out \
  -e DB_DIR=/data/test_databases \
  -e TEST_JSON=/data/test.json \
  -e PREDICT_JSON=/data/out/predict.json \
  -e CHUNKS_JSON=/data/out/schema_chunks_test.json \
  bird-qwen-rag
```

That single `docker run` starts vLLM, builds schema chunks, runs inference,
validates `predict.json`, and stops vLLM. Logs land next to the outputs under
`out/` (`*.log`, `*.vllm.log`).

Optional HF cache (avoids re-downloading ~52 GB weights):

```bash
  -v /path/to/hf_cache:/app/.cache/huggingface \
```

## Compliance

- **No third-party APIs / no DB upload.** Only `requests` to a **local** vLLM on
  `http://localhost:8000/v1` (`OPENAI_API_KEY=EMPTY`).
- Candidate SQL is executed on the provided SQLite DBs for syntax/voting only.
  **Ground-truth SQL is never used.**

## What the pipeline does

`run_pipeline.sh` → `prepare_data.py` → vLLM serve → `inference.py` →
`validate_predict.py`.

- Rerank **all** schema chunks per DB (no embedding / FAISS).
- Evidence **on** (matches finetune). Voting: `--num-votes 10`.
- On 80 G: `--parallel-questions 5`, `--max-num-seqs 64`, `GPU_MEM_UTIL=0.88`.

`predict.json` keys are **0-based indices** in `test.json` order:

```json
{
  "0": "SELECT ...\t----- bird -----\tdb_id"
}
```

## Resume / logging

`inference.py` writes each successful key immediately. Re-run the same Docker /
`./run_pipeline.sh` command to skip completed keys. Failures are not written
(so they retry). If anything is still missing after retries, a final pass with
`inference.py --fill-missing` writes `SELECT 1` for gaps (scores 0 EX).

## Dev predictions

`dev_predict.json` is included for reproduction. Local EX **71.64%** on the
original BIRD-dev split (not `bird-sql-dev-1106`): simple 76.65%, moderate
65.73%, challenging 58.62%.

## Bare-metal alternative (if not using Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m venv venv-vllm
./venv-vllm/bin/pip install -U pip
./venv-vllm/bin/pip install --no-cache-dir "vllm==0.26.0"

export VLLM_BIN=./venv-vllm/bin/vllm
export DB_DIR=/path/to/test_databases
export TEST_JSON=/path/to/test.json
export PREDICT_JSON=./predict.json
./run_pipeline.sh
```

Needs `nvcc` on `PATH` (CUDA **devel** toolkit). Prefer the Docker path above.

## Notes

- `test_tables.json` is not required by our scripts (DBs + `test.json` suffice).
- Contact: [BIRD submission guidelines](https://bird-bench.github.io/).
