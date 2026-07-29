# BIRD Submission - Qwen RAG Pipeline

Inference and data-preparation scripts for evaluating our fine-tuned NL-to-SQL model on the hidden BIRD test set.

Train and validate on **dev** in the Finetune project first; use this repo only to produce the official test `predict.json`.

## Compliance Note
- **No Third-Party APIs**: Uses the standard Python `requests` library against a *locally-hosted* vLLM server. The `OPENAI_BASE_URL` / `OPENAI_API_KEY` env vars are only used to reach that local server and default to `http://localhost:8000/v1` and `EMPTY`; no data is sent off-box and the `openai` package is not used.
- **Execution-Guided Decoding**: Candidate SQLs are executed on the provided test databases only to check syntax/runtime validity and vote. It **does not** use ground-truth SQLs.

## Test inputs we use
- `test.json`, `test_databases/`, `test_tables.json`: **required.**
- `column_meaning.json`: **optional but used if provided.** When passed to
  `prepare_data.py --column-meaning`, it replaces the per-DB CSV metadata for
  richer column descriptions; if omitted, we fall back to the BIRD
  `database_description/*.csv` files. Either path works — supply it if you'd like
  us to use it.
- Predicted dev SQLs: see [Dev-set predictions](#dev-set-predictions) below.

## Environment Setup

Two environments, kept separate on purpose (vLLM pins its own `torch`/`xformers`
that clash with the client stack).

**1. Client env** (runs `prepare_data.py`, `inference.py`, `validate_predict.py`
and loads the bge reranker):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. vLLM serving env** (serves the merged LLM on `localhost:8000`):
```bash
python -m venv venv-vllm
./venv-vllm/bin/pip install -U pip
./venv-vllm/bin/pip install -U vllm     # must be new enough for Qwen3.6 / Qwen3_5ForConditionalGeneration
```
- This checkpoint is a **Qwen3.6-27B multimodal Gated-DeltaNet** model. It needs
  a vLLM release that includes `Qwen3_5ForConditionalGeneration` support
  (upstream PR #34110) and **transformers >= 5.1.0**; if the current stable vLLM
  predates it, install a nightly (`--pre` / the vLLM nightly index).
- The GDN kernels **JIT-compile at runtime and require `nvcc`** (a CUDA *devel*
  toolkit, not just the runtime). The eval CUDA is 12.2 / 12.3 — make sure the
  CUDA compiler is on PATH (a `-devel` CUDA base image bakes it in).
- Point `run_pipeline.sh`'s `VLLM_BIN` at `./venv-vllm/bin/vllm`.

## Submission category & resources
- **Category 1 (single A100 80G).** The merged LLM is ~27B; in bfloat16 it fits
  on one 80GB A100 alongside the small bge reranker.
- **Expected resources / runtime:** 1× A100 80G. On BIRD **dev** (1534 questions)
  a full run took roughly **X hours** end-to-end with the default
  `--num-votes 10` and `--parallel-questions 10`. Scale accordingly for the test
  set. *(Update X with your measured dev runtime before sending.)*
- If more GPUs are available, raise `--tensor-parallel-size` in `run_pipeline.sh`
  to speed up serving.

## Model paths (important)

Point weights at the **same** artifacts you used for your best Finetune BIRD-dev run:

| Component | Artifact (HF Hub) |
|-----------|-------------------|
| LLM | `treprepr/qwen3.6-27b-bird-evidence-finetuned` (merged, served by vLLM) |
| Reranker | `treprepr/reranker-bird-evidence-finetuned` (bge-reranker-v2-m3 specialist, evidence) |

There is **no embedding model or FAISS index** — same as Finetune’s BIRD eval path (rerank all chunks per DB). Keep evidence **on** (default) when Finetune used `USE_EVIDENCE=True`. Both HF repos are **public**, so no token is needed. (If you later make them private, `export HF_TOKEN=...` so vLLM/transformers can download them.)

## Running the Evaluation

### Option A: Docker (recommended)

The included `Dockerfile` bakes in the exact working environment: a CUDA
**-devel** base (for the `nvcc` the GDN kernels JIT-compile against), the client
env from `requirements.txt`, and an isolated `venv-vllm` with a Qwen3.6-capable
vLLM. This avoids the version/toolkit pitfalls of a bare `pip install`.

```bash
docker build -t bird-qwen-rag .

docker run --gpus all \
  -v /host/test_databases:/data/test_databases \
  -v /host/test.json:/data/test.json \
  -v /host/column_meaning.json:/data/column_meaning.json \
  -v "$PWD/out":/data/out \
  -e DB_DIR=/data/test_databases \
  -e TEST_JSON=/data/test.json \
  -e COLUMN_MEANING_JSON=/data/column_meaning.json \
  -e PREDICT_JSON=/data/out/predict.json \
  -e CHUNKS_JSON=/data/out/schema_chunks_test.json \
  bird-qwen-rag
```

**CUDA tag:** the base image is `nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04`,
matching BIRD's eval driver (CUDA 12.2/12.3). A 12.3 toolkit runs natively there
and also on newer dev drivers (e.g. an H200 pod reporting CUDA 13.0) by backward
compatibility, so the *same image* works in both places. Don't raise this above
the eval driver's CUDA — the GDN kernels JIT at runtime, and a too-new toolkit can
emit code the older eval driver can't load.

**MIG / multi-GPU:** on a MIG slice (our dev H200 runs one ~143 GB slice), keep
`--tensor-parallel-size 1` — vLLM can't shard across MIG instances. On a full
multi-GPU box you can raise TP for faster serving.

### Option B: bare metal

1. Set up both environments (see [Environment Setup](#environment-setup)) and
   point `run_pipeline.sh`'s `VLLM_BIN` at `./venv-vllm/bin/vllm`.
2. Edit `run_pipeline.sh` paths (test DBs, `test.json`, model weights) — or export
   `DB_DIR` / `TEST_JSON` / `COLUMN_MEANING_JSON` / `LLM_MODEL_PATH` as env vars.
3. Run:

```bash
./run_pipeline.sh
```

The script starts vLLM, builds schema chunks, runs inference, validates `predict.json`, then stops vLLM.

### Manual Step-by-Step Execution

1. **Start vLLM** (from the `venv-vllm` env):
```bash
./venv-vllm/bin/vllm serve treprepr/qwen3.6-27b-bird-evidence-finetuned \
  --served-model-name "qwen-rag" \
  --trust-remote-code \
  --language-model-only \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.85 &
```

2. **Chunk test databases:**
```bash
python prepare_data.py \
  --db-dir /path/to/test_databases \
  --output schema_chunks_test.json \
  --column-meaning /path/to/column_meaning.json   # optional (TA-SQL flat or nested)
```

3. **Run inference:**
```bash
python inference.py \
  --input /path/to/test.json \
  --db-dir /path/to/test_databases \
  --chunks schema_chunks_test.json \
  --output predict.json \
  --llm-model "qwen-rag" \
  --reranker-model treprepr/reranker-bird-evidence-finetuned
```

4. **Validate before sending:**
```bash
python validate_predict.py --input /path/to/test.json --predict predict.json
```

### predict.json format

Official BIRD shape (file-order string keys):

```json
{
  "0": "SELECT ...\t----- bird -----\tdb_id",
  "1": "SELECT ...\t----- bird -----\tdb_id"
}
```

Default keying is **index in `test.json` order** (`--key-by index`). Use `--key-by question_id` only if organizers require it.

### Resuming from interruptions

`inference.py` saves after each successful question. Re-run the same command to skip completed keys. Failed workers are **not** written (so they retry). The run exits non-zero if any keys are still missing.

If a question keeps failing after retries and you need a complete file to submit, do a final pass with `--fill-missing`: any still-missing keys are written with `config.FALLBACK_SQL` (`SELECT 1`) so the file is complete. Those entries score 0 EX but won't break the organizers' eval. Use it only once you've exhausted normal resumes.

### Logs

Each long-running Python step writes a **rotating** log (10 MB × 5 backups) next to its primary output, and also mirrors the same lines to stdout:

| Step | Default log path |
|------|------------------|
| `prepare_data.py` | sibling of `--output` → e.g. `schema_chunks_test.log` |
| `inference.py` | sibling of `--output` → e.g. `predict.log` |
| `validate_predict.py` | sibling of `--predict` → e.g. `predict.validate.log` |
| vLLM (via `run_pipeline.sh`) | sibling of `PREDICT_JSON` → e.g. `predict.vllm.log` |

Override any Python step with `--log-file PATH`. Re-running inference still resumes from existing keys in `predict.json`; the log records successes, failures that were **not** saved, API retries, and end-of-run missing-key summaries so you can pick up from a failed example.

`*.log` is already in `.dockerignore` (keeps image builds lean). The submission zip can omit large logs, but **retain** the full dev-run log locally — organizers can ask you to attach `*.log` when debugging.

## Dev-set predictions

Our predicted SQLs on BIRD **dev** are included as `dev_predict.json` (same
`SQL\t----- bird -----\tdb_id` format) so you can follow and reproduce our dev
results. *(Add the file to the zip before sending — or, if you prefer not to
open-source dev predictions, delete this section and note that to the organizers
in your email.)*

## Notes

- Schema selection is **rerank-all-chunks** only (no embedding / FAISS), matching Finetune’s BIRD eval path.
- `column_meaning.json` supports TA-SQL keys `db|table|column` and nested dicts; otherwise BIRD `database_description/*.csv` is used.
- Contact organizers per [BIRD submission guidelines](https://bird-bench.github.io/) for test evaluation.
