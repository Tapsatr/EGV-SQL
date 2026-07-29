import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Used by the RAG Manager
RERANKER_MAX_LENGTH = 2048

# How many top schema chunks to keep after reranking (no FAISS/embedding on BIRD).
RAG_PARAMS = {
    "top_k": 12,
}

# Must match how the LLM / reranker were trained (Finetune USE_EVIDENCE).
USE_EVIDENCE = True

# Official BIRD predict.json value delimiter: "{sql}\t----- bird -----\t{db_id}"
BIRD_PRED_DELIMITER = "\t----- bird -----\t"

# Sentinel written for keys that never produced a prediction, only when
# inference.py is run with --fill-missing. Non-empty and executable so the
# predict file stays complete/submittable (scores 0 EX rather than breaking eval).
FALLBACK_SQL = "SELECT 1"

# The number of votes and temperature for the LLM.
# You can override these via CLI arguments in inference.py
NUM_VOTES = 10
TEMPERATURE = 0.6
PARALLEL_QUESTIONS = 10
MAX_OUTPUT_TOKENS = 6000
API_MAX_RETRIES = 2
API_REQUEST_TIMEOUT = 600.0
