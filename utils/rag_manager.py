import torch
import gc
import re
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder

import os
import sys

# Ensure project root is in path (utils/ -> bird_submission/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
import config


class RAGManager:
    """Reranker + schema reconstruction for BIRD submission.

    Matches the Finetune BIRD eval path: skip FAISS/embedding and rerank all
    schema chunks in the DB, keeping the top-k.
    """

    def __init__(self, reranker_model_name):
        self.reranker_model_name = reranker_model_name
        self.reranker_model = None
        self.reranker_tokenizer = None
        # CrossEncoder.predict() and the lazy model load are not thread-safe.
        self._reranker_lock = threading.Lock()

    def _load_reranker(self):
        """Loads the reranker model based on its name."""
        if self.reranker_model is not None:
            return
        with self._reranker_lock:
            if self.reranker_model is not None:
                return
            self._load_reranker_locked()

    def _load_reranker_locked(self):
        print("--- Loading Reranker Model ---")
        if 'qwen' in self.reranker_model_name.lower() and 'seq-cls' not in self.reranker_model_name.lower():
            print(f"Loading Qwen-style reranker: {self.reranker_model_name}")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.reranker_model.eval()
        else:
            print(f"Loading standard CrossEncoder reranker: {self.reranker_model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.reranker_model = CrossEncoder(
                self.reranker_model_name, max_length=config.RERANKER_MAX_LENGTH, device=device,
                automodel_args={"torch_dtype": torch.bfloat16}
            )

    def _get_batch_size(self):
        if 'bge' in self.reranker_model_name.lower():
            return 128
        elif 'qwen' in self.reranker_model_name.lower():
            return 32
        else:
            return 128

    def _rerank(self, all_rerank_pairs):
        """Handles the reranking part of the pipeline."""
        print(f"Reranking {len(all_rerank_pairs)} pairs...")
        batch_size = self._get_batch_size()

        with self._reranker_lock:
            if isinstance(self.reranker_model, CrossEncoder):
                print("Using CrossEncoder.predict() for reranking.")
                return self.reranker_model.predict(all_rerank_pairs, batch_size=batch_size, show_progress_bar=True)

            print("Using manual batching for SequenceClassification reranker.")
            all_rerank_scores = []
            with torch.no_grad():
                for i in range(0, len(all_rerank_pairs), batch_size):
                    batch = all_rerank_pairs[i:i + batch_size]
                    inputs = self.reranker_tokenizer(
                        [p[0] for p in batch], [p[1] for p in batch],
                        padding=True, truncation=True, return_tensors="pt", max_length=config.RERANKER_MAX_LENGTH
                    ).to(self.reranker_model.device)
                    scores = self.reranker_model(**inputs).logits.sigmoid().squeeze(-1).cpu().tolist()
                    if isinstance(scores, float):
                        scores = [scores]
                    all_rerank_scores.extend(scores)
                    if (i // batch_size) % 100 == 0:
                        print(f"   Reranked batch {i // batch_size}...")
            return all_rerank_scores

    def unload_models(self):
        print("\nUnloading RAG models to free up VRAM...")
        try:
            if self.reranker_tokenizer:
                del self.reranker_tokenizer
            if self.reranker_model:
                del self.reranker_model
        except Exception:
            pass
        self.reranker_tokenizer = self.reranker_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("RAG models unloaded and cache explicitly cleared.")

    @staticmethod
    def find_used_tables(sql_query, table_list):
        """Helper to find which tables from a list appear in a SQL string."""
        used = set()
        sql_lower = sql_query.lower()
        for table in table_list:
            pattern = r'(?<!\w)' + re.escape(table.lower()) + r'(?!\w)'
            if re.search(pattern, sql_lower):
                used.add(table)
        return used

    @staticmethod
    def reconstruct_final_schema(db_id, reranked_chunks):
        """Reconstructs the partial schema string from table chunks with deduplication."""
        final_schema_parts = []
        seen_contents = set()

        sorted_chunks = sorted(reranked_chunks, key=lambda x: x.get('table_name', ''))

        for chunk in sorted_chunks:
            if chunk.get('db_id') == db_id and chunk['content'] not in seen_contents:
                final_schema_parts.append(chunk['content'])
                seen_contents.add(chunk['content'])

        return "\n".join(final_schema_parts) or f"-- No relevant schema context found for database '{db_id}'."
