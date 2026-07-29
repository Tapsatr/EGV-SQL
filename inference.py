import os
import sys
import json
import sqlite3
import time
from pathlib import Path
import requests
import concurrent.futures

import config
from utils.rag_manager import RAGManager
from utils.model_handlers import get_model_handler
from utils.vllm_lifecycle import stop_vllm_on_exit_if
from utils.logging_setup import (
    setup_logging,
    get_logger,
    resolve_log_path,
    sibling_log_path,
)

logger = get_logger(__name__)


class VLLMApiError(Exception):
    pass


def format_bird_prediction(sql, db_id):
    """Official BIRD predict.json value: SQL\\t----- bird -----\\tdb_id"""
    return f"{normalize_sql(sql)}{config.BIRD_PRED_DELIMITER}{db_id}"


def prediction_key(idx, item, key_by):
    if key_by == "question_id":
        return str(item.get("question_id", idx))
    return str(idx)


def _parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for BIRD Evaluation.")
    parser.add_argument("--input", type=str, required=True, help="Path to test.json")
    parser.add_argument("--db-dir", type=str, required=True, help="Path to database directory (e.g. test_databases/)")
    parser.add_argument("--output", type=str, required=True, help="Path to save predict.json")
    parser.add_argument("--chunks", type=str, required=True, help="Path to schema_chunks.json")

    parser.add_argument("--llm-model", type=str, default="qwen-rag", help="LLM model name served by vLLM")
    parser.add_argument("--reranker-model", type=str, required=True,
                        help="Reranker model path (use your Finetune evidence specialist when USE_EVIDENCE=True)")

    parser.add_argument("--num-votes", type=int, default=config.NUM_VOTES, help="Number of voting samples per question.")
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("--parallel-questions", type=int, default=config.PARALLEL_QUESTIONS,
                        help="Number of questions to process in parallel.")
    parser.add_argument("--no-retry", action="store_true", help="Disable execution-guided retry.")
    parser.add_argument("--no-evidence", action="store_true",
                        help="Disable BIRD evidence injection (only for ablation; keep off for submission).")
    parser.add_argument("--key-by", choices=("index", "question_id"), default="index",
                        help="predict.json key scheme. Default 'index' matches official BIRD examples.")
    parser.add_argument("--fill-missing", action="store_true",
                        help="After the run, write fallback SQL (config.FALLBACK_SQL) for any keys that "
                             "still failed, so predict.json is complete/submittable. Use only for the final pass.")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Rotating log file path (default: sibling of --output as {stem}.log).",
    )

    parser.add_argument("--stop-vllm-on-exit", action="store_true", help="Stop the vLLM server when evaluation finishes.")

    return parser.parse_args()


def load_schema_chunks(chunks_path):
    logger.info("--- Loading schema chunks from: %s ---", chunks_path)
    try:
        with open(chunks_path, 'r', encoding='utf-8') as f:
            schema_chunks = json.load(f)
        logger.info("✅ Successfully loaded %d schema chunks.", len(schema_chunks))
        return schema_chunks
    except Exception as e:
        logger.error("❌ Failed to load schema chunks: %s", e)
        sys.exit(1)


def execute_sql(db_path, query, timeout=30):
    if not os.path.exists(db_path):
        return "db_not_found", None

    conn = None
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        start_time = time.time()

        def progress_handler():
            if time.time() - start_time > timeout:
                return 1
            return 0

        conn.set_progress_handler(progress_handler, 10000)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        results = [tuple(str(v).lower() if isinstance(v, str) else round(v, 4) if isinstance(v, float) else v for v in row) for row in results]
        return "success", results

    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            return f"error: query timed out after {timeout} seconds", None
        return f"error: {str(e)}", None
    except Exception as e:
        return f"error: {str(e)}", None
    finally:
        if conn:
            conn.close()


def retrieve_schema_chunks(question, db_id, schema_chunks, rag_manager, evidence=None):
    db_chunks = [chunk for chunk in schema_chunks if chunk.get('db_id') == db_id]
    if evidence and str(evidence).strip():
        rerank_query = f"{question} {str(evidence).strip()}"
    else:
        rerank_query = question

    if not db_chunks:
        return []

    # Same path as Finetune validate_vllm for BIRD: rerank all chunks in the DB (no FAISS).
    if not rag_manager.reranker_model:
        rag_manager._load_reranker()
    seen_contents = set()
    unique_db_chunks = []
    for chunk in db_chunks:
        if chunk['content'] not in seen_contents:
            unique_db_chunks.append(chunk)
            seen_contents.add(chunk['content'])

    all_rerank_pairs = [[rerank_query, chunk['content']] for chunk in unique_db_chunks]
    all_rerank_scores = rag_manager._rerank(all_rerank_pairs)

    valid_scored_chunks = [(score, chunk) for score, chunk in zip(all_rerank_scores, unique_db_chunks)]
    scored_chunks = sorted(valid_scored_chunks, key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored_chunks[:config.RAG_PARAMS["top_k"]]]


def generate_responses_requests(api_base, model_name, user_prompt, temp_override, num_votes):
    current_temp = temp_override if temp_override is not None else config.TEMPERATURE
    if current_temp == 0.0:
        num_votes = 1

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'EMPTY')}"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": config.MAX_OUTPUT_TOKENS,
        "temperature": current_temp,
        "top_p": 0.95,
        "n": num_votes,
        "stream": False
    }

    for attempt in range(config.API_MAX_RETRIES + 1):
        try:
            response = requests.post(
                f"{api_base}/chat/completions", headers=headers, json=payload, timeout=config.API_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            responses = []
            for choice in data.get('choices', []):
                responses.append(choice.get('message', {}).get('content', ''))
            return responses
        except Exception as e:
            if attempt < config.API_MAX_RETRIES:
                wait_secs = 2 ** attempt
                logger.warning(
                    "API Error (attempt %d/%d): %s. Retrying in %ds...",
                    attempt + 1, config.API_MAX_RETRIES + 1, e, wait_secs,
                )
                time.sleep(wait_secs)
            else:
                logger.error("API Error: %s", e)
                raise VLLMApiError(str(e)) from e


def generate_sql(question, db_id, api_base, model_name, schema_chunks, rag_manager, previous_sql=None, error_msg=None, retrieved_schema=None, temp_override=None, evidence=None, num_votes=1):
    if retrieved_schema is None:
        reranked_chunks = retrieve_schema_chunks(question, db_id, schema_chunks, rag_manager, evidence=evidence)
        retrieved_schema = RAGManager.reconstruct_final_schema(db_id, reranked_chunks)

    user_prompt = f"""You are an expert SQLite programmer. Your task is to write a SQL query that answers the following question.
    You must ONLY use the tables and columns provided in the schema below. Do not use any tables or columns that are not explicitly listed.

    Provide your step-by-step reasoning inside <think></think> tags before writing the final SQL.

    DB:
    {db_id}

    Schema:
    ```sql
    {retrieved_schema}
    ```

    Question:
    {question}"""

    if evidence and str(evidence).strip():
        user_prompt += f"\n\n    Hint:\n    {str(evidence).strip()}"

    if previous_sql and error_msg:
        user_prompt += f"\n\nYour previous SQL query:\n```sql\n{previous_sql}\n```\nFailed with the following error:\n{error_msg}\nPlease fix the error and provide a corrected SQL query."

    responses = generate_responses_requests(api_base, model_name, user_prompt, temp_override, num_votes)
    return responses, retrieved_schema


def normalize_sql(sql):
    if not isinstance(sql, str):
        return ""
    sql = sql.strip()
    if sql.endswith(';'):
        sql = sql[:-1]
    return sql


def get_best_voted_sql(question, db_id, api_base, model_name, schema_chunks, rag_manager, model_handler,
                       predicted_sql_raw_list, db_path, retrieved_context, evidence=None, enable_retry=True, num_votes=1):
    executed_sqls = {}
    MAX_RETRIES = 3 if enable_retry else 1
    current_retry = 0
    current_raw_list = predicted_sql_raw_list

    while current_retry < MAX_RETRIES:
        executed_sqls.clear()
        successful_candidates = []

        for raw_sql in current_raw_list:
            candidate_sql = model_handler.extract_sql(raw_sql)
            norm_candidate_sql = normalize_sql(candidate_sql)

            if norm_candidate_sql not in executed_sqls:
                status, results = execute_sql(db_path, norm_candidate_sql)
                executed_sqls[norm_candidate_sql] = (status, results)
            else:
                status, results = executed_sqls[norm_candidate_sql]

            if status == "success":
                successful_candidates.append({'sql': norm_candidate_sql, 'raw_result': results, 'status': status})

        if successful_candidates:
            sql_groups = {}
            for c in successful_candidates:
                group_key = tuple(sorted(c['raw_result'], key=lambda row: tuple(str(v) for v in row))) if c['raw_result'] else ()
                if group_key not in sql_groups:
                    sql_groups[group_key] = {'count': 0, 'weight': 0, 'sql': c['sql'], 'status': c['status'], 'raw_result': c['raw_result']}
                else:
                    if len(c['sql']) < len(sql_groups[group_key]['sql']):
                        sql_groups[group_key]['sql'] = c['sql']
                sql_groups[group_key]['count'] += 1
                if c['raw_result']:
                    sql_groups[group_key]['weight'] += 1.0
                else:
                    sql_groups[group_key]['weight'] += 0.5

            best_group = max(sql_groups.values(), key=lambda x: (x['weight'], x['count'], -len(x['sql'])))
            return best_group['sql'], best_group['status'], best_group['raw_result']
        else:
            first_norm_sql = normalize_sql(model_handler.extract_sql(current_raw_list[0]))
            exec_status, predicted_results = execute_sql(db_path, first_norm_sql)

            current_retry += 1
            if current_retry < MAX_RETRIES:
                retry_temp = 0.2 + (current_retry * 0.2) if config.TEMPERATURE == 0.0 else config.TEMPERATURE
                current_raw_list, _ = generate_sql(
                    question, db_id, api_base, model_name, schema_chunks, rag_manager,
                    previous_sql=first_norm_sql, error_msg=exec_status, retrieved_schema=retrieved_context,
                    temp_override=retry_temp, evidence=evidence, num_votes=num_votes
                )
            else:
                return first_norm_sql, exec_status, predicted_results


def evaluate_one_question(item, current_case, args, api_base, model_handler, schema_chunks, rag_manager, enable_retry):
    db_id = item['db_id']
    question = item['question']
    key = prediction_key(current_case, item, args.key_by)
    use_evidence = config.USE_EVIDENCE and not args.no_evidence
    evidence = item.get('evidence', '') if use_evidence else ''
    db_path = Path(args.db_dir) / db_id / f"{db_id}.sqlite"

    if not db_path.exists():
        err = f"SQLite DB not found at {db_path}"
        logger.error("❌ key=%s db=%s status=db_not_found | %s (NOT saved to predict)", key, db_id, err)
        return {'key': key, 'case': current_case, 'db_id': db_id, 'status': 'db_not_found', 'error': err}

    try:
        responses, retrieved_context = generate_sql(
            question, db_id, api_base, args.llm_model, schema_chunks, rag_manager,
            evidence=evidence, num_votes=args.num_votes
        )

        norm_predicted_sql, exec_status, _ = get_best_voted_sql(
            question, db_id, api_base, args.llm_model, schema_chunks, rag_manager, model_handler,
            responses, db_path, retrieved_context, evidence=evidence, enable_retry=enable_retry, num_votes=args.num_votes
        )

        # Only treat as a savable success when the chosen SQL actually executed and
        # is non-empty. Otherwise report failure so it is NOT written to predict.json
        # and can be retried on resume (avoids BIRD's >5% abnormal-SQL penalty).
        if exec_status != "success" or not norm_predicted_sql.strip():
            logger.warning(
                "⚠️ key=%s db=%s status=exec_failed | exec_status=%s (NOT saved to predict)",
                key, db_id, exec_status,
            )
            return {
                'key': key,
                'case': current_case,
                'db_id': db_id,
                'status': 'exec_failed',
                'error': f"exec_status={exec_status!r}",
            }

        return {
            'key': key,
            'case': current_case,
            'db_id': db_id,
            'status': 'success',
            'sql': norm_predicted_sql,
            'exec_status': exec_status,
        }
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.warning(
            "⚠️ key=%s db=%s status=worker_error | %s (NOT saved to predict)",
            key, db_id, err,
        )
        return {'key': key, 'case': current_case, 'db_id': db_id, 'status': 'worker_error', 'error': err}


def atomic_write_json(path, data):
    """Crash-safe write: dump to a temp file, fsync, then atomically replace.

    Keeps a `.bak` of the previous good file so a crash mid-write can never
    destroy already-completed predictions.
    """
    path = os.fspath(path)
    dir_name = os.path.dirname(os.path.abspath(path))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        try:
            os.replace(path, f"{path}.bak")
        except OSError:
            pass
    os.replace(tmp_path, path)
    try:
        dir_fd = os.open(dir_name, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def load_previous_results(predict_path):
    """Load existing predictions, falling back to the `.bak` if the main file
    is missing or corrupt (e.g. a crash during a previous checkpoint write).

    Distinguishes a genuine fresh start (no checkpoint on disk) from a corrupt
    checkpoint. If any checkpoint file exists but *every* existing candidate is
    unparseable, we exit nonzero instead of silently returning {} — starting
    from {} would treat the whole run as unfinished and, on the first save,
    `atomic_write_json` would move the corrupt-but-maybe-recoverable file over
    the `.bak`, destroying resume state and any salvageable predictions.
    """
    candidates = (predict_path, f"{predict_path}.bak")
    existing = [c for c in candidates if os.path.exists(c)]

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        with open(candidate, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                logger.warning("⚠️ Could not parse %s; trying fallback.", candidate)
                continue
        if candidate != predict_path:
            logger.warning("⚠️ Recovered predictions from backup %s.", candidate)
        return data

    if existing:
        logger.error(
            "❌ Checkpoint file(s) present but unreadable: %s. Refusing to start "
            "from scratch so resume state / prior predictions are not overwritten. "
            "Inspect or remove the corrupt file(s) before re-running. exit_code=1",
            ", ".join(existing),
        )
        sys.exit(1)

    return {}


def main():
    args = _parse_arguments()
    log_path = resolve_log_path(args.log_file, sibling_log_path(args.output, ".log"))
    setup_logging(log_path)
    args.log_file = log_path
    with stop_vllm_on_exit_if(args.stop_vllm_on_exit):
        _run_inference(args)


def _run_inference(args):
    api_base = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    model_handler = get_model_handler(args.llm_model)
    enable_retry = not args.no_retry

    if args.no_evidence and config.USE_EVIDENCE:
        logger.warning("⚠️ --no-evidence set while config.USE_EVIDENCE=True. Use only for ablation.")

    schema_chunks = load_schema_chunks(args.chunks)

    logger.info("--- Schema selection: rerank all chunks per DB (no embedding / FAISS) ---")
    rag_manager = RAGManager(args.reranker_model)

    with open(args.input, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predict_results = load_previous_results(args.output)

    remaining_data = []
    for idx, item in enumerate(test_data):
        key = prediction_key(idx, item, args.key_by)
        if key not in predict_results:
            remaining_data.append((idx, item))

    logger.info(
        "=== RUN HEADER === input=%s output=%s log=%s key-by=%s num_votes=%d "
        "parallel=%d llm=%s reranker=%s fill_missing=%s",
        args.input,
        args.output,
        args.log_file,
        args.key_by,
        args.num_votes,
        args.parallel_questions,
        args.llm_model,
        args.reranker_model,
        args.fill_missing,
    )
    logger.info(
        "Starting inference for %d questions (Skipping %d already completed; key-by=%s)",
        len(remaining_data),
        len(predict_results),
        args.key_by,
    )

    total_failed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_questions) as executor:
            futures = {
                executor.submit(
                    evaluate_one_question, item, idx, args, api_base, model_handler,
                    schema_chunks, rag_manager, enable_retry
                ): (idx, item) for idx, item in remaining_data
            }

            try:
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    key = str(result['key'])

                    if result['status'] != 'success':
                        total_failed += 1
                        logger.warning(
                            "[%d/%d] ⏭️ NOT SAVED key=%s db=%s status=%s | %s",
                            completed + total_failed,
                            len(remaining_data),
                            key,
                            result.get('db_id'),
                            result['status'],
                            result.get('error'),
                        )
                        continue

                    predict_results[key] = format_bird_prediction(result['sql'], result['db_id'])
                    completed += 1

                    atomic_write_json(args.output, predict_results)

                    preview = result['sql'][:60]
                    logger.info(
                        "[%d/%d] ✅ key=%s db=%s status=success | SQL: %s...",
                        completed,
                        len(remaining_data),
                        key,
                        result.get('db_id'),
                        preview,
                    )
            except KeyboardInterrupt:
                logger.warning("\n\n🛑 Inference stopped by user. Pending tasks canceled.")
                for f in futures:
                    f.cancel()
                raise
    except KeyboardInterrupt:
        pass

    key_to_item = {prediction_key(idx, item, args.key_by): item for idx, item in enumerate(test_data)}
    expected_keys = set(key_to_item)

    def _sorted_missing():
        remaining = expected_keys - set(predict_results.keys())
        return sorted(remaining, key=lambda k: int(k) if k.isdigit() else k)

    missing = _sorted_missing()

    if missing and args.fill_missing:
        logger.info(
            "--- Filling %d missing key(s) with fallback SQL %r ---",
            len(missing),
            config.FALLBACK_SQL,
        )
        for key in missing:
            db_id = key_to_item[key].get('db_id', '')
            predict_results[key] = format_bird_prediction(config.FALLBACK_SQL, db_id)
            logger.info("fill-missing key=%s db=%s FALLBACK_SQL=%r", key, db_id, config.FALLBACK_SQL)
        atomic_write_json(args.output, predict_results)
        missing = _sorted_missing()

    logger.info("=" * 50)
    logger.info("                📊 INFERENCE COMPLETE 📊")
    logger.info("=" * 50)
    logger.info("Saved %d / %d predictions to %s", len(predict_results), len(test_data), args.output)
    if total_failed:
        logger.info("Failed this run (not written; safe to resume): %d", total_failed)
    if missing:
        logger.error(
            "❌ Missing %d prediction keys (e.g. %s). Full missing count=%d.",
            len(missing),
            missing[:5],
            len(missing),
        )
        logger.error(
            "   Re-run to resume, or re-run with --fill-missing to write fallback SQL for the gaps. "
            "exit_code=1"
        )
        sys.exit(1)

    logger.info(
        "✅ predict.json is complete. Run: python validate_predict.py --input %s --predict %s",
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()
