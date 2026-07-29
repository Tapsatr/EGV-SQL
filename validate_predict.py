#!/usr/bin/env python3
"""Validate a BIRD predict.json before submission."""

from __future__ import annotations

import argparse
import json
import sys

import config
from utils.logging_setup import (
    setup_logging,
    get_logger,
    resolve_log_path,
    sibling_log_path,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate BIRD predict.json format and coverage.")
    parser.add_argument("--input", required=True, help="Path to test.json (or dev.json)")
    parser.add_argument("--predict", required=True, help="Path to predict.json")
    parser.add_argument(
        "--key-by",
        choices=("index", "question_id"),
        default="index",
        help="Key scheme used when writing predict.json (default: index)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Rotating log file path (default: sibling of --predict as {stem}.validate.log).",
    )
    return parser.parse_args()


def expected_key(idx, item, key_by):
    if key_by == "question_id":
        return str(item.get("question_id", idx))
    return str(idx)


def main():
    args = parse_args()
    log_path = resolve_log_path(args.log_file, sibling_log_path(args.predict, ".validate.log"))
    setup_logging(log_path)

    logger.info(
        "=== RUN HEADER === input=%s predict=%s log=%s key-by=%s",
        args.input,
        args.predict,
        log_path,
        args.key_by,
    )

    delim = config.BIRD_PRED_DELIMITER

    with open(args.input, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(args.predict, "r", encoding="utf-8") as f:
        predict = json.load(f)

    if not isinstance(predict, dict):
        logger.error("❌ predict.json must be a JSON object keyed by question index")
        sys.exit(1)

    errors = []
    warnings = []

    expected_keys = [expected_key(i, item, args.key_by) for i, item in enumerate(test_data)]
    expected_set = set(expected_keys)
    predict_keys = set(str(k) for k in predict.keys())

    missing = [k for k in expected_keys if k not in predict_keys]
    extra = sorted(predict_keys - expected_set)

    if missing:
        errors.append(f"Missing {len(missing)} keys (first: {missing[:5]})")
    if extra:
        warnings.append(f"{len(extra)} unexpected keys (first: {extra[:5]})")

    empty_sql = 0
    bad_delim = 0
    db_mismatch = 0
    sentinel_fill = 0
    fallback_sql = config.FALLBACK_SQL.strip().rstrip(";").lower()

    for idx, item in enumerate(test_data):
        key = expected_keys[idx]
        if key not in predict:
            continue
        value = predict[key]
        if not isinstance(value, str):
            errors.append(f"Key {key}: value is not a string")
            continue
        if delim not in value:
            bad_delim += 1
            continue
        sql, db_id = value.rsplit(delim, 1)
        if not sql.strip():
            empty_sql += 1
        elif sql.strip().rstrip(";").lower() == fallback_sql:
            sentinel_fill += 1
        if db_id != item.get("db_id"):
            db_mismatch += 1
            if db_mismatch <= 5:
                errors.append(
                    f"Key {key}: db_id mismatch (predict={db_id!r}, test={item.get('db_id')!r})"
                )

    if bad_delim:
        errors.append(
            f"{bad_delim} values missing delimiter {delim!r} "
            f"(expected format: SQL{delim}db_id)"
        )
    if empty_sql:
        errors.append(f"{empty_sql} values have empty SQL")
    if db_mismatch > 5:
        errors.append(f"{db_mismatch} total db_id mismatches (showing first 5 above)")
    elif db_mismatch and not any("db_id mismatch" in e for e in errors):
        errors.append(f"{db_mismatch} db_id mismatches")
    if sentinel_fill:
        warnings.append(
            f"{sentinel_fill} values are fallback sentinels ({config.FALLBACK_SQL!r} from --fill-missing); "
            "these score 0 EX. Re-run those questions if you want real predictions."
        )

    logger.info("Test questions: %d", len(test_data))
    logger.info("Predictions:    %d", len(predict))
    logger.info("Key scheme:     %s", args.key_by)
    if warnings:
        for w in warnings:
            logger.warning("⚠️  %s", w)
    if errors:
        logger.error("❌ Validation failed:")
        for e in errors:
            logger.error("  - %s", e)
        sys.exit(1)

    logger.info("✅ predict.json looks valid for BIRD submission")


if __name__ == "__main__":
    main()
