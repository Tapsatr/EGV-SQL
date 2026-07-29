import os
import sys
import json
import sqlite3
import csv
import re
import argparse
from pathlib import Path

from utils.logging_setup import (
    setup_logging,
    get_logger,
    resolve_log_path,
    sibling_log_path,
)

logger = get_logger(__name__)

# --- Configuration ---
NUM_SAMPLE_VALUES = 3
MAX_VALUE_LENGTH = 75
HEADER_MAX_CHARS = 700
HEADER_VAL_HINT_MAX = 70
INLINE_COL_DESC_MAX = 120
INLINE_VAL_DESC_MAX = 180
LARGE_TABLE_THRESHOLD = 50000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate_value(value, max_len):
    s_val = str(value).strip()
    if len(s_val) > max_len:
        return s_val[:max_len] + "..."
    return s_val

def _clean_value_description(val_desc, max_length=INLINE_VAL_DESC_MAX):
    if not val_desc:
        return ""
    val_desc = re.sub(r'\s+', ' ', val_desc).strip()
    val_desc = re.sub(r'^commonsense (?:evidence|reasoning):\s*', '', val_desc, flags=re.IGNORECASE).strip()
    if len(val_desc) > max_length:
        val_desc = val_desc[:max_length] + "..."
    return val_desc

def _resolve_english_label(col_name, meta):
    if not meta:
        return ""
    for candidate in (
        meta.get('column_name', ''),
        meta.get('column_description', ''),
        col_name,
    ):
        text = (candidate or '').strip()
        if not text:
            continue
        if text.lower() == col_name.lower():
            continue
        return text
    return ""

def _find_csv_for_table(db_path, table_name):
    desc_dir = db_path / "database_description"
    if not desc_dir.is_dir():
        return None
    exact = desc_dir / f"{table_name}.csv"
    if exact.exists(): return exact
    table_lower = table_name.lower()
    for f in desc_dir.iterdir():
        if f.suffix.lower() == '.csv' and f.stem.lower() == table_lower: return f
    for f in desc_dir.iterdir():
        if f.suffix.lower() == '.csv':
            stem_lower = f.stem.lower()
            if table_lower in stem_lower or stem_lower in table_lower: return f
    return None

def normalize_column_meaning(raw):
    """
    Normalize column_meaning.json into nested: db_id -> table -> col -> description.

    Supports:
      - TA-SQL flat keys: "database_id|table_name|column_name" -> description
      - Nested dicts: db_id -> table_name -> {col: description}
    """
    if not raw or not isinstance(raw, dict):
        return None

    sample_key = next(iter(raw.keys()))
    if isinstance(sample_key, str) and sample_key.count("|") >= 2:
        nested = {}
        for key, desc in raw.items():
            parts = str(key).split("|")
            if len(parts) < 3:
                continue
            db_id, table_name, col_name = parts[0], parts[1], "|".join(parts[2:])
            nested.setdefault(db_id, {}).setdefault(table_name, {})[col_name] = desc
        return nested

    return raw


def _table_lookup(db_data, table_name):
    if not isinstance(db_data, dict):
        return {}
    if table_name in db_data:
        return db_data[table_name]
    table_lower = table_name.lower()
    for key, value in db_data.items():
        if isinstance(key, str) and key.lower() == table_lower:
            return value
    return {}


def load_column_descriptions(db_id, db_path, table_name, column_meaning_data):
    # Prefer column_meaning.json when provided; fall back to BIRD CSV metadata.
    if column_meaning_data:
        db_data = column_meaning_data.get(db_id, {})
        if not db_data:
            # Case-insensitive db_id match
            for key, value in column_meaning_data.items():
                if isinstance(key, str) and key.lower() == db_id.lower():
                    db_data = value
                    break
        table_data = _table_lookup(db_data, table_name)
        if table_data and isinstance(table_data, dict):
            descriptions = {}
            for col, desc in table_data.items():
                descriptions[col] = {
                    'column_name': '',
                    'column_description': desc if isinstance(desc, str) else str(desc),
                    'value_description': ''
                }
            if descriptions:
                return descriptions

    # Fallback to BIRD CSV metadata if available in the directory
    csv_file = _find_csv_for_table(db_path, table_name)
    if csv_file is None:
        return {}

    descriptions = {}
    for encoding in ('utf-8-sig', 'latin-1'):
        try:
            with open(csv_file, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    col_name = (row.get('original_column_name') or '').strip()
                    if not col_name: continue
                    english_name = (row.get('column_name') or '').strip()
                    col_desc = (row.get('column_description') or '').strip()
                    val_desc = (row.get('value_description') or '').strip()
                    if english_name or col_desc or val_desc:
                        descriptions[col_name] = {
                            'column_name': english_name,
                            'column_description': col_desc,
                            'value_description': val_desc,
                        }
            break
        except UnicodeDecodeError:
            descriptions.clear()
            continue
        except Exception as e:
            logger.warning("  ⚠️  Could not read %s: %s", csv_file.name, e)
            break

    return descriptions

def _lookup_description(col_name, col_descriptions):
    if col_name in col_descriptions: return col_descriptions[col_name]
    col_lower = col_name.lower()
    for key, value in col_descriptions.items():
        if key.lower() == col_lower: return value
    return None

_CONSTRAINT_KEYWORDS = frozenset({'primary', 'foreign', 'constraint', 'unique', 'check', 'create', 'table'})
_COL_RE = re.compile(r'^\s*(?:`([^`]+)`|"([^"]+)"|\[([^\]]+)\]|(\w+))')
_FK_RE = re.compile(r'foreign\s+key\s*\(\s*["`]?([^)"`]+)["`]?\s*\)', re.IGNORECASE)

def _extract_ddl_column_names(create_sql):
    names = []
    for line in create_sql.split('\n'):
        m = _COL_RE.match(line)
        if not m: continue
        col_name = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        if col_name and col_name.lower() not in _CONSTRAINT_KEYWORDS:
            names.append(col_name)
    return names

def _get_pk_columns(cursor, table_name):
    try:
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        return {row[1] for row in cursor.fetchall() if row[5]}
    except Exception:
        return set()

def _get_fk_columns(create_sql):
    fk_cols = set()
    for m in _FK_RE.finditer(create_sql):
        raw = m.group(1)
        for part in raw.split(','):
            col = part.strip().strip('"`[]')
            if col: fk_cols.add(col)
    return fk_cols

_FK_FULL_RE = re.compile(r'foreign\s+key\s*\(\s*([^)]+)\s*\)\s*references\s+["`\[]?(\w+)["`\]]?', re.IGNORECASE)

def _get_fk_references(create_sql):
    fk_refs = {}
    for m in _FK_FULL_RE.finditer(create_sql):
        raw_cols = m.group(1)
        ref_table = m.group(2)
        for part in raw_cols.split(','):
            col = part.strip().strip('"`[]')
            if col: fk_refs[col] = ref_table
    return fk_refs

def _header_priority(col_name, meta, pk_cols, fk_cols):
    score = 0
    if col_name in pk_cols: score += 1000
    if col_name in fk_cols: score += 500
    if meta:
        english = _resolve_english_label(col_name, meta)
        if english: score += 100
        if (meta.get('value_description') or '').strip(): score += 50
        col_desc = (meta.get('column_description') or '').strip()
        if col_desc and col_desc.lower() != col_name.lower(): score += 25
    return score

def _format_header_entry(col_name, meta, fk_refs=None):
    fk_arrow = ""
    if fk_refs and col_name in fk_refs:
        fk_arrow = f" → {fk_refs[col_name]}"
    if not meta:
        return f"{col_name}{fk_arrow}" if fk_arrow else col_name
    english = _resolve_english_label(col_name, meta)
    if english:
        return f"{col_name}{fk_arrow}: {english}"
    val_hint = _clean_value_description(meta.get('value_description', ''), max_length=HEADER_VAL_HINT_MAX)
    if val_hint:
        return f"{col_name}{fk_arrow}: {val_hint}"
    return f"{col_name}{fk_arrow}" if fk_arrow else col_name

def generate_table_header(table_name, col_descriptions, ddl_col_order, pk_cols, fk_cols, fk_refs=None):
    if fk_refs is None: fk_refs = {}
    prefix = f"/* Table: {table_name} | "
    suffix = " */"
    budget = HEADER_MAX_CHARS - len(prefix) - len(suffix)
    if budget <= 0: return prefix.rstrip(" | ") + suffix

    indexed_cols = list(enumerate(ddl_col_order))
    indexed_cols.sort(key=lambda item: (-_header_priority(item[1], _lookup_description(item[1], col_descriptions), pk_cols, fk_cols), item[0]))

    entries = []
    used = set()
    for _, col_name in indexed_cols:
        if col_name in used: continue
        meta = _lookup_description(col_name, col_descriptions)
        if not meta and col_name not in pk_cols and col_name not in fk_cols: continue
        entry = _format_header_entry(col_name, meta, fk_refs)
        sep = "; " if entries else ""
        if len(sep) + len(entry) > budget: break
        entries.append(entry)
        used.add(col_name)
        budget -= len(sep) + len(entry)

    for col_name in ddl_col_order:
        if col_name in used: continue
        entry = col_name
        sep = "; " if entries else ""
        if len(sep) + len(entry) > budget: break
        entries.append(entry)
        used.add(col_name)
        budget -= len(sep) + len(entry)

    if not entries: return f"/* Table: {table_name} */"
    return prefix + "; ".join(entries) + suffix

def inject_column_descriptions(create_sql, col_descriptions):
    if not col_descriptions: return create_sql, 0
    lines = create_sql.split('\n')
    new_lines = []
    annotated = 0
    for line in lines:
        m = _COL_RE.match(line)
        if m:
            col_name = m.group(1) or m.group(2) or m.group(3) or m.group(4)
            if col_name and col_name.lower() not in _CONSTRAINT_KEYWORDS:
                meta = _lookup_description(col_name, col_descriptions)
                if meta:
                    parts = []
                    english = _resolve_english_label(col_name, meta)
                    if english:
                        parts.append(truncate_value(english, INLINE_COL_DESC_MAX))
                    else:
                        col_desc = (meta.get('column_description') or '').strip()
                        if col_desc and col_desc.lower() != col_name.lower():
                            parts.append(truncate_value(col_desc, INLINE_COL_DESC_MAX))
                    val_desc = _clean_value_description(meta.get('value_description', ''))
                    if val_desc:
                        parts.append(f"Values: {val_desc}")
                    if parts:
                        comment = " | ".join(parts)
                        line = f"{line.rstrip()} -- {comment}"
                        annotated += 1
        new_lines.append(line)
    return '\n'.join(new_lines), annotated

def get_sample_values(cursor, table_name, num_samples, max_value_length, row_count=None):
    sample_lines = []
    large_table = row_count is not None and row_count > LARGE_TABLE_THRESHOLD
    try:
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        columns = cursor.fetchall()
        for row in columns:
            col_name, col_type = row[1], (row[2] or '').upper()
            try:
                is_numeric = any(t in col_type for t in ('INT', 'REAL', 'NUM', 'FLOAT', 'DOUBLE', 'DECIMAL'))
                if is_numeric and not large_table:
                    cursor.execute(f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM "{table_name}" WHERE "{col_name}" IS NOT NULL;')
                    min_max = cursor.fetchone()
                    if min_max and min_max[0] is not None:
                        vals = [min_max[0]]
                        cursor.execute(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL ORDER BY "{col_name}" LIMIT {max(1, num_samples - 2)} OFFSET (SELECT COUNT(DISTINCT "{col_name}") / 2 FROM "{table_name}" WHERE "{col_name}" IS NOT NULL);')
                        mids = [r[0] for r in cursor.fetchall() if r[0] is not None]
                        vals.extend(mids)
                        if min_max[1] is not None and min_max[1] != min_max[0]: vals.append(min_max[1])
                        seen = set()
                        unique_vals = []
                        for v in vals:
                            if v not in seen:
                                seen.add(v)
                                unique_vals.append(v)
                        safe_values = [truncate_value(v, max_value_length) for v in unique_vals[:num_samples]]
                    else:
                        safe_values = []
                elif is_numeric:
                    cursor.execute(f'SELECT MIN("{col_name}"), MAX("{col_name}") FROM "{table_name}" WHERE "{col_name}" IS NOT NULL;')
                    min_max = cursor.fetchone()
                    if min_max and min_max[0] is not None:
                        vals = [min_max[0]]
                        if min_max[1] is not None and min_max[1] != min_max[0]: vals.append(min_max[1])
                        safe_values = [truncate_value(v, max_value_length) for v in vals]
                    else:
                        safe_values = []
                else:
                    query = f'SELECT DISTINCT "{col_name}" FROM "{table_name}" WHERE "{col_name}" IS NOT NULL AND "{col_name}" != "" LIMIT {num_samples};'
                    cursor.execute(query)
                    samples = cursor.fetchall()
                    safe_values = [truncate_value(s[0], max_value_length) for s in samples if s[0] is not None]

                if safe_values:
                    vals_str = ", ".join(safe_values)
                    sample_lines.append(f"-- Sample values for column `{col_name}`: {vals_str}")
            except Exception:
                continue
    except Exception:
        pass
    return sample_lines

def build_chunk_content(create_sql, table_name, cursor, db_id, db_path, num_samples, column_meaning_data):
    col_descriptions = load_column_descriptions(db_id, db_path, table_name, column_meaning_data)
    ddl_col_order = _extract_ddl_column_names(create_sql)
    pk_cols = _get_pk_columns(cursor, table_name)
    fk_cols = _get_fk_columns(create_sql)
    fk_refs = _get_fk_references(create_sql)

    row_count = None
    try:
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
        row_count = cursor.fetchone()[0]
    except Exception: pass

    header = generate_table_header(table_name, col_descriptions, ddl_col_order, pk_cols, fk_cols, fk_refs)
    annotated_ddl, n_annotated = inject_column_descriptions(create_sql, col_descriptions)
    sample_lines = get_sample_values(cursor, table_name, num_samples, MAX_VALUE_LENGTH, row_count)

    parts = [header, annotated_ddl]
    if row_count is not None:
        parts.append(f"-- Approximate rows: {row_count}")
    parts.extend(sample_lines)

    return "\n".join(parts), n_annotated, len(col_descriptions)

def parse_args():
    parser = argparse.ArgumentParser(description="Create schema chunks for BIRD database.")
    parser.add_argument("--db-dir", type=str, required=True, help="Path to the directory containing databases (e.g. test_databases/)")
    parser.add_argument("--output", type=str, required=True, help="Path to the output JSON chunk file.")
    parser.add_argument("--column-meaning", type=str, default=None, help="Path to column_meaning.json (optional, replaces CSV metadata)")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Rotating log file path (default: sibling of --output as {stem}.log).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    log_path = resolve_log_path(args.log_file, sibling_log_path(args.output, ".log"))
    setup_logging(log_path)

    logger.info(
        "=== RUN HEADER === db_dir=%s output=%s log=%s column_meaning=%s",
        args.db_dir,
        args.output,
        log_path,
        args.column_meaning,
    )

    db_dir = Path(args.db_dir)
    if not db_dir.exists():
        logger.error("❌ ERROR: Database directory not found at '%s'. exit_code=1", db_dir)
        sys.exit(1)

    column_meaning_data = None
    if args.column_meaning and Path(args.column_meaning).exists():
        logger.info("Loading column meaning from %s", args.column_meaning)
        with open(args.column_meaning, 'r', encoding='utf-8') as f:
            column_meaning_data = normalize_column_meaning(json.load(f))
        if column_meaning_data:
            logger.info("  Normalized column meanings for %d databases", len(column_meaning_data))
    elif args.column_meaning:
        logger.warning("⚠️ column_meaning path not found: %s (falling back to CSVs)", args.column_meaning)

    db_dirs = sorted(d for d in os.listdir(db_dir) if os.path.isdir(db_dir / d))
    all_chunks = []

    logger.info("Processing %d databases...", len(db_dirs))
    for db_id in db_dirs:
        db_path = db_dir / db_id
        db_file = db_path / f"{db_id}.sqlite"

        if not db_file.exists():
            logger.warning("⚠️ SQLite file for '%s' not found. Skipping.", db_id)
            continue

        conn = None
        try:
            conn = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table_name, create_sql in tables:
                if not create_sql: continue
                content, n_ann, _ = build_chunk_content(create_sql, table_name, cursor, db_id, db_path, NUM_SAMPLE_VALUES, column_meaning_data)
                all_chunks.append({
                    "db_id": db_id,
                    "table_name": table_name,
                    "content": content,
                })
        except Exception as e:
            logger.error("❌ ERROR processing database %s: %s", db_id, e)
        finally:
            if conn: conn.close()

    if not all_chunks:
        logger.error(
            "❌ ERROR: Produced 0 schema chunks from '%s'. Inference would run with "
            "empty schema context and generate garbage SQL. Check that each "
            "<db_id>/<db_id>.sqlite exists and is readable. exit_code=1",
            db_dir,
        )
        sys.exit(1)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)

    logger.info("✅ Chunks successfully written to %s (%d chunks total).", args.output, len(all_chunks))

if __name__ == "__main__":
    main()
