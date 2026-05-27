import argparse
import glob
import json
import os
import re
import logging
import time
from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('extract_schema.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(
    r'^(?P<ident>"(?:[^"]|"")*"|[A-Za-z_][A-Za-z0-9_\-]*)'
)


def _strip_identifier(ident: str) -> str:
    ident = ident.strip()
    if ident.startswith('"') and ident.endswith('"'):
        inner = ident[1:-1]
        return inner.replace('""', '"')
    return ident


def _split_top_level_commas(s: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_single_quote = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            if in_single_quote:
                if i + 1 < len(s) and s[i + 1] == "'":
                    buf.append("''")
                    i += 2
                    continue
                in_single_quote = False
            else:
                in_single_quote = True
            buf.append(ch)
            i += 1
            continue

        if not in_single_quote:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth = max(0, depth - 1)
            elif ch == ',' and depth == 0:
                parts.append(''.join(buf).strip())
                buf = []
                i += 1
                continue

        buf.append(ch)
        i += 1

    tail = ''.join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_balanced_parentheses(text: str, open_paren_index: int) -> Tuple[str, int]:
    depth = 0
    in_single_quote = False
    i = open_paren_index
    if i >= len(text) or text[i] != '(':
        raise ValueError('open_paren_index must point to an opening parenthesis')

    while i < len(text):
        ch = text[i]
        if ch == "'":
            if in_single_quote:
                if i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                in_single_quote = False
            else:
                in_single_quote = True
            i += 1
            continue

        if not in_single_quote:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return text[open_paren_index + 1 : i], i + 1
        i += 1

    raise ValueError('unbalanced parentheses')


def _parse_qualified_name(raw: str, default_schema: str) -> Tuple[str, str]:
    raw = raw.strip()
    if raw.startswith('ONLY '):
        raw = raw[5:].strip()

    parts = [p.strip() for p in raw.split('.')]
    if len(parts) == 2:
        schema = _strip_identifier(parts[0])
        name = _strip_identifier(parts[1])
        return schema, name
    return default_schema, _strip_identifier(raw)


def _infer_schema(sql_text: str) -> str:
    m = re.search(r"SET\s+search_path\s*=\s*([^,;\n]+)", sql_text, flags=re.IGNORECASE)
    if m:
        return _strip_identifier(m.group(1).strip())

    m = re.search(r"CREATE\s+SCHEMA\s+([^;\n]+)", sql_text, flags=re.IGNORECASE)
    if m:
        return _strip_identifier(m.group(1).strip())

    m = re.search(r"DROP\s+SCHEMA\s+IF\s+EXISTS\s+([^\s;\n]+)", sql_text, flags=re.IGNORECASE)
    if m:
        return _strip_identifier(m.group(1).strip())

    return ''


def _parse_type(type_and_constraints: str) -> str:
    s = type_and_constraints.strip()
    if not s:
        return ''

    keywords = {
        'not',
        'null',
        'default',
        'constraint',
        'primary',
        'unique',
        'references',
        'check',
        'collate',
        'generated',
        'identity',
    }

    tokens: List[str] = []
    depth = 0
    cur: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace() and depth == 0:
            if cur:
                tokens.append(''.join(cur))
                cur = []
            i += 1
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        cur.append(ch)
        i += 1
    if cur:
        tokens.append(''.join(cur))

    type_tokens: List[str] = []
    for t in tokens:
        if t.lower() in keywords:
            break
        type_tokens.append(t)

    return ' '.join(type_tokens).strip()


@dataclass
class ColumnInfo:
    column_name_en: str
    column_name_ch: str
    column_description: str
    value_type: str
    is_primary_key: bool
    is_foreign_key: bool
    foreign_key: Optional[Dict[str, str]] = None


@dataclass
class TableInfo:
    table_name_en: str
    table_name_ch: str
    db_name_en: str
    table_description: str
    columns: List[ColumnInfo]


def parse_sql_schema(sql_text: str) -> List[TableInfo]:
    logger.info('Parsing SQL schema (len=%d chars)', len(sql_text))
    default_schema = _infer_schema(sql_text)

    ddl_snippets: List[str] = []

    table_comment: Dict[Tuple[str, str], str] = {}
    column_comment: Dict[Tuple[str, str, str], str] = {}

    for m in re.finditer(
        r"COMMENT\s+ON\s+TABLE\s+(?P<tbl>[^\s]+)\s+IS\s+'(?P<txt>(?:[^']|'')*)'\s*;",
        sql_text,
        flags=re.IGNORECASE,
    ):
        schema, name = _parse_qualified_name(m.group('tbl'), default_schema)
        table_comment[(schema, name)] = m.group('txt').replace("''", "'")
        ddl_snippets.append(m.group(0).strip())

    for m in re.finditer(
        r"COMMENT\s+ON\s+COLUMN\s+(?P<tblcol>[^\s]+)\s+IS\s+'(?P<txt>(?:[^']|'')*)'\s*;",
        sql_text,
        flags=re.IGNORECASE,
    ):
        tblcol = m.group('tblcol')
        if '.' in tblcol:
            left, col = tblcol.rsplit('.', 1)
            schema, tbl = _parse_qualified_name(left, default_schema)
            column_comment[(schema, tbl, _strip_identifier(col))] = m.group('txt').replace("''", "'")
            ddl_snippets.append(m.group(0).strip())

    pk_cols: Dict[Tuple[str, str], Set[str]] = {}
    for m in re.finditer(
        r"ALTER\s+TABLE\s+ONLY\s+(?P<table>[^\s]+)\s+ADD\s+CONSTRAINT\s+[^\s]+\s+PRIMARY\s+KEY\s*\((?P<cols>[^\)]*)\)\s*;",
        sql_text,
        flags=re.IGNORECASE,
    ):
        schema, tbl = _parse_qualified_name(m.group('table'), default_schema)
        cols = [
            _strip_identifier(c.strip())
            for c in _split_top_level_commas(m.group('cols'))
            if c.strip()
        ]
        pk_cols.setdefault((schema, tbl), set()).update(cols)
        ddl_snippets.append(m.group(0).strip())

    fk_targets: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for m in re.finditer(
        r"ALTER\s+TABLE\s+ONLY\s+(?P<table>[^\s]+)\s+ADD\s+CONSTRAINT\s+[^\s]+\s+FOREIGN\s+KEY\s*\((?P<src_cols>[^\)]*)\)\s+REFERENCES\s+(?P<target_table>(?:ONLY\s+)?[^\s(]+)\s*\((?P<target_cols>[^\)]*)\)(?:[^;]*)\s*;",
        sql_text,
        flags=re.IGNORECASE,
    ):
        schema, tbl = _parse_qualified_name(m.group('table'), default_schema)
        src_cols = [
            _strip_identifier(c.strip())
            for c in _split_top_level_commas(m.group('src_cols'))
            if c.strip()
        ]
        target_schema, target_tbl = _parse_qualified_name(m.group('target_table'), default_schema)
        target_table_name = target_tbl if target_schema == default_schema else f'{target_schema}.{target_tbl}'
        target_cols = [
            _strip_identifier(c.strip())
            for c in _split_top_level_commas(m.group('target_cols'))
            if c.strip()
        ]
        if len(src_cols) != len(target_cols):
            logger.warning(
                'Foreign key column count mismatch for %s.%s: source=%s target=%s',
                schema,
                tbl,
                src_cols,
                target_cols,
            )
        for src_col, target_col in zip(src_cols, target_cols):
            key = (schema, tbl, src_col)
            target = {'target_table': target_table_name, 'target_column': target_col}
            existing_target = fk_targets.get(key)
            if existing_target and existing_target != target:
                logger.warning(
                    'Multiple foreign keys found for %s.%s.%s; keeping first target=%s and ignoring target=%s',
                    schema,
                    tbl,
                    src_col,
                    existing_target,
                    target,
                )
                continue
            fk_targets[key] = target
        ddl_snippets.append(m.group(0).strip())

    index_cols: Dict[Tuple[str, str], Set[str]] = {}
    for m in re.finditer(
        r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+[^\s]+\s+ON\s+(?P<table>[^\s]+)\s*(?:USING\s+\w+\s*)?\((?P<cols>[^\)]*)\)\s*;",
        sql_text,
        flags=re.IGNORECASE,
    ):
        schema, tbl = _parse_qualified_name(m.group('table'), default_schema)
        cols = [
            _strip_identifier(c.strip())
            for c in _split_top_level_commas(m.group('cols'))
            if c.strip()
        ]
        index_cols.setdefault((schema, tbl), set()).update(cols)
        ddl_snippets.append(m.group(0).strip())

    tables: List[TableInfo] = []

    i = 0
    upper = sql_text.upper()
    while True:
        pos = upper.find('CREATE TABLE', i)
        if pos < 0:
            break

        j = pos + len('CREATE TABLE')
        while j < len(sql_text) and sql_text[j].isspace():
            j += 1

        if upper.startswith('IF NOT EXISTS', j):
            j += len('IF NOT EXISTS')
            while j < len(sql_text) and sql_text[j].isspace():
                j += 1

        m = _IDENTIFIER_RE.match(sql_text[j:])
        if not m:
            i = j
            continue
        raw_table_ident = m.group('ident')
        j += len(raw_table_ident)

        schema, tbl = _parse_qualified_name(raw_table_ident, default_schema)

        while j < len(sql_text) and sql_text[j].isspace():
            j += 1
        if j >= len(sql_text) or sql_text[j] != '(':
            i = j
            continue

        cols_block, after_paren = _extract_balanced_parentheses(sql_text, j)
        i = after_paren

        # best-effort: move past trailing ");" to avoid pathological loops
        semi = sql_text.find(';', i)
        if semi != -1:
            i = semi + 1

        ddl_snippets.append(f'CREATE TABLE {raw_table_ident} (\n{cols_block}\n);')

        col_items = _split_top_level_commas(cols_block)
        columns: List[ColumnInfo] = []

        for item in col_items:
            item_stripped = item.strip()
            if not item_stripped:
                continue

            head = item_stripped.split(None, 1)[0]
            head_l = head.strip('"').lower()
            if head_l in {'constraint', 'primary', 'foreign', 'unique', 'check'}:
                continue

            mcol = _IDENTIFIER_RE.match(item_stripped)
            if not mcol:
                continue
            raw_col = mcol.group('ident')
            col_name = _strip_identifier(raw_col)
            rest = item_stripped[len(raw_col) :].strip()
            col_type = _parse_type(rest)

            cmt = column_comment.get((schema, tbl, col_name), '')

            columns.append(
                ColumnInfo(
                    column_name_en=col_name,
                    column_name_ch=cmt,
                    column_description='',
                    value_type=col_type,
                    is_primary_key=False,
                    is_foreign_key=False,
                    foreign_key={},
                )
            )

        pkset = pk_cols.get((schema, tbl), set())
        if pkset or fk_targets:
            for c in columns:
                if c.column_name_en in pkset:
                    c.is_primary_key = True
                fk_target = fk_targets.get((schema, tbl, c.column_name_en))
                if fk_target:
                    c.is_foreign_key = True
                    c.foreign_key = fk_target

        tables.append(
            TableInfo(
                table_name_en=tbl,
                table_name_ch='',
                db_name_en=schema,
                table_description=table_comment.get((schema, tbl), ''),
                columns=columns,
            )
        )

    # Attach DDL-only context for downstream LLM enrichment
    parse_sql_schema.ddl_context = '\n\n'.join(ddl_snippets)  # type: ignore[attr-defined]
    logger.info('Parsed %d tables; collected %d DDL snippets for LLM context', len(tables), len(ddl_snippets))
    return tables


def get_last_ddl_context() -> str:
    return str(getattr(parse_sql_schema, 'ddl_context', ''))


def _merge_llm_enrichment(schema: Dict[str, object], enrich: Dict[str, object]) -> None:
    logger.info('Merging LLM enrichment into extracted schema')
    tables = schema.get('tables')
    enrich_tables = enrich.get('tables') if isinstance(enrich, dict) else None
    if not isinstance(tables, list) or not isinstance(enrich_tables, list):
        return

    enrich_by_table: Dict[str, Dict[str, object]] = {}
    for t in enrich_tables:
        if isinstance(t, dict) and isinstance(t.get('table_name_en'), str):
            enrich_by_table[t['table_name_en']] = t

    for t in tables:
        if not isinstance(t, dict):
            continue
        ten = t.get('table_name_en')
        if not isinstance(ten, str):
            continue
        et = enrich_by_table.get(ten)
        if not et:
            continue

        if isinstance(et.get('table_name_ch'), str):
            t['table_name_ch'] = et.get('table_name_ch') or ''
        if isinstance(et.get('table_description'), str):
            t['table_description'] = et.get('table_description') or ''

        cols = t.get('columns')
        ecols = et.get('columns')
        if not isinstance(cols, list) or not isinstance(ecols, list):
            continue

        ecols_by_name: Dict[str, Dict[str, object]] = {}
        for c in ecols:
            if isinstance(c, dict) and isinstance(c.get('column_name_en'), str):
                ecols_by_name[c['column_name_en']] = c

        for c in cols:
            if not isinstance(c, dict):
                continue
            cen = c.get('column_name_en')
            if not isinstance(cen, str):
                continue
            ec = ecols_by_name.get(cen)
            if not ec:
                continue
            if isinstance(ec.get('column_name_ch'), str):
                c['column_name_ch'] = ec.get('column_name_ch') or ''
            if isinstance(ec.get('column_description'), str):
                c['column_description'] = ec.get('column_description') or ''


def _table_needs_llm_enrichment(table: Dict[str, object]) -> bool:
    table_name_ch = table.get('table_name_ch')
    table_description = table.get('table_description')
    return not (isinstance(table_name_ch, str) and table_name_ch.strip()) or not (
        isinstance(table_description, str) and table_description.strip()
    )


def _load_existing_output(output_path: str, project: str) -> Optional[Dict[str, object]]:
    if not output_path or output_path in {'-', 'stdout'}:
        return None
    if not os.path.exists(output_path):
        logger.info('Resume requested but output file does not exist: %s', output_path)
        return None

    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning('Failed to load existing output for resume (%s): %s', output_path, e)
        return None

    if not isinstance(data, dict):
        logger.warning('Existing output is not a JSON object, ignoring resume file: %s', output_path)
        return None

    existing_project = data.get('project')
    if isinstance(existing_project, str) and existing_project and existing_project != project:
        logger.warning(
            'Existing output project mismatch (existing=%s, current=%s), ignoring resume file: %s',
            existing_project,
            project,
            output_path,
        )
        return None

    return data


def _merge_existing_output_into_result(result: Dict[str, object], existing: Dict[str, object]) -> int:
    result_tables = result.get('tables')
    existing_tables = existing.get('tables')
    if not isinstance(result_tables, list) or not isinstance(existing_tables, list):
        return 0

    existing_by_name: Dict[str, Dict[str, object]] = {}
    for table in existing_tables:
        if isinstance(table, dict) and isinstance(table.get('table_name_en'), str):
            existing_by_name[table['table_name_en']] = table

    merged = 0
    for table in result_tables:
        if not isinstance(table, dict):
            continue
        table_name_en = table.get('table_name_en')
        if not isinstance(table_name_en, str):
            continue
        existing_table = existing_by_name.get(table_name_en)
        if existing_table:
            if isinstance(existing_table.get('table_name_ch'), str) and existing_table['table_name_ch'].strip():
                table['table_name_ch'] = existing_table['table_name_ch']
            if isinstance(existing_table.get('table_description'), str) and existing_table['table_description'].strip():
                table['table_description'] = existing_table['table_description']

            result_cols = table.get('columns')
            existing_cols = existing_table.get('columns')
            if isinstance(result_cols, list) and isinstance(existing_cols, list):
                existing_by_name: Dict[str, Dict[str, object]] = {}
                for col in existing_cols:
                    if isinstance(col, dict) and isinstance(col.get('column_name_en'), str):
                        existing_by_name[col['column_name_en']] = col

                for col in result_cols:
                    if not isinstance(col, dict):
                        continue
                    col_name_en = col.get('column_name_en')
                    if not isinstance(col_name_en, str):
                        continue
                    existing_col = existing_by_name.get(col_name_en)
                    if not existing_col:
                        continue
                    if isinstance(existing_col.get('column_name_ch'), str) and existing_col['column_name_ch'].strip():
                        col['column_name_ch'] = existing_col['column_name_ch']
                    if isinstance(existing_col.get('column_description'), str) and existing_col['column_description'].strip():
                        col['column_description'] = existing_col['column_description']
                    if col.get('foreign_key') is None and existing_col.get('foreign_key') is not None:
                        col['foreign_key'] = existing_col['foreign_key']

            merged += 1

    return merged


def enrich_schema_with_llm(
    *,
    ddl_sql_text: str,
    schema: Dict[str, object],
) -> None:
    from llm_client.llm_client import LlmClient
    from llm_client import config as llm_config

    from prompts.prompts_template import SCHEMA_ENRICH_PROMPT

    prompt_template = SCHEMA_ENRICH_PROMPT

    logger.info('Preparing LLM prompt (ddl_sql_text_len=%d chars)', len(ddl_sql_text))
    llm_input = {
        'sql': ddl_sql_text,
        'extracted': schema,
    }
    prompt = prompt_template + "\n\n" + json.dumps(llm_input, ensure_ascii=False, indent=2)

    client = LlmClient()
    json_retries = int(getattr(llm_config, 'LLM_JSON_RETRIES', 0))
    attempt_total = json_retries + 1

    last_error: Optional[BaseException] = None
    for attempt in range(1, attempt_total + 1):
        try:
            logger.info('Calling LLM to enrich schema... (json_attempt %d/%d)', attempt, attempt_total)
            raw = client.chat(prompt)
            logger.info('LLM responded (raw_len=%d chars)', len(raw))
            data = json.loads(raw)
            _merge_llm_enrichment(schema, data)
            logger.info('LLM enrichment merged successfully')
            return
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                'LLM returned invalid JSON (attempt %d/%d): %s',
                attempt,
                attempt_total,
                e,
            )
            if attempt < attempt_total:
                continue
        except Exception:
            raise

    raise RuntimeError(f'LLM returned invalid JSON after {attempt_total} attempts: {last_error}')


def enrich_schema_with_llm_chunked(
    *,
    ddl_sql_text: str,
    schema: Dict[str, object],
    chunk_size: int,
    target_table_names: Optional[Set[str]] = None,
) -> None:
    tables = schema.get('tables')
    if not isinstance(tables, list):
        raise ValueError('schema.tables must be a list')

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError('chunk_size must be > 0')

    if target_table_names:
        target_tables = [t for t in tables if isinstance(t, dict) and t.get('table_name_en') in target_table_names]
    else:
        target_tables = [t for t in tables if isinstance(t, dict)]

    total = len(target_tables)
    if total == 0:
        logger.info('No tables found, skipping LLM enrichment')
        return

    logger.info('Starting chunked LLM enrichment: total_tables=%d, chunk_size=%d', total, chunk_size)

    start_all = time.perf_counter()
    chunk_index = 0
    for start in range(0, total, chunk_size):
        chunk_index += 1
        end = min(total, start + chunk_size)
        chunk_tables = target_tables[start:end]

        sub_schema: Dict[str, object] = {
            'project': schema.get('project', ''),
            'tables': chunk_tables,
        }

        logger.info('LLM chunk %d: tables [%d, %d) (%d tables) %s', chunk_index, start, end, len(chunk_tables), chunk_tables)
        t0 = time.perf_counter()
        enrich_schema_with_llm(ddl_sql_text=ddl_sql_text, schema=sub_schema)
        dt = time.perf_counter() - t0

        logger.info('LLM chunk %d finished in %.2fs', chunk_index, dt)

    logger.info('Chunked LLM enrichment completed in %.2fs', time.perf_counter() - start_all)


def extract_from_directory(
    input_dir: str,
    project: str,
    *,
    use_llm: bool,
    llm_chunk_size: int,
    resume: bool,
    output_path: str,
) -> Dict[str, object]:
    logger.info(f"\n================================= START =================================")
    logger.info('Extracting schema from directory: %s', input_dir)
    sql_files = sorted(glob.glob(os.path.join(input_dir, '*.sql')))
    logger.info('Found %d SQL files', len(sql_files))
    all_tables: List[TableInfo] = []
    ddl_contexts: List[str] = []

    for fp in sql_files:
        logger.info('Reading SQL file: %s', fp)
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            sql_text = f.read()
        file_tables = parse_sql_schema(sql_text)
        all_tables.extend(file_tables)
        ddl_context = get_last_ddl_context()
        if ddl_context:
            ddl_contexts.append(ddl_context)

    out_tables = []
    for t in all_tables:
        out_tables.append(
            {
                'table_name_en': t.table_name_en,
                'table_name_ch': t.table_name_ch,
                'db_name_en': t.db_name_en,
                'table_description': t.table_description,
                'columns': [
                    {
                        'column_name_en': c.column_name_en,
                        'column_name_ch': c.column_name_ch,
                        'column_description': c.column_description,
                        'value_type': c.value_type,
                        'is_primary_key': bool(c.is_primary_key),
                        'is_foreign_key': bool(c.is_foreign_key),
                        'foreign_key': c.foreign_key,
                    }
                    for c in t.columns
                ],
            }
        )

    result: Dict[str, object] = {'project': project, 'tables': out_tables}
    logger.info('Base extraction complete: %d tables', len(out_tables))

    existing_output = _load_existing_output(output_path, project) if resume else None
    if existing_output:
        merged_count = _merge_existing_output_into_result(result, existing_output)
        logger.info('Resume mode enabled: merged %d tables from existing output', merged_count)

    if use_llm:
        try:
            target_table_names: Optional[Set[str]] = None
            if existing_output:
                tables_to_enrich = [t for t in result['tables'] if isinstance(t, dict) and _table_needs_llm_enrichment(t)]
                target_table_names = {
                    t['table_name_en']
                    for t in tables_to_enrich
                    if isinstance(t.get('table_name_en'), str)
                }
                logger.info(
                    'Resume mode: %d/%d tables (%s) need LLM enrichment',
                    len(target_table_names),
                    len(result['tables']) if isinstance(result.get('tables'), list) else 0,
                    target_table_names,
                )
                if not target_table_names:
                    logger.info('Resume mode: no tables require LLM enrichment, skipping LLM step')
                    logger.info('Extraction finished')
                    return result

            enrich_schema_with_llm_chunked(
                ddl_sql_text='\n\n'.join(ddl_contexts),
                schema=result,
                chunk_size=int(llm_chunk_size),
                target_table_names=target_table_names,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception('LLM enrichment failed: %s', e)

    logger.info('Extraction finished')

    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True, help="输入目录，包含dump.sql原始文件")
    p.add_argument('--project', required=True, help="输出schema.json中的project字段值")
    p.add_argument('--output', default='.\\output.json', help="输出文件路径，默认为当前目录下的output.json")
    p.add_argument('--use-llm', action='store_true', help="是否使用LLM进行增强")
    p.add_argument('--llm-chunk-size', type=int, default=10, help="提交给LLM的DDL块大小（表数量），默认每次10个表")
    p.add_argument('--resume', action='store_true', help="如果输出文件已存在，则读取并只对缺失表中文名/表描述的表继续LLM增强")
    args = p.parse_args()

    data = extract_from_directory(
        args.input_dir,
        args.project,
        use_llm=bool(args.use_llm),
        llm_chunk_size=int(args.llm_chunk_size),
        resume=bool(args.resume),
        output_path=str(args.output),
    )
    payload = json.dumps(data, ensure_ascii=False, indent=2)

    if args.output == '-' or args.output.lower() == 'stdout':
        print(payload)
    else:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(payload)


if __name__ == '__main__':
    main()
