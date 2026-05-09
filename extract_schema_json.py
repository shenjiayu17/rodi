import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


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
    value_type: str
    is_primary_key: bool
    is_foreign_key: bool


@dataclass
class TableInfo:
    table_name_en: str
    table_name_ch: str
    db_name_en: str
    table_description: str
    columns: List[ColumnInfo]


def parse_sql_schema(sql_text: str) -> List[TableInfo]:
    default_schema = _infer_schema(sql_text)

    table_comment: Dict[Tuple[str, str], str] = {}
    column_comment: Dict[Tuple[str, str, str], str] = {}

    for m in re.finditer(
        r"COMMENT\s+ON\s+TABLE\s+(?P<tbl>[^\s]+)\s+IS\s+'(?P<txt>(?:[^']|'')*)'\s*;",
        sql_text,
        flags=re.IGNORECASE,
    ):
        schema, name = _parse_qualified_name(m.group('tbl'), default_schema)
        table_comment[(schema, name)] = m.group('txt').replace("''", "'")

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

    fk_cols: Dict[Tuple[str, str], Set[str]] = {}
    for m in re.finditer(
        r"ALTER\s+TABLE\s+ONLY\s+(?P<table>[^\s]+)\s+ADD\s+CONSTRAINT\s+[^\s]+\s+FOREIGN\s+KEY\s*\((?P<cols>[^\)]*)\)\s+REFERENCES\s+[^;]+;",
        sql_text,
        flags=re.IGNORECASE,
    ):
        schema, tbl = _parse_qualified_name(m.group('table'), default_schema)
        cols = [
            _strip_identifier(c.strip())
            for c in _split_top_level_commas(m.group('cols'))
            if c.strip()
        ]
        fk_cols.setdefault((schema, tbl), set()).update(cols)

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
                    value_type=col_type,
                    is_primary_key=False,
                    is_foreign_key=False,
                )
            )

        pkset = pk_cols.get((schema, tbl), set())
        fkset = fk_cols.get((schema, tbl), set())
        if pkset or fkset:
            for c in columns:
                if c.column_name_en in pkset:
                    c.is_primary_key = True
                if c.column_name_en in fkset:
                    c.is_foreign_key = True

        tables.append(
            TableInfo(
                table_name_en=tbl,
                table_name_ch='',
                db_name_en=schema,
                table_description=table_comment.get((schema, tbl), ''),
                columns=columns,
            )
        )

    return tables


def extract_from_directory(input_dir: str, project: str) -> Dict[str, object]:
    sql_files = sorted(glob.glob(os.path.join(input_dir, '*.sql')))
    all_tables: List[TableInfo] = []

    for fp in sql_files:
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            sql_text = f.read()
        all_tables.extend(parse_sql_schema(sql_text))

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
                        'value_type': c.value_type,
                        'is_primary_key': bool(c.is_primary_key),
                        'is_foreign_key': bool(c.is_foreign_key),
                    }
                    for c in t.columns
                ],
            }
        )

    return {'project': project, 'tables': out_tables}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--output', default='-')
    args = p.parse_args()

    data = extract_from_directory(args.input_dir, args.project)
    payload = json.dumps(data, ensure_ascii=False, indent=2)

    if args.output == '-' or args.output.lower() == 'stdout':
        print(payload)
    else:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(payload)


if __name__ == '__main__':
    main()
