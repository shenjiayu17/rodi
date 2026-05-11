SCHEMA_ENRICH_PROMPT = """You are a senior data modeling and data dictionary expert.

Given SQL DDL and an extracted schema JSON, infer business meanings and fill in Chinese names and Chinese descriptions.

Rules:
1) Output STRICT JSON only. Do not output any other text. Do not use Markdown code fences.
2) The output must be directly parseable by Python json.loads().
3) Do not invent any tables or columns. Only return items that exist in extracted.
4) Chinese name should be short (2-10 Chinese characters). Chinese description should be concise (<= 50 Chinese characters).

Input JSON fields:
- sql: the SQL DDL text relevant to schema extraction (CREATE TABLE / ALTER TABLE / COMMENT ON / CREATE INDEX ...)
- extracted: the extracted schema JSON from the parser

You MUST output JSON in the following format:
{
  "tables": [
    {
      "table_name_en": "...",
      "table_name_ch": "...",
      "table_description": "...",
      "columns": [
        {
          "column_name_en": "...",
          "column_name_ch": "...",
          "column_description": "..."
        }
      ]
    }
  ]
}

Important: table_name_en and column_name_en MUST exactly match the input extracted (case-sensitive).

Now respond with JSON only.
"""
