import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import duckdb
import gradio as gr
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET = os.environ.get("TG_DATASET", "Nischayydv/tg-dataset")
HF_FILE    = os.environ.get("TG_FILE", "merged_all.parquet")
HF_REVISION = os.environ.get("TG_REVISION", "main")

PARQUET_URL = (
    f"https://huggingface.co/datasets/{HF_DATASET}"
    f"/resolve/{HF_REVISION}/{HF_FILE}"
)

PARALLELISM     = int(os.environ.get("TG_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("TG_THREADS_PER_CONN", "2"))
DUPLICATE_CAP   = 2

SEARCH_FIELDS = ["user_id", "phone_number", "username", "country", "country_code"]

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: List[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # Only /tmp is writable on Vercel
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute(f"SET threads = {THREADS_PER_CONN}")

    # Expose the remote Parquet file as a view called `people`
    con.execute(
        f"CREATE OR REPLACE VIEW people AS "
        f"SELECT * FROM read_parquet('{PARQUET_URL}')"
    )
    return con


def _thread_id() -> int:
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns)
            _thread_local.id = tid
    return tid


def _get_conn() -> duckdb.DuckDBPyConnection:
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident:
            _conns.append(_new_conn())
    return _conns[ident]


# ── Helpers ─────────────────────────────────────────────────────────────────
def _escape(v: str) -> str:
    return v.replace("'", "''")


def _run_query(sql: str, limit: int) -> List[dict]:
    con = _get_conn()
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = [dict(zip(cols, r)) for r in rows]
    # Deduplicate by user_id, cap duplicates
    seen: Dict[Any, int] = {}
    out = []
    for r in results:
        key = (r.get("user_id"), r.get("phone_number"))
        n = seen.get(key, 0)
        if n < DUPLICATE_CAP:
            seen[key] = n + 1
            out.append(r)
    return out[:limit]


# ── Search Logic ────────────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = _escape(value)

    if mode == "exact":
        sql = f"SELECT * FROM people WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    elif mode == "contains":
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = (f"SELECT * FROM people WHERE {field} ILIKE '%{v2}%' "
               f"ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    results = _run_query(sql, limit)
    return {"field": field, "value": value, "mode": mode,
            "count": len(results), "results": results}


def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not q:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    v = _escape(q)
    # Detect type of input
    is_digits = q.isdigit()

    where_parts = [
        f"username ILIKE '%{v}%'",
        f"CAST(user_id AS VARCHAR) ILIKE '%{v}%'",
        f"CAST(phone_number AS VARCHAR) ILIKE '%{v}%'",
    ]
    if not is_digits:
        where_parts.append(f"country ILIKE '%{v}%'")

    where_clause = " OR ".join(where_parts)
    sql = f"SELECT * FROM people WHERE {where_clause} LIMIT {limit * DUPLICATE_CAP + 20}"

    results = _run_query(sql, limit)
    return {
        "query": q,
        "searched_fields": ["username", "user_id", "phone_number", "country"],
        "count": len(results),
        "results": results,
    }


# ── FastAPI (top-level instance — required by Vercel) ──────────────────────
app = FastAPI(title="TG Dataset Search API")


class BatchRequest(BaseModel):
    queries: List[dict]
    limit: int = 10


@app.get("/")
def root():
    return {
        "app": "TG Dataset Search API",
        "dataset": HF_DATASET,
        "file": HF_FILE,
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "ui": "/ui",
    }


@app.get("/health")
def health():
    try:
        con = _get_conn()
        n = con.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        return {"status": "ok", "rows": n}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/search")
async def search(
    q: str | None = Query(None),
    field: str | None = Query(None),
    mode: str = Query("contains"),
    limit: int = Query(10, ge=1, le=100),
    pretty: bool = Query(True),
):
    if not q or not q.strip():
        raise HTTPException(422, "Provide q")
    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(
            pool, _run_field_search, field, q.strip(), mode, limit
        )
    else:
        data = await loop.run_in_executor(pool, _unified_search, q.strip(), limit)

    result = {"success": bool(data["count"]), **data, "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@app.post("/search/parallel")
async def search_parallel(req: BatchRequest):
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool, _run_field_search,
            item.get("field", "username"),
            item.get("value", ""),
            item.get("mode", "contains"),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps(
            {"searches": len(req.queries), "results": list(results)},
            indent=2, ensure_ascii=False,
        ),
        media_type="application/json",
    )


# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")
    return "\n\n".join(lines)


def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Please enter a username, user ID, or phone number."
    q = query.strip()
    try:
        data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"

    count = data["count"]
    results = data["results"]
    searched = ", ".join(data.get("searched_fields", []))

    if not results:
        return (f"🔍 **Query:** `{q}`\n**Searched:** {searched}\n\n"
                f"❌ **No data found.**")

    header = (f"🔍 **Query:** `{q}`  |  **Found:** {count} result(s)  |  "
              f"**Searched:** {searched}\n\n---\n\n")
    parts = [f"### Result {i}\n{format_result(row)}"
             for i, row in enumerate(results, 1)]
    return header + "\n\n---\n\n".join(parts)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TG Dataset Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 TG Dataset Search")
        gr.Markdown("Search by **username**, **user_id**, or **phone_number**")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g. Masterpeace999, 912711252, 6738212085",
                    lines=1,
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="Max Results",
                )

        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")

        search_btn.click(fn=search_ui,
                         inputs=[query_input, limit_slider],
                         outputs=output)
        query_input.submit(fn=search_ui,
                           inputs=[query_input, limit_slider],
                           outputs=output)
    return demo


# Mount Gradio onto the EXISTING FastAPI instance (do NOT reassign `app`)
gr.mount_gradio_app(app, build_ui(), path="/ui")
