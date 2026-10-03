import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import duckdb
import gradio as gr
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
HF_DATASET  = os.environ.get("TG_DATASET", "Nischayydv/tg-dataset")
HF_FILE     = os.environ.get("TG_FILE", "merged_all.parquet")
HF_REVISION = os.environ.get("TG_REVISION", "main")

PARQUET_URL = (
    f"https://huggingface.co/datasets/{HF_DATASET}"
    f"/resolve/{HF_REVISION}/{HF_FILE}"
)

PARALLELISM      = int(os.environ.get("TG_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("TG_THREADS_PER_CONN", "2"))

SEARCH_FIELDS = ["user_id", "phone_number", "username", "country", "country_code"]

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: List[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute(f"SET threads = {THREADS_PER_CONN}")

    # Optional HF token for private datasets / higher rate limits
    token = os.environ.get("HF_TOKEN")
    if token:
        con.execute(
            f"CREATE OR REPLACE SECRET hf "
            f"(TYPE huggingface, TOKEN '{token}')"
        )

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


def _rows_to_dicts(con, rows) -> List[dict]:
    cols = [d[0] for d in con.description]
    return [dict(zip(cols, r)) for r in rows]


# ── Core: user_id → phone_number ────────────────────────────────────────────
def lookup_by_user_id(user_id: str, limit: int = 20) -> dict:
    """
    Given a user_id, return all rows (phone_number, username, etc.).
    """
    uid = _escape(str(user_id).strip())
    sql = (
        f"SELECT user_id, phone_number, username, country, country_code "
        f"FROM people "
        f"WHERE CAST(user_id AS VARCHAR) = '{uid}' "
        f"LIMIT {limit}"
    )
    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())

    # Extract unique phone numbers
    phones = []
    seen = set()
    for r in rows:
        p = r.get("phone_number")
        if p and p not in seen:
            seen.add(p)
            phones.append(p)

    return {
        "user_id": user_id,
        "phone_numbers": phones,
        "count": len(rows),
        "results": rows,
    }


# ── General unified search (kept for flexibility) ───────────────────────────
def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not q:
        return {"query": q, "count": 0, "results": []}
    v = _escape(q)
    where = (
        f"username ILIKE '%{v}%' "
        f"OR CAST(user_id AS VARCHAR) ILIKE '%{v}%' "
        f"OR CAST(phone_number AS VARCHAR) ILIKE '%{v}%' "
        f"OR country ILIKE '%{v}%'"
    )
    sql = f"SELECT * FROM people WHERE {where} LIMIT {limit}"
    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    return {"query": q, "count": len(rows), "results": rows}


def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    v = _escape(value)

    if mode == "exact":
        sql = f"SELECT * FROM people WHERE {field} = '{v}' LIMIT {limit}"
    elif mode == "contains":
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = (f"SELECT * FROM people WHERE {field} ILIKE '%{v2}%' "
               f"ESCAPE '\\' LIMIT {limit}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    return {"field": field, "value": value, "mode": mode,
            "count": len(rows), "results": rows}


# ── FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI(title="TG Dataset — user_id → phone_number")


class BatchUserIDRequest(BaseModel):
    user_ids: List[str]
    limit: int = 20


@app.get("/")
def root():
    return {
        "app": "TG Dataset — user_id → phone_number",
        "dataset": HF_DATASET,
        "columns": SEARCH_FIELDS,
        "primary_endpoint": "/user/{user_id}",
        "endpoints": {
            "lookup_user_id": "/user/{user_id}",
            "batch_lookup": "POST /users/batch",
            "unified_search": "/search?q=...",
            "field_search": "/search?q=...&field=phone_number&mode=exact",
        },
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


# ⭐ Primary endpoint
@app.get("/user/{user_id}")
async def user_lookup(
    user_id: str,
    limit: int = Query(20, ge=1, le=200),
    pretty: bool = Query(True),
):
    """Look up phone numbers by user_id."""
    if not user_id.strip():
        raise HTTPException(422, "user_id cannot be empty")
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, lookup_by_user_id, user_id, limit)
    result = {"success": bool(data["count"]), **data}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


# Batch lookup: send a list of user_ids, get a map back
@app.post("/users/batch")
async def users_batch(req: BatchUserIDRequest):
    if not req.user_ids:
        raise HTTPException(400, "user_ids must not be empty")
    if len(req.user_ids) > 100:
        raise HTTPException(400, "max 100 user_ids per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(pool, lookup_by_user_id, uid, req.limit)
        for uid in req.user_ids
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps({"count": len(results), "results": results},
                           indent=2, ensure_ascii=False),
        media_type="application/json",
    )


# Keep general search too
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


# ── Gradio UI ───────────────────────────────────────────────────────────────
def lookup_ui(user_id: str, limit: int) -> str:
    if not user_id or not user_id.strip():
        return "⚠️ Please enter a user ID."
    uid = user_id.strip()
    try:
        data = lookup_by_user_id(uid, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"

    if not data["count"]:
        return f"🔍 **User ID:** `{uid}`\n\n❌ **No records found.**"

    phones = data["phone_numbers"]
    phones_str = ", ".join(f"`{p}`" for p in phones) if phones else "_none_"

    lines = [
        f"🔍 **User ID:** `{uid}`",
        f"📞 **Phone number(s):** {phones_str}",
        f"**Matches:** {data['count']}",
        "",
        "---",
        "",
    ]
    for i, r in enumerate(data["results"], 1):
        lines.append(f"### Result {i}")
        for k in SEARCH_FIELDS:
            v = r.get(k)
            if v:
                lines.append(f"**{k}:** {v}")
        lines.append("")
    return "\n\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="User ID → Phone Lookup", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📞 User ID → Phone Number Lookup")
        gr.Markdown("Enter a **user_id** to get all associated phone numbers and details.")

        with gr.Row():
            with gr.Column(scale=3):
                uid_input = gr.Textbox(
                    label="User ID",
                    placeholder="e.g. 912711252",
                    lines=1,
                )
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=1, maximum=100, value=20, step=1,
                    label="Max Results",
                )

        btn = gr.Button("🔍 Lookup", variant="primary", size="lg")
        output = gr.Markdown(label="Results")

        btn.click(fn=lookup_ui, inputs=[uid_input, limit_slider], outputs=output)
        uid_input.submit(fn=lookup_ui, inputs=[uid_input, limit_slider], outputs=output)
    return demo


# Mount Gradio onto the EXISTING FastAPI instance
gr.mount_gradio_app(app, build_ui(), path="/ui")
