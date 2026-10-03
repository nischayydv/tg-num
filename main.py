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

# ── Datasets ────────────────────────────────────────────────────────────────
DATASETS = {
    "tg": {
        "label": "TG Dataset",
        "url": os.environ.get(
            "TG_PARQUET_URL",
            "https://huggingface.co/datasets/Nischayydv/tg-dataset/resolve/main/merged_all.parquet",
        ),
        "columns": ["user_id", "phone_number", "username", "country", "country_code"],
        "phone_col": "phone_number",
    },
    "telegram": {
        "label": "Telegram Dataset",
        "url": os.environ.get(
            "TELEGRAM_PARQUET_URL",
            "hf://datasets/sauravsingh2111/Telegram/Telegram_10Digit_Chunk_part1.parquet",
        ),
        "columns": ["user_id", "phone", "username", "first_name", "last_name", "email"],
        "phone_col": "phone",
    },
}

# Unified column set across both datasets
UNIFIED_COLUMNS = [
    "user_id", "phone_number", "phone", "username",
    "country", "country_code", "first_name", "last_name", "email",
]

PARALLELISM      = int(os.environ.get("TG_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("TG_THREADS_PER_CONN", "2"))

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

    token = os.environ.get("HF_TOKEN")
    if token:
        con.execute(
            f"CREATE OR REPLACE SECRET hf "
            f"(TYPE huggingface, TOKEN '{token}')"
        )

    # ── Per-dataset views ───────────────────────────────────────────────
    for key, cfg in DATASETS.items():
        con.execute(
            f"CREATE OR REPLACE VIEW people_{key} AS "
            f"SELECT * FROM read_parquet('{cfg['url']}')"
        )

    # ── Unified view: both datasets merged by column name ───────────────
    # union_by_name = true fills missing columns with NULL so that
    # phone_number (tg) and phone (telegram) both survive.
    urls = [f"'{cfg['url']}'" for cfg in DATASETS.values()]
    con.execute(
        f"CREATE OR REPLACE VIEW people_all AS "
        f"SELECT * FROM read_parquet([{', '.join(urls)}], union_by_name = true)"
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


def _resolve_dataset(name: str) -> str:
    name = (name or "telegram").lower()
    if name not in DATASETS:
        raise HTTPException(400, f"Unknown dataset '{name}'. Use one of: {list(DATASETS)}")
    return name


# ── Core: user_id → phone ───────────────────────────────────────────────────
def lookup_by_user_id(dataset: str, user_id: str, limit: int = 20) -> dict:
    ds = _resolve_dataset(dataset)
    uid = _escape(str(user_id).strip())
    cols = DATASETS[ds]["columns"]
    phone_col = DATASETS[ds]["phone_col"]
    sql = (
        f"SELECT {', '.join(cols)} FROM people_{ds} "
        f"WHERE CAST(user_id AS VARCHAR) = '{uid}' LIMIT {limit}"
    )
    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())

    phones, seen = [], set()
    for r in rows:
        p = r.get(phone_col)
        if p and p not in seen:
            seen.add(p)
            phones.append(p)

    return {
        "dataset": ds,
        "user_id": user_id,
        "phone_numbers": phones,
        "count": len(rows),
        "results": rows,
    }


# ── ⭐ SEARCH BOTH DATASETS SIMULTANEOUSLY ──────────────────────────────────
def search_all(
    user_id: str = "",
    username: str = "",
    phone: str = "",
    limit: int = 50,
) -> dict:
    """
    Search across BOTH datasets at once.
    Provide at least one of: user_id, username, or phone.
    """
    conditions = []
    if user_id and user_id.strip():
        v = _escape(user_id.strip())
        conditions.append(f"CAST(user_id AS VARCHAR) = '{v}'")
    if username and username.strip():
        v = _escape(username.strip())
        conditions.append(f"username ILIKE '%{v}%'")
    if phone and phone.strip():
        v = _escape(phone.strip())
        # phone_number is from tg dataset; phone is from telegram dataset
        conditions.append(
            f"(CAST(phone_number AS VARCHAR) = '{v}' "
            f"OR CAST(phone AS VARCHAR) = '{v}')"
        )

    if not conditions:
        raise HTTPException(422, "Provide at least one of: user_id, username, phone")

    where = " AND ".join(conditions)
    sql = f"SELECT * FROM people_all WHERE {where} LIMIT {limit}"

    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())

    # Add a `source` label so you know which dataset each row came from
    for r in rows:
        if r.get("phone_number") and r.get("country"):
            r["source"] = "tg"
        else:
            r["source"] = "telegram"

    return {
        "query": {"user_id": user_id, "username": username, "phone": phone},
        "count": len(rows),
        "results": rows,
    }


# ── Unified search across all string columns (both datasets) ────────────────
def search_both_datasets(q: str, limit: int = 20) -> dict:
    q = q.strip()
    if not q:
        return {"query": q, "count": 0, "results": []}
    v = _escape(q)

    # Search across every column present in the unified view
    where = " OR ".join(
        f"CAST({c} AS VARCHAR) ILIKE '%{v}%'" for c in UNIFIED_COLUMNS
    )
    sql = f"SELECT * FROM people_all WHERE {where} LIMIT {limit}"

    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())

    for r in rows:
        if r.get("phone_number") and r.get("country"):
            r["source"] = "tg"
        else:
            r["source"] = "telegram"

    return {"query": q, "count": len(rows), "results": rows}


# ── Per-dataset helpers (kept for compatibility) ────────────────────────────
def _unified_search(dataset: str, q: str, limit: int = 10) -> dict:
    ds = _resolve_dataset(dataset)
    q = q.strip()
    if not q:
        return {"dataset": ds, "query": q, "count": 0, "results": []}
    cols = DATASETS[ds]["columns"]
    v = _escape(q)
    where = " OR ".join(f"CAST({c} AS VARCHAR) ILIKE '%{v}%'" for c in cols)
    sql = f"SELECT * FROM people_{ds} WHERE {where} LIMIT {limit}"
    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    return {"dataset": ds, "query": q, "count": len(rows), "results": rows}


def _run_field_search(dataset: str, field: str, value: str, mode: str, limit: int) -> dict:
    ds = _resolve_dataset(dataset)
    if field not in DATASETS[ds]["columns"]:
        raise ValueError(f"Unknown field '{field}' for dataset '{ds}'")
    v = _escape(value)
    if mode == "exact":
        sql = f"SELECT * FROM people_{ds} WHERE CAST({field} AS VARCHAR) = '{v}' LIMIT {limit}"
    elif mode == "contains":
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = (f"SELECT * FROM people_{ds} "
               f"WHERE CAST({field} AS VARCHAR) ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit}")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    con = _get_conn()
    rows = _rows_to_dicts(con, con.execute(sql).fetchall())
    return {"dataset": ds, "field": field, "value": value, "mode": mode,
            "count": len(rows), "results": rows}


# ── FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Unified TG + Telegram Search API")


class BatchUserIDRequest(BaseModel):
    user_ids: List[str]
    dataset: str = "telegram"
    limit: int = 20


@app.get("/")
def root():
    return {
        "app": "Unified TG + Telegram Search API",
        "datasets": {
            k: {"label": v["label"], "columns": v["columns"]}
            for k, v in DATASETS.items()
        },
        "endpoints": {
            "search_all": "/search/all?user_id=...&username=...&phone=...",
            "search_both": "/search/both?q=...",
            "lookup_user_id": "/user/{user_id}?dataset=telegram",
            "unified_search": "/search?q=...&dataset=telegram",
            "bots_search": "/bots?q=razvitie&dataset=telegram",
            "batch_lookup": "POST /users/batch",
            "debug": "/debug",
        },
        "docs": "/docs",
        "ui": "/ui",
    }


@app.get("/health")
def health():
    out = {}
    for key in DATASETS:
        try:
            con = _get_conn()
            n = con.execute(f"SELECT COUNT(*) FROM people_{key}").fetchone()[0]
            out[key] = {"status": "ok", "rows": n}
        except Exception as e:
            out[key] = {"status": "error", "detail": str(e)}
    try:
        con = _get_conn()
        n = con.execute("SELECT COUNT(*) FROM people_all").fetchone()[0]
        out["all"] = {"status": "ok", "rows": n}
    except Exception as e:
        out["all"] = {"status": "error", "detail": str(e)}
    return out


@app.get("/debug")
def debug():
    import traceback
    out = {}
    try:
        con = _new_conn()
        out["duckdb"] = "connected"
    except Exception as e:
        out["duckdb_error"] = str(e)
        out["traceback"] = traceback.format_exc()
        return out

    for key in list(DATASETS) + ["all"]:
        view = f"people_{key}"
        try:
            n = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
            out[key] = {"ok": True, "rows": n}
        except Exception as e:
            out[key] = {"ok": False, "error": str(e)}
    return out


# ⭐ Search BOTH datasets at once
@app.get("/search/all")
async def search_all_endpoint(
    user_id: str = Query("", description="Exact user_id match"),
    username: str = Query("", description="Partial username match (contains)"),
    phone: str = Query("", description="Exact phone match (both datasets)"),
    limit: int = Query(50, ge=1, le=500),
    pretty: bool = Query(True),
):
    """
    Search across BOTH datasets simultaneously.

    Examples:
      /search/all?user_id=1686533205
      /search/all?username=faceless
      /search/all?phone=5111381608
      /search/all?username=faceless&phone=5111381608
    """
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(
        pool, search_all, user_id, username, phone, limit
    )
    result = {"success": bool(data["count"]), **data}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


# ⭐ Free-text search across both datasets
@app.get("/search/both")
async def search_both_endpoint(
    q: str = Query(..., description="Free-text search across all columns in both datasets"),
    limit: int = Query(20, ge=1, le=200),
    pretty: bool = Query(True),
):
    """
    Free-text search across BOTH datasets.
    Matches any column (user_id, phone, username, names, etc.).

    Examples:
      /search/both?q=faceless_14
      /search/both?q=Razvitiekanala_bot
      /search/both?q=5111381608
    """
    if not q.strip():
        raise HTTPException(422, "Provide q")
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, search_both_datasets, q.strip(), limit)
    result = {"success": bool(data["count"]), **data}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


# Per-dataset endpoints (still work)
@app.get("/user/{user_id}")
async def user_lookup(
    user_id: str,
    dataset: str = Query("telegram"),
    limit: int = Query(20, ge=1, le=200),
    pretty: bool = Query(True),
):
    if not user_id.strip():
        raise HTTPException(422, "user_id cannot be empty")
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(pool, lookup_by_user_id, dataset, user_id, limit)
    result = {"success": bool(data["count"]), **data}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@app.get("/search")
async def search(
    q: str | None = Query(None),
    dataset: str = Query("telegram"),
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
            pool, _run_field_search, dataset, field, q.strip(), mode, limit
        )
    else:
        data = await loop.run_in_executor(pool, _unified_search, dataset, q.strip(), limit)
    result = {"success": bool(data["count"]), **data, "total": data["count"]}
    content = json.dumps(result, indent=2 if pretty else None, ensure_ascii=False)
    return Response(content=content, media_type="application/json")


@app.post("/users/batch")
async def users_batch(req: BatchUserIDRequest):
    if not req.user_ids:
        raise HTTPException(400, "user_ids must not be empty")
    if len(req.user_ids) > 100:
        raise HTTPException(400, "max 100 user_ids per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(pool, lookup_by_user_id, req.dataset, uid, req.limit)
        for uid in req.user_ids
    ]
    results = await asyncio.gather(*tasks)
    return Response(
        content=json.dumps({"count": len(results), "results": list(results)},
                           indent=2, ensure_ascii=False),
        media_type="application/json",
    )


# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_row(r: dict, columns: list) -> str:
    lines = []
    for k in columns:
        v = r.get(k)
        if v:
            lines.append(f"**{k}:** {v}")
    src = r.get("source")
    if src:
        lines.append(f"**source:** `{src}`")
    return "\n\n".join(lines)


def search_all_ui(user_id: str, username: str, phone: str, limit: int) -> str:
    if not (user_id or username or phone):
        return "⚠️ Enter at least one of: user_id, username, or phone."
    try:
        data = search_all(user_id or "", username or "", phone or "", int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"

    if not data["count"]:
        return "❌ **No results found across both datasets.**"

    lines = [f"🔍 **Searching BOTH datasets**  |  **Found:** {data['count']}", "", "---", ""]
    for i, r in enumerate(data["results"], 1):
        lines.append(f"### Result {i}")
        lines.append(format_row(r, UNIFIED_COLUMNS))
        lines.append("")
    return "\n\n".join(lines)


def search_both_ui(q: str, limit: int) -> str:
    if not q or not q.strip():
        return "⚠️ Please enter a search term."
    try:
        data = search_both_datasets(q.strip(), int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"

    if not data["count"]:
        return f"🔍 **Query:** `{q}`\n\n❌ **No results found across both datasets.**"

    lines = [f"🔍 **Query:** `{q}`  |  **Found:** {data['count']}", "", "---", ""]
    for i, r in enumerate(data["results"], 1):
        lines.append(f"### Result {i}")
        lines.append(format_row(r, UNIFIED_COLUMNS))
        lines.append("")
    return "\n\n".join(lines)


def lookup_ui(dataset: str, user_id: str, limit: int) -> str:
    if not user_id or not user_id.strip():
        return "⚠️ Please enter a user ID."
    uid = user_id.strip()
    try:
        data = lookup_by_user_id(dataset, uid, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    if not data["count"]:
        return f"🔍 **Dataset:** `{dataset}`  \n**User ID:** `{uid}`\n\n❌ **No records found.**"
    phones = data["phone_numbers"]
    phones_str = ", ".join(f"`{p}`" for p in phones) if phones else "_none_"
    lines = [
        f"🔍 **Dataset:** `{dataset}`",
        f"🆔 **User ID:** `{uid}`",
        f"📞 **Phone number(s):** {phones_str}",
        f"**Matches:** {data['count']}",
        "", "---", "",
    ]
    for i, r in enumerate(data["results"], 1):
        lines.append(f"### Result {i}")
        lines.append(format_row(r, DATASETS[data["dataset"]]["columns"]))
        lines.append("")
    return "\n\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Unified TG + Telegram Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Unified TG + Telegram Search")
        gr.Markdown("Search across **both datasets simultaneously** using DuckDB's `union_by_name`.")

        with gr.Tab("🔎 Search Both"):
            with gr.Row():
                uid_in = gr.Textbox(label="User ID (exact)", placeholder="e.g. 1686533205")
                uname_in = gr.Textbox(label="Username (contains)", placeholder="e.g. faceless")
                phone_in = gr.Textbox(label="Phone (exact)", placeholder="e.g. 5111381608")
                limit_all = gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Max Results")
            btn_all = gr.Button("🔍 Search Both Datasets", variant="primary", size="lg")
            out_all = gr.Markdown(label="Results")
            btn_all.click(fn=search_all_ui, inputs=[uid_in, uname_in, phone_in, limit_all], outputs=out_all)

        with gr.Tab("💬 Free-text Search"):
            with gr.Row():
                q_in = gr.Textbox(label="Search Query", placeholder="e.g. Razvitiekanala_bot")
                limit_both = gr.Slider(minimum=1, maximum=200, value=20, step=1, label="Max Results")
            btn_both = gr.Button("🔍 Search Both", variant="primary", size="lg")
            out_both = gr.Markdown(label="Results")
            btn_both.click(fn=search_both_ui, inputs=[q_in, limit_both], outputs=out_both)

        with gr.Tab("🆔 Single Dataset"):
            with gr.Row():
                ds_dd = gr.Dropdown(choices=["telegram", "tg"], value="telegram", label="Dataset")
                uid2 = gr.Textbox(label="User ID", placeholder="e.g. 1686533205")
                limit_ds = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Max Results")
            btn_ds = gr.Button("🔍 Lookup", variant="primary")
            out_ds = gr.Markdown(label="Results")
            btn_ds.click(fn=lookup_ui, inputs=[ds_dd, uid2, limit_ds], outputs=out_ds)
    return demo


gr.mount_gradio_app(app, build_ui(), path="/ui")
