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

PARALLELISM      = int(os.environ.get("TG_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("TG_THREADS_PER_CONN", "2"))
DUPLICATE_CAP    = 2

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns: List[duckdb.DuckDBPyConnection] = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")


def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # Vercel only allows writes to /tmp
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute(f"SET threads = {THREADS_PER_CONN}")

    # Optional HF token for private/gated datasets
    token = os.environ.get("HF_TOKEN")
    if token:
        con.execute(
            f"CREATE OR REPLACE SECRET hf "
            f"(TYPE huggingface, TOKEN '{token}')"
        )

    # One view per dataset
    for key, cfg in DATASETS.items():
        url = cfg["url"]
        # Use hf:// protocol if available, else direct URL
        if url.startswith("hf://"):
            con.execute(
                f"CREATE OR REPLACE VIEW people_{key} AS "
                f"SELECT * FROM read_parquet('{url}')"
            )
        else:
            con.execute(
                f"CREATE OR REPLACE VIEW people_{key} AS "
                f"SELECT * FROM read_parquet('{url}')"
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


# ── Unified search across all string columns ────────────────────────────────
def _unified_search(dataset: str, q: str, limit: int = 10) -> dict:
    ds = _resolve_dataset(dataset)
    q = q.strip()
    if not q:
        return {"dataset": ds, "query": q, "count": 0, "results": []}

    cols = DATASETS[ds]["columns"]
    v = _escape(q)
    where = " OR ".join(
        f"CAST({c} AS VARCHAR) ILIKE '%{v}%'" for c in cols
    )
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
app = FastAPI(title="TG + Telegram Search API")


class BatchUserIDRequest(BaseModel):
    user_ids: List[str]
    dataset: str = "telegram"
    limit: int = 20


@app.get("/")
def root():
    return {
        "app": "TG + Telegram Search API",
        "datasets": {
            k: {"label": v["label"], "columns": v["columns"]}
            for k, v in DATASETS.items()
        },
        "endpoints": {
            "lookup_user_id": "/user/{user_id}?dataset=telegram",
            "unified_search": "/search?q=...&dataset=telegram",
            "field_search": "/search?q=...&field=phone&mode=exact&dataset=telegram",
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

    for key, cfg in DATASETS.items():
        try:
            n = con.execute(f"SELECT COUNT(*) FROM people_{key}").fetchone()[0]
            out[key] = {"ok": True, "rows": n, "url": cfg["url"]}
        except Exception as e:
            out[key] = {"ok": False, "error": str(e), "url": cfg["url"]}
    return out


# ⭐ Primary: user_id → phone
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


# ── Gradio UI ───────────────────────────────────────────────────────────────
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
        "",
        "---",
        "",
    ]
    for i, r in enumerate(data["results"], 1):
        lines.append(f"### Result {i}")
        for k in DATASETS[data["dataset"]]["columns"]:
            v = r.get(k)
            if v:
                lines.append(f"**{k}:** {v}")
        lines.append("")
    return "\n\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="TG + Telegram Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 TG + Telegram Search")
        gr.Markdown("Look up **user_id → phone** across both datasets.")

        with gr.Row():
            dataset_dd = gr.Dropdown(
                choices=["telegram", "tg"],
                value="telegram",
                label="Dataset",
            )
            uid_input = gr.Textbox(label="User ID", placeholder="e.g. 1686533205", lines=1)
            limit_slider = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Max Results")

        btn = gr.Button("🔍 Lookup", variant="primary", size="lg")
        output = gr.Markdown(label="Results")

        btn.click(fn=lookup_ui, inputs=[dataset_dd, uid_input, limit_slider], outputs=output)
        uid_input.submit(fn=lookup_ui, inputs=[dataset_dd, uid_input, limit_slider], outputs=output)
    return demo


gr.mount_gradio_app(app, build_ui(), path="/ui")
