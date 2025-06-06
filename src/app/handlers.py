# src/app/modules/variance/handlers.py

import os
import duckdb
from litestar import (
    Litestar,
    MediaType,
    Request,
    UploadFile,
    post,
)
from litestar.params import Form, File
from litestar.exceptions import HTTPException
from litestar.status_codes import HTTP_400_BAD_REQUEST
from services.llm_engine import generate_variance_narrative_async
from db.router import get_client_session
from dotenv import load_dotenv

load_dotenv()

def read_tb_into_duckdb(raw_bytes: bytes) -> list[dict]:
    """
    Load CSV bytes into a DuckDB in-memory table, compute variance, return list of dicts.
    """
    import pandas as pd
    from io import BytesIO

    df = pd.read_csv(BytesIO(raw_bytes))
    if not {"account_code", "prev", "curr"}.issubset(df.columns):
        raise ValueError("CSV must include columns: account_code, prev, curr")

    df["variance"] = ((df["curr"] - df["prev"]) / df["prev"]).round(2) * 100
    return df.to_dict(orient="records")


@post("/generate-variance", media_type=MediaType.JSON)
async def generate_variance_handler(
    request: Request,
    org_id: int = Form(...),
    schema_name: str = Form(...),
    period_start: str = Form(...),
    period_end: str = Form(...),
    tone: str = Form("controller"),
    language: str = Form("en"),
    model: str = Form("openai"),  # 'openai', 'claude', or 'llama_local'
    tb_file: UploadFile = File(...),
) -> dict:
    """
    Enqueue a Celery task to process the trial balance (via DuckDB + LLM).
    Returns { task_id, status: "pending" }.
    """
    try:
        raw_bytes = await tb_file.read()
        tb_data = read_tb_into_duckdb(raw_bytes)

        # Kick off an async Celery task
        task = generate_variance_narrative_async.delay(
            tb_data,
            org_id,
            schema_name,
            period_start,
            period_end,
            tone,
            language,
            model,
        )

        return {"task_id": task.id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))


@post("/get-variance-result", media_type=MediaType.JSON)
async def get_variance_result(request: Request, task_id: str = Form(...)) -> dict:
    """
    Poll a Celery task’s status (pending / success / failure) and return result if ready.
    """
    from celery import Celery

    celery_app = Celery(broker=os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0"))
    async_result = celery_app.AsyncResult(task_id)

    if async_result.state == "PENDING":
        return {"task_id": task_id, "status": "pending"}
    if async_result.state == "SUCCESS":
        return {"task_id": task_id, "status": "completed", "result": async_result.result}
    return {"task_id": task_id, "status": async_result.state, "result": str(async_result.info)}

