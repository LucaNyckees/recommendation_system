from fastapi import APIRouter, FastAPI, HTTPException
import json
from pydantic import BaseModel
from contextlib import asynccontextmanager
from psycopg.sql import SQL
from typing import Any

from src.fastapi_app.schemas.schemas import CategorySummary
from src.log.logger import logger
from src.fastapi_app.utils.config import config
from src.database.db_functions import load_dicts_from_query
from src.database.connection import connect


base_router = APIRouter()

@base_router.post("/dashboard/summary")
async def get_summary() -> list[dict[str, Any]]:
    with connect() as conn:
        with conn.cursor() as cur:
            query = SQL("""""")
            fetched = load_dicts_from_query(cur=cur, query=query, params=None)
    return fetched
