from fastapi import APIRouter, FastAPI, HTTPException
import json
from pydantic import BaseModel
from contextlib import asynccontextmanager
from psycopg.sql import SQL
from typing import Any
import pandas as pd

from src.fastapi_app.schemas.schemas import CategorySummary
from src.log.logger import logger
from src.fastapi_app.utils.config import config
from src.database.db_functions import load_dicts_from_query
from src.database.connection import connect


base_router = APIRouter()

@base_router.post("/dashboard/table_summary_all_categories")
async def get_summary() -> list[dict[str, Any]]:
    """
    A route to produce a table summary over all categories in the dashboard.
    """
    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            query = SQL("""
                SELECT main_category, AVG(average_rating) as avg_average_rating, SUM(rating_number) as total_rating_number
                FROM rs_amazon_products p
                INNER JOIN rs_amazon_reviews r
                ON p.parent_asin = r.parent_asin
                GROUP BY main_category;""")
            summary = load_dicts_from_query(cur=cur, query=query, params=None)
    return summary
