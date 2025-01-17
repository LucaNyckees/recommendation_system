from fastapi import APIRouter, FastAPI, HTTPException
import json
from pydantic import BaseModel
from contextlib import asynccontextmanager
from psycopg.sql import SQL
from typing import Any
import pandas as pd

from src.fastapi_app.schemas.schemas import CategorySummary, ChatQueryInput, ChatQueryOutput
from src.log.logger import logger
from src.fastapi_app.utils.config import config
from src.database.db_functions import load_dicts_from_query
from src.database.connection import connect
from src.fastapi_app.utils.async_utils import async_retry
from src.rag_agent.agent import rag_agent_executor


base_router = APIRouter()


@base_router.get("/health")
def health():
    return {"status": "healthy"}


@base_router.post("/dashboard/all_categories/table_summary")
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


@base_router.post("/dashboard/all_categories/marimekko_price_volume")
async def get_summary() -> list[dict[str, Any]]:
    """
    A route to produce a marimekko chart for price volume per category in the dashboard.
    """
    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            query = SQL("""        
                SELECT
                main_category,
                SUM(price) AS total_price,
                SUM(rating_number) AS total_rating_number,
                SUM(price * rating_number) AS total_volume
                FROM rs_amazon_products p
                INNER JOIN rs_amazon_reviews r
                ON p.parent_asin = r.parent_asin
                WHERE price != 'NaN'
                GROUP BY main_category;""")
            marimekko_data = load_dicts_from_query(cur=cur, query=query, params=None)
    return marimekko_data


@base_router.post("/dashboard/all_categories/transaction_volume_time_series")
async def get_summary() -> list[dict[str, Any]]:
    """
    A route to produce a marimekko chart for price volume per category in the dashboard.
    """
    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            query = SQL("""        
                SELECT
                    DATE_TRUNC('month', CAST(date AS DATE)) AS month,
                    SUM(price * rating_number) AS total_volume
                FROM rs_amazon_products p
                INNER JOIN rs_amazon_reviews r
                ON p.parent_asin = r.parent_asin
                WHERE price != 'NaN'
                GROUP BY month
                ORDER BY month;""")
            volumes = load_dicts_from_query(cur=cur, query=query, params=None)
    return volumes


@base_router.post("/dashboard/all_categories/avg_rating")
async def get_summary() -> list[dict[str, Any]]:
    """
    A route to produce a marimekko chart for price volume per category in the dashboard.
    """
    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            query = SQL("""               
                SELECT
                    AVG(average_rating) AS average_rating, main_category
                FROM rs_amazon_products p
                INNER JOIN rs_amazon_reviews r
                ON p.parent_asin = r.parent_asin
                GROUP BY main_category
                ORDER BY average_rating DESC;""")
            avg_ratings = load_dicts_from_query(cur=cur, query=query, params=None)
    return avg_ratings


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return await rag_agent_executor.ainvoke({"input": query})


@base_router.post("/dashboard/all_categories/rag-agent")
async def rag_agent(query: ChatQueryInput) -> ChatQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response