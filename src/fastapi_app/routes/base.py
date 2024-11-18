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
                GROUP BY main_category;""")
            avg_ratings = load_dicts_from_query(cur=cur, query=query, params=None)
    return avg_ratings
