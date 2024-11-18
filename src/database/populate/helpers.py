from psycopg.sql import SQL
from src.database.db_functions import load_dicts_from_query
from src.database.connection import connect
from src.log.logger import logger


def get_amazon_categories_in_db() -> list[str]:
    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            query = SQL("""
                        SELECT DISTINCT(main_category)
                        FROM rs_amazon_products p
                        INNER JOIN rs_amazon_reviews r
                        ON p.parent_asin = r.parent_asin;
                        """)
            fetched = load_dicts_from_query(cur=cur, query=query, params=None)
            categories_in_db = [d["main_category"].replace(" ", "_") for d in fetched]
    logger.info(f"Categories already present in db : {categories_in_db}")
    return categories_in_db