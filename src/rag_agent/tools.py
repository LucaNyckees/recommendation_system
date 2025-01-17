from psycopg.sql import SQL, Literal

from src.database.connection import connect
from src.database.db_functions import load_dicts_from_query


def get_product_description(product_id: str) -> str:
    """
    Get the product's description formatted as a string.

    Args:
        - product_id (str): e.g. 'B076WQZGPM'
    """

    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            query = SQL("""
                SELECT description
                FROM rs_amazon_products
                WHERE parent_asin = {parent_asin};""").format(parent_asin=Literal(product_id))
            res = load_dicts_from_query(cur=cur, query=query, params=None)

    desc = res[0]

    return desc