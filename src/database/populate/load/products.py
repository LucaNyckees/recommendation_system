from rich.console import Console
import pandas as pd
import json
import os
from psycopg.sql import SQL, Identifier, Placeholder

from src.paths import DATA_PATH, RESOURCES_PATH
from src.database.connection import connect
from src.log.logger import logger


with open(RESOURCES_PATH / "amazon_product_categories.json") as f:
    categories_dict = json.load(f)


db_cols_to_inserted_cols_mapping = {
    "parent_asin": "parent_asin",
    "name": "title",
    "main_category": "main_category",
    "average_rating": "average_rating",
    "rating_number": "rating_number",
    "features": "",
    "description": "description",
    "price": "price",
    "image_urls": "images",
    "store": "store",
    "categories": "categories",
    "details": "details"
}


def load_products() -> None:

    logger.info("Loading products")

    products_dataframe = None

    for category in categories_dict["categories"]:

        logger.info(f"Accessing category {category}")

        file_path = DATA_PATH / f"meta_{category}.jsonl"
        if not os.path.isfile(file_path):
            continue

        if products_dataframe is None:
            products_dataframe = pd.read_json(file_path, lines=True)
        else:
            to_append = pd.read_json(file_path, lines=True)
            products_dataframe = products_dataframe.append(to_append, ignore_index=True)

        products_list_of_dicts = products_dataframe.to_dict("records")

    logger.info("Inserting values...")
    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            insert_query = SQL("INSERT INTO rs_amazon_products ({db_cols}) VALUES ({inserted_cols})").format(
                db_cols=SQL(", ").join(map(Identifier, db_cols_to_inserted_cols_mapping.keys())),
                inserted_cols=SQL(", ").join(map(Placeholder, db_cols_to_inserted_cols_mapping.values())),
                )
            cur.executemany(query=insert_query, params_seq=products_list_of_dicts)
    Console().log("Countries loaded")
