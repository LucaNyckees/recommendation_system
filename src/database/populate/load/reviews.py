from rich.console import Console
import pandas as pd
import json
import os
from psycopg.sql import SQL, Identifier, Placeholder

from src.paths import DATA_PATH, RESOURCES_PATH
from src.database.connection import connect
from src.log.logger import logger
from src.database.db_functions import insert_values, load_dicts_from_query
from src.database.populate.helpers import get_amazon_categories_in_db


with open(RESOURCES_PATH / "amazon_product_categories.json") as f:
    categories_dict = json.load(f)


db_cols_to_inserted_cols_mapping = {
    "asin": "asin",
    "parent_asin": "parent_asin",
    "title": "title",
    "rating": "rating",
    "text": "text",
    "user_id": "user_id",
    "date": "date",
    "helpful_vote": "helpful_vote",
    "verified_purchase": "verified_purchase"
}


def load_reviews() -> None:

    logger.info("Loading reviews")

    reviews_dataframe = None

    categories_in_db = get_amazon_categories_in_db()

    logger.info(f"Categories already present in db : {categories_in_db}")

    categories_to_insert = set(categories_dict["categories"]) - set(categories_in_db)
    if categories_to_insert == set():
        logger.info("No categories to insert.")
        return None

    for category in categories_to_insert:

        logger.info(f"Accessing new category {category}")

        file_path = DATA_PATH / "amazon" / f"{category}.jsonl"
        if not os.path.isfile(file_path):
            logger.warning(f"No dataset found at {file_path}.")
            continue

        if reviews_dataframe is None:
            reviews_dataframe = pd.read_json(file_path, lines=True)
        else:
            to_append = pd.read_json(file_path, lines=True)
            reviews_dataframe = reviews_dataframe._append(to_append, ignore_index=True)

    reviews_dataframe["date"] = reviews_dataframe["timestamp"].dt.date

    reviews_list_of_dicts = reviews_dataframe.to_dict("records")

    logger.info("Inserting values...")
    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            insert_values(cur=cur, table="rs_amazon_reviews", values=reviews_list_of_dicts, cols_mapping=db_cols_to_inserted_cols_mapping)
    Console().log("Reviews loaded")
