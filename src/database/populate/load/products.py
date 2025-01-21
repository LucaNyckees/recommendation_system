from rich.console import Console
import pandas as pd
import json
import os
from psycopg.types.json import Jsonb
from langchain.embeddings import OpenAIEmbeddings

from src.paths import DATA_PATH, RESOURCES_PATH
from src.database.connection import connect
from src.log.logger import logger
from src.database.db_functions import insert_values
from src.database.populate.helpers import get_amazon_categories_in_db


with open(RESOURCES_PATH / "amazon_product_categories.json") as f:
    categories_dict = json.load(f)


db_cols_to_inserted_cols_mapping = {
    "parent_asin": "parent_asin",
    "name": "title",
    "main_category": "main_category",
    "average_rating": "average_rating",
    "rating_number": "rating_number",
    "features": "features",
    "description": "description",
    "price": "price",
    # "image_urls": "images",
    "store": "store",
    "categories": "categories",
    "details": "details",
    "name_embedding": "name_embedding",
    "description_embedding": "description_embedding",
}


def load_products() -> None:

    logger.info("Loading products")

    products_dataframe = None

    categories_in_db = get_amazon_categories_in_db()

    categories_to_insert = set(categories_dict["categories"]) - set(categories_in_db)
    if categories_to_insert == set():
        logger.info("No categories to insert.")
        return None

    for category in categories_to_insert:

        logger.info(f"Accessing new category {category}")

        file_path = DATA_PATH / "amazon" / f"meta_{category}.jsonl"
        if not os.path.isfile(file_path):
            logger.warning(f"No dataset found at {file_path}.")
            continue

        if products_dataframe is None:
            products_dataframe = pd.read_json(file_path, lines=True)
        else:
            to_append = pd.read_json(file_path, lines=True)
            products_dataframe = products_dataframe._append(to_append, ignore_index=True)

    products_dicts = products_dataframe.to_dict("records")
    for p in products_dicts:
        p["details"] = Jsonb(p["details"])

    logger.info("Computing review embeddings...")
    products_dicts = products_dataframe.to_dict("records")
    embedding_model = OpenAIEmbeddings()
    products_dicts = [{**d, "name_embedding": embedding_model.embed_query(d["name"])} for d in products_dicts]
    products_dicts = [{**d, "description_embedding": embedding_model.embed_query(d["description"])} for d in products_dicts]

    logger.info("Inserting values...")
    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            insert_values(cur=cur, table="rs_amazon_products", values=products_dicts, cols_mapping=db_cols_to_inserted_cols_mapping)
    Console().log("Products loaded")
