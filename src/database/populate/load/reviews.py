from rich.console import Console
import pandas as pd
import json
import os
from psycopg.sql import SQL, Identifier, Placeholder

from src.paths import DATA_PATH, RESOURCES_PATH
from src.database.connection import connect


with open(RESOURCES_PATH / "amazon_product_categories.json") as f:
    categories_dict = json.load(f)


db_cols_to_inserted_cols_mapping = {
    "asin": "asin",
    "parent_asin": "parent_asin",
    "title": "title",
    "rating": "rating",
    "text": "text",
    "user_id": "user_id",
    "timestamp": "timestamp",
    "helpful_vote": "helpful_vote",
    "verified_purchase": "verified_purchase"
}


def load_reviews():

    reviews_dataframe = None

    for category in categories_dict["categories"]:

        file_path = DATA_PATH / f"{category}.jsonl"
        if not os.path.isfile(file_path):
            continue

        if reviews_dataframe is None:
            reviews_dataframe = pd.read_json(file_path, lines=True)
        else:
            to_append = pd.read_json(file_path, lines=True)
            reviews_dataframe = reviews_dataframe.append(to_append, ignore_index=True)

        reviews_list_of_dicts = reviews_dataframe.to_dict("records")

    with connect(db_key="main") as conn:
        with conn.cursor() as cur:
            insert_query = SQL("INSERT INTO rs_amazon_reviews ({db_cols}) VALUES ({inserted_cols})").format(
                db_cols=", ".join(map(Identifier, db_cols_to_inserted_cols_mapping.keys())),
                inserted_cols=", ".join(map(Placeholder, db_cols_to_inserted_cols_mapping.values())),
                )
            cur.executemany(query=insert_query, params_seq=reviews_list_of_dicts)
    Console().log("Countries loaded")
