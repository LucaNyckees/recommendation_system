from rich.console import Console
import pandas as pd
import json
import os
from psycopg.types.json import Jsonb
from psycopg.sql import SQL
import numpy as np

from src.paths import DATA_PATH, RESOURCES_PATH
from src.database.connection import connect
from src.log.logger import logger
from src.database.db_functions import insert_values, load_dataframe_from_query
from src.database.populate.helpers import get_amazon_categories_in_db


with open(RESOURCES_PATH / "amazon_product_categories.json") as f:
    categories_dict = json.load(f)


db_cols_to_inserted_cols_mapping = {
    "user_id": "user_id",
    "age": "age",
    "gender": "gender",
}

cities = {
    "paris": 1/3,
    "berlin": 1/3,
    "madrid": 1/3,
}

# Verify the sum of probabilities
assert abs(sum(cities.values()) - 1) < 1e-9, f"Values do not sum to 1, but to {sum(cities.values())}"

genders = {"female": 0.45, "male": 0.45, "non-binary": 0.05, "other": 0.05}


def load_users() -> None:

    logger.info("Loading users")

    users_dataframe = None

    with connect(db_key="main") as conn:
        with conn.cursor() as cur:

            query = SQL("""SELECT DISTINCT(user_id) FROM rs_amazon_reviews;""")
            users_dataframe = load_dataframe_from_query(cur=cur, query=query, params=None)
            nb_users = len(users_dataframe)

            assert nb_users > 0, "no user found in reviews data, please load the datasets first."

            logger.info(f"Generating random data for {nb_users} users")
            users_dataframe["age"] = np.random.normal(loc=35, scale=7, size=nb_users)
            users_dataframe["gender"] = np.random.choice(list(genders.keys()), size=nb_users, p=list(genders.values()))
            users_dataframe["city"] = np.random.choice(list(cities.keys()), size=nb_users, p=list(cities.values()))

            logger.info("Inserting values...")
            insert_values(
                cur=cur,
                table="rs_amazon_users",
                values=users_dataframe.to_dict("records"),
                cols_mapping=db_cols_to_inserted_cols_mapping,
            )
    Console().log("Users loaded")
