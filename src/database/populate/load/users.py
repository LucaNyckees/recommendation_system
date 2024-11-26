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
    "paris": 0.075,       # Approx. population: 11,000,000
    "berlin": 0.062,      # Approx. population: 9,000,000
    "madrid": 0.058,      # Approx. population: 8,500,000
    "rome": 0.051,        # Approx. population: 7,500,000
    "london": 0.125,      # Approx. population: 18,500,000
    "amsterdam": 0.012,   # Approx. population: 1,800,000
    "brussels": 0.015,    # Approx. population: 2,200,000
    "vienna": 0.017,      # Approx. population: 2,500,000
    "warsaw": 0.021,      # Approx. population: 3,100,000
    "stockholm": 0.012,   # Approx. population: 1,800,000
    "oslo": 0.008,        # Approx. population: 1,200,000
    "helsinki": 0.007,    # Approx. population: 1,050,000
    "copenhagen": 0.011,  # Approx. population: 1,650,000
    "lisbon": 0.013,      # Approx. population: 1,900,000
    "athens": 0.021,      # Approx. population: 3,100,000
    "prague": 0.014,      # Approx. population: 2,000,000
    "budapest": 0.015,    # Approx. population: 2,200,000
    "dublin": 0.010,      # Approx. population: 1,500,000
    "zagreb": 0.007,      # Approx. population: 1,050,000
    "sofia": 0.008,       # Approx. population: 1,200,000
    "belgrade": 0.009,    # Approx. population: 1,350,000
    "bucharest": 0.020    # Approx. population: 3,000,000
}

# Verify the sum of probabilities
assert abs(sum(cities.values()) - 1) < 1e-9, "Values do not sum to 1"

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
            users_dataframe["gender"] = np.random.choice(genders.keys(), size=nb_users, p=genders.values())
            users_dataframe["city"] = np.random.choice(cities.keys(), size=nb_users, p=cities.values())

            logger.info("Inserting values...")
            insert_values(
                cur=cur,
                table="rs_amazon_users",
                values=users_dataframe.to_dict("records"),
                cols_mapping=db_cols_to_inserted_cols_mapping,
            )
    Console().log("Users loaded")
