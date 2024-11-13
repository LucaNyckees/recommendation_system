from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import pandas as pd
import os

ROOT_DIR = ("/").join(os.getcwd().split("/"))
import sys
print(ROOT_DIR)

sys.path.append(ROOT_DIR)

from src.database.connection import connect
from src.database.db_functions import get_amazon_dataframe

"""
Ideas:
Section 1 : basic data visualization about the Amazon datasets
- histograms for
    * price
    * average rating
    * number of ratings
    * piechart of sentiment
- dataframe description (e.g. nb of unique users, and unique products)
Section 2 : models performance
- user-item matrix and a pyvis graph where nodes are users maybe, and edges...?
- reports for classifiers targetting sentiments
- comparing textblob sentiments with labeled sentiments (extracted from ratings)
"""


# loading data (later from FastAPI application)
with connect(db_key="main") as conn:
    with conn.cursor() as cur:
        df = get_amazon_dataframe(cur=cur, categories=None, limit=10000)

app = Dash(__name__)



if __name__ == "__main__":
    app.run_server(debug=True)
