"""
Ideas:
Section 1 : basic data visualization about the Amazon datasets
OK
- dataframe description (e.g. nb of unique users, and unique products)
Section 2 : models performance
- user-item matrix and a pyvis graph where nodes are users maybe, and edges...?
- reports for classifiers targetting sentiments
"""

from dash import Dash, html, dcc, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

ROOT_DIR = ("/").join(os.getcwd().split("/"))
import sys

sys.path.append(ROOT_DIR)

from src.database.connection import connect
from src.database.db_functions import get_amazon_dataframe
from src.nlp.sentiment_analysis.helpers import apply_sentiment_analysis


# Load the data from the database
with connect(db_key="main") as conn:
    with conn.cursor() as cur:
        df = get_amazon_dataframe(cur=cur, categories=None, limit=10000)
        df["review_input"] = df["title"] + df["text"]
        df = apply_sentiment_analysis(df=df)

app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Data and Models Visualization Dashboard"),
    html.Hr(),
    dcc.RadioItems(
        options=[
            {'label': 'Data Visualization', 'value': 'DataViz'},
            {'label': 'Model Performances', 'value': 'ModelPerf'},
            {'label': 'Recommendation Demo', 'value': 'RecomDemo'}
        ],
        value='DataViz',
        id='section-radio',
    ),
    html.Div(id='section-content')
])

# Callback to dynamically update the display section based on the selection
@app.callback(
    Output('section-content', 'children'),
    Input('section-radio', 'value')
)
def display_section(selected_section):
    if selected_section == 'DataViz':
        return html.Div([
            html.H3("Data Visualizations"),
            # Flexbox container for horizontal layout for the first line of plots
            html.Div([
                dcc.Graph(id='rating-histogram', style={'flex': '1'}),
                dcc.Graph(id='sentiment-scatterplot', style={'flex': '1'}),
                dcc.Graph(id='tb-sentiment-piechart', style={'flex': '1'}),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
            html.Div([
                dcc.Graph(id='price-histogram', style={'flex': '1'}),
                dcc.Graph(id='num-ratings-histogram', style={'flex': '1'}),
                dcc.Graph(id='sentiment-piechart', style={'flex': '1'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'margin-top': '20px'})
        ])
    else:
        return html.Div("Selected section: " + selected_section)

# Callback to update graphs in the Data Visualization section
@app.callback(
    Output('price-histogram', 'figure'),
    Output('rating-histogram', 'figure'),
    Output('num-ratings-histogram', 'figure'),
    Output('tb-sentiment-piechart', 'figure'),
    Output('sentiment-scatterplot', 'figure'),
    Output('sentiment-piechart', 'figure'),
    Input('section-radio', 'value')
)
def update_graphs(selected_section):
    if selected_section == 'DataViz':
        # Histogram for price
        price_histogram = px.histogram(df, x='price', nbins=30, title="Price Distribution")

        # Overlayed Histogram for average rating and average TextBlob sentiment rating
        rating_histogram = go.Figure()
        
        # First histogram trace for 'average_rating'
        rating_histogram.add_trace(
            go.Histogram(
                x=df['average_rating'],
                nbinsx=20,
                name='user',
                opacity=0.6,
                marker_color='blue'
            )
        )
        
        # Second histogram trace for 'average_tb_sentiment_rating'
        rating_histogram.add_trace(
            go.Histogram(
                x=df['average_tb_sentiment_rating'],
                nbinsx=20,
                name='textblob',
                opacity=0.6,
                marker_color='orange'
            )
        )
        
        # Update layout for the overlay
        rating_histogram.update_layout(
            title="avg rating distribution",
            xaxis_title="Rating",
            yaxis_title="Count",
            barmode='overlay'
        )

        # Histogram for number of ratings
        num_ratings_histogram = px.histogram(df, x='rating_number', nbins=20, title="Number of Ratings Distribution")

        # Pie charts for sentiment
        tb_sentiment_counts = df['tb_sentiment_category'].value_counts().reset_index()
        tb_sentiment_counts.columns = ['tb_sentiment_category', 'count']
        tb_sentiment_piechart = px.pie(tb_sentiment_counts, names='tb_sentiment_category', values='count', title="textblob sentiments")

        sentiment_counts = df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment_category', 'count']
        sentiment_piechart = px.pie(sentiment_counts, names='sentiment_category', values='count', title="user sentiments")

        # Scatter plot for average_tb_sentiment_rating vs tb_sentiment_rating
        sentiment_scatterplot = px.scatter(
            df,
            x='average_tb_sentiment_rating',
            y='tb_sentiment_rating',
            title="avg textblob vs. avg user ratings",
            labels={'average_tb_sentiment_rating': 'Average TextBlob Sentiment Rating', 'tb_sentiment_rating': 'TextBlob Sentiment Rating'},
            opacity=0.7
        )

        return price_histogram, rating_histogram, num_ratings_histogram, tb_sentiment_piechart, sentiment_scatterplot, sentiment_piechart

    return {}, {}, {}, {}, {}, {}

if __name__ == "__main__":
    app.run_server(debug=True)
