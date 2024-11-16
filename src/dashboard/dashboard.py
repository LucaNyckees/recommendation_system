from dash import Dash, html, dcc, Input, Output
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
from src.dashboard.helpers import darkmode_layout


# Define a color map to match the specified palette
color_palette = {
    "light_blue": "#4cc9f0",
    "dark_blue": "#4361ee",
    "violet": "#3a0ca3",
    "light_violet": "#7209b7",
    "accent_pink": "#f72585"
}

color_map = {
    "positive": color_palette["light_blue"],
    "neutral": color_palette["light_violet"],
    "negative": color_palette["accent_pink"]
}

# Load the data from the database
with connect(db_key="main") as conn:
    with conn.cursor() as cur:
        df = get_amazon_dataframe(cur=cur, categories=None, limit=10000)
        df["review_input"] = df["title"] + df["text"]
        df = apply_sentiment_analysis(df=df)

app = Dash(__name__)

# App layout with updated styles for color coherence
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
        className="radio-buttons",
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
            # html.H3("Data Visualization"),
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
        price_histogram = px.histogram(df, x='price', nbins=100, title="Price Distribution",
                                       color_discrete_sequence=[color_palette["dark_blue"]])
        price_histogram = darkmode_layout(fig=price_histogram, sublib="px")

        # Overlayed Histogram for average rating and average TextBlob sentiment rating
        rating_histogram = go.Figure()
        
        rating_histogram.add_trace(
            go.Histogram(
                x=df['average_rating'],
                nbinsx=60,
                name='User',
                opacity=0.75,
                marker_color=color_palette["light_blue"]
            )
        )
        
        rating_histogram.add_trace(
            go.Histogram(
                x=df['average_tb_sentiment_rating'],
                nbinsx=60,
                name='TextBlob',
                opacity=0.75,
                marker_color=color_palette["light_violet"],
            )
        )
        
        rating_histogram.update_layout(
            title="Average Rating Distribution",
            xaxis_title="Rating",
            yaxis_title="Count",
            barmode='overlay',
        )
        rating_histogram = darkmode_layout(fig=rating_histogram, sublib="go")

        # Histogram for number of ratings
        num_ratings_histogram = px.histogram(df, x='rating_number', nbins=100, title="Number of Ratings Distribution",
                                             color_discrete_sequence=[color_palette["dark_blue"]])
        num_ratings_histogram = darkmode_layout(fig=num_ratings_histogram, sublib="px")

        # Pie charts for sentiment
        tb_sentiment_counts = df['tb_sentiment_category'].value_counts().reset_index()
        tb_sentiment_counts.columns = ['tb_sentiment_category', 'count']
        tb_sentiment_piechart = px.pie(
            tb_sentiment_counts,
            names='tb_sentiment_category',
            values='count',
            title="TextBlob Sentiments",
            color="tb_sentiment_category",
            color_discrete_map=color_map,
            hole=0.63
        )
        tb_sentiment_piechart = darkmode_layout(fig=tb_sentiment_piechart, sublib="px")

        sentiment_counts = df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment_category', 'count']
        sentiment_piechart = px.pie(
            sentiment_counts,
            names='sentiment_category',
            values='count',
            title="User Sentiments",
            color="sentiment_category",
            color_discrete_map=color_map,
            hole=0.63
        )
        sentiment_piechart = darkmode_layout(fig=sentiment_piechart, sublib="px")

        # Scatter plot for average_tb_sentiment_rating vs tb_sentiment_rating
        sentiment_scatterplot = px.scatter(
            df,
            x='average_tb_sentiment_rating',
            y='tb_sentiment_rating',
            title="Average TextBlob vs. Average User Ratings",
            labels={'average_tb_sentiment_rating': 'Average TextBlob Sentiment Rating', 'tb_sentiment_rating': 'TextBlob Sentiment Rating'},
            opacity=0.7,
            color_discrete_sequence=[color_palette["dark_blue"]]
        )
        sentiment_scatterplot = darkmode_layout(fig=sentiment_scatterplot, sublib="px")

        return price_histogram, rating_histogram, num_ratings_histogram, tb_sentiment_piechart, sentiment_scatterplot, sentiment_piechart

    return {}, {}, {}, {}, {}, {}

if __name__ == "__main__":
    app.run_server(debug=True)
