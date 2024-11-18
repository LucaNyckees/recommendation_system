from dash import Dash, html, dcc, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import requests

ROOT_DIR = ("/").join(os.getcwd().split("/"))
import sys
sys.path.append(ROOT_DIR)

from src.database.connection import connect
from src.database.db_functions import get_amazon_dataframe
from src.nlp.sentiment_analysis.helpers import apply_sentiment_analysis
from src.dashboard.helpers import apply_layout, get_route_result

suppress_callback_exceptions=True

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

components_style = {"flex": "1", "border": "1px solid gray", "margin-left": "20px", "margin-right": "20px"}

# Load the data from the database
with connect(db_key="main") as conn:
    with conn.cursor() as cur:
        df = get_amazon_dataframe(cur=cur, categories=None, limit=10000)
        df["review_input"] = df["title"] + df["text"]
        df = apply_sentiment_analysis(df=df)


data_summary = get_route_result(url="http://127.0.0.1:8000/dashboard/all_categories/table_summary")
marimekko_data = get_route_result(url="http://127.0.0.1:8000/dashboard/all_categories/marimekko_price_volume")
transaction_time_series = get_route_result(url="http://127.0.0.1:8000/dashboard/all_categories/transaction_volume_time_series")
avg_ratings_data = get_route_result(url="http://127.0.0.1:8000/dashboard/all_categories/avg_rating")


app = Dash(__name__)

# App layout with updated styles for color coherence
app.layout = html.Div([
    html.H1("RecSys Results Dashboard"),
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
    html.Hr(),
    html.Div(id='section-content'),
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
            html.Div(
                dash_table.DataTable(
                    id='data-table',  # Assign an ID for later updates
                    style_table={**{'overflowX': 'auto', 'margin-bottom': '20px'}, **components_style},
                    style_cell={'textAlign': 'left'},
                    data=data_summary,
                ),
                style={'margin-bottom': '20px'}
            ),
            html.Div([
                dcc.Graph(id='avg-rating-barplot', style=components_style),
                dcc.Graph(id='sentiment-scatterplot', style=components_style),
                dcc.Graph(id='tb-sentiment-piechart', style=components_style),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
            html.Div([
                dcc.Graph(id='transactions-ts-chart', style=components_style),
            ], style={'display': 'flex', 'flex-direction': 'row', 'margin-top': '20px'}),
            html.Div([
                dcc.Graph(id='marimekko-chart', style=components_style),
            ], style={'display': 'flex', 'flex-direction': 'row', 'margin-top': '20px'}),
        ])
    else:
        return html.Div("Selected section: " + selected_section)


# Callback to update graphs in the Data Visualization section
@app.callback(
    Output("data-table", "table"),
    Output('avg-rating-barplot', 'figure'),
    Output('tb-sentiment-piechart', 'figure'),
    Output('sentiment-scatterplot', 'figure'),
    Output("marimekko-chart", "figure"),
    Output("transactions-ts-chart", "figure"),
    Input('section-radio', 'value')
)
def update_graphs(selected_section):
    if selected_section == 'DataViz':

        avg_ratings_fig = px.bar(avg_ratings_data, x="main_category", y="average_rating")

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
        rating_histogram = apply_layout(fig=rating_histogram, sublib="go")

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
        tb_sentiment_piechart = apply_layout(fig=tb_sentiment_piechart, sublib="px")

        # Scatter plot for average_tb_sentiment_rating vs tb_sentiment_rating
        sentiment_scatterplot = px.scatter(
            df,
            x='average_tb_sentiment_rating',
            y='tb_sentiment_rating',
            title="Average TextBlob vs. Average User Ratings",
            labels={'average_tb_sentiment_rating': 'Average TextBlob Sentiment Rating', 'tb_sentiment_rating': 'TextBlob Sentiment Rating'},
            opacity=0.7,
            color_discrete_sequence=[color_palette["dark_blue"]],
            trendline="ols",
        )
        sentiment_scatterplot = apply_layout(fig=sentiment_scatterplot, sublib="px")

        df_marimekko = pd.DataFrame(marimekko_data)

        # Normalize the data for better visualization
        df_marimekko['relative_price'] = df_marimekko['total_price'] / df_marimekko['total_price'].sum()
        df_marimekko['relative_rating'] = df_marimekko['total_rating_number'] / df_marimekko['total_rating_number'].sum()

        # Create the Marimekko chart using Treemap
        marimekko_fig = go.Figure(
            go.Treemap(
                labels=df_marimekko['main_category'],
                parents=[""] * len(df_marimekko),  # Treemap needs a parent-child relationship
                values=df_marimekko['total_volume'],  # Define the size of each rectangle
                customdata=df_marimekko[['total_price', 'total_rating_number']],
                texttemplate=(
                    "<b>%{label}</b><br>"
                    "Total Price: %{customdata[0]:,.2f}<br>"
                    "Total Ratings: %{customdata[1]:,.0f}<br>"
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Total Price: %{customdata[0]:,.2f}<br>"
                    "Total Ratings: %{customdata[1]:,.0f}<br>"
                ),
                marker=dict(
                    colors=list(color_palette.values())[:len(df_marimekko['main_category'].unique())]
                )
            )
        )

        marimekko_fig.update_layout(
            title="Transaction Volume Marimekko Chart",
            margin=dict(t=50, l=25, r=25, b=25)
        )

        df_transaction_time_series = pd.DataFrame(transaction_time_series)

        # Ensure the 'month' column is in datetime format
        df_transaction_time_series['month'] = pd.to_datetime(df_transaction_time_series['month'])

        # Create the line chart using Plotly Express
        fig_transaction_time_series = px.line(
            df_transaction_time_series,
            x='month',
            y='total_volume',
            title="Monthly Total Volume Over Time",
            labels={'month': 'Month', 'total_volume': 'Total Volume'},
            markers=True  # Adds markers to the line for better visibility
        )

        # Update layout for better visualization
        fig_transaction_time_series.update_layout(
            xaxis_title="Month",
            yaxis_title="Total Volume",
            margin=dict(t=50, l=25, r=25, b=25)
        )

        return (
            data_summary,
            avg_ratings_fig,
            tb_sentiment_piechart,
            sentiment_scatterplot,
            marimekko_fig,
            fig_transaction_time_series,
        )
    return {}, {}, {}, {}, {}, {}

if __name__ == "__main__":
    app.run_server(debug=True)
