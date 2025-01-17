from dash import Dash, html, dcc, Input, Output, dash_table, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import toml
import requests

ROOT_DIR = ("/").join(os.getcwd().split("/"))
import sys
sys.path.append(ROOT_DIR)

from src.database.connection import connect
from src.database.db_functions import get_amazon_dataframe
from src.nlp.sentiment_analysis.helpers import apply_sentiment_analysis
from src.dashboard.helpers import apply_layout, get_route_result
from src.paths import ROOT

with open(os.path.join(ROOT, "config.toml"), "r") as f:
    config = toml.load(f)
    fastapi_config = config["apps"]["fastapi"]
    dash_config = config["apps"]["dash"]

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/hospital-rag-agent")

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

components_style = {"flex": "1", "border": "1px solid gray", "margin-left": "10px", "margin-right": "10px", "border-radius": "10px"}

# Load the data from the database
with connect(db_key="main") as conn:
    with conn.cursor() as cur:
        df = get_amazon_dataframe(cur=cur, categories=None, limit=10000)
        df["review_input"] = df["title"] + df["text"]
        df = apply_sentiment_analysis(df=df)


fastapi_route_start = f"http://{fastapi_config['host']}:{fastapi_config['port']}/dashboard/all_categories"

data_summary = get_route_result(url=f"{fastapi_route_start}/table_summary")
marimekko_data = get_route_result(url=f"{fastapi_route_start}/marimekko_price_volume")
transaction_time_series = get_route_result(url=f"{fastapi_route_start}/transaction_volume_time_series")
avg_ratings_data = get_route_result(url=f"{fastapi_route_start}/avg_rating")


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
            html.Div(
                dash_table.DataTable(
                    id='data-table',  # Assign an ID for later updates
                    style_table={**{'overflowX': 'auto', 'margin-bottom': '20px'}, **components_style},
                    style_cell={'textAlign': 'left'},
                    data=data_summary,
                ),
                style={'margin-bottom': '20px', 'margin-right': '20px'}
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
    elif selected_section == "RecomDemo":
        return html.Div([
            html.H2("Amazon Products Recommendation System Chatbot"),
            html.Div(
                "Ask me questions about products: their characteristics, ratings, general reviews, user feedback, and recommendations.",
                style={"margin-bottom": "20px"}
            ),
            dcc.Textarea(
                id="chat-input",
                placeholder="Type your question here...",
                style={"width": "100%", "height": "100px", "margin-bottom": "10px"}
            ),
            html.Button("Submit", id="submit-button", n_clicks=0),
            html.Div(id="chat-history", style={"margin-top": "20px", "border": "1px solid #ccc", "padding": "10px", "max-height": "400px", "overflow-y": "scroll"})
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

        avg_ratings_fig = px.bar(avg_ratings_data, x="main_category", y="average_rating", title="Avg. Rating per Category")
        avg_ratings_fig = apply_layout(fig=avg_ratings_fig, sublib="px")

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

        sentiment_scatterplot = px.scatter(
            df,
            x="average_tb_sentiment_rating",
            y="average_rating",
            marginal_x="box",
            marginal_y="box",
            title="TextBlob vs. User Ratings",
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
        marimekko_fig = apply_layout(fig=marimekko_fig, sublib="go")

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
        fig_transaction_time_series = apply_layout(fig=fig_transaction_time_series, sublib="px")

        return (
            data_summary,
            avg_ratings_fig,
            tb_sentiment_piechart,
            sentiment_scatterplot,
            marimekko_fig,
            fig_transaction_time_series,
        )
    return {}, {}, {}, {}, {}, {}


# Callback to handle chatbot interaction
@app.callback(
    Output("chat-history", "children"),
    Input("submit-button", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history", "children")
)
def update_chat_history(n_clicks, user_input, chat_history):
    if n_clicks > 0 and user_input:
        chat_history = chat_history or []

        # User message
        user_message = html.Div([
            html.Div("User:", style={"font-weight": "bold", "color": "blue"}),
            html.Div(user_input)
        ], style={"margin-bottom": "10px"})

        # Send request to the chatbot API
        try:
            response = requests.post(CHATBOT_URL, json={"text": user_input})
            if response.status_code == 200:
                chatbot_output = response.json().get("output", "No response")
                explanation = response.json().get("intermediate_steps", "No explanation available")
            else:
                chatbot_output = "An error occurred while processing your message. Please try again."
                explanation = "N/A"
        except Exception as e:
            chatbot_output = f"Error: {e}"
            explanation = "N/A"

        # Chatbot response
        bot_message = html.Div([
            html.Div("Chatbot:", style={"font-weight": "bold", "color": "green"}),
            html.Div(chatbot_output),
            html.Div(f"Explanation: {explanation}", style={"font-style": "italic", "color": "gray", "margin-top": "5px"})
        ], style={"margin-bottom": "20px"})

        # Update chat history
        chat_history.extend([user_message, bot_message])
        return chat_history

    return chat_history

if __name__ == "__main__":
    app.run_server(host=dash_config["host"], port=dash_config["port"], debug=True)
