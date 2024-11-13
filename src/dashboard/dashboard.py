from dash import Dash, html, dcc, Input, Output
import pandas as pd
import random  # For demonstration, simulate recommendation results
import plotly.express as px
import pandas as pd


# Simulated data
users = ['user_1', 'user_2', 'user_3']  # Replace with your actual users
products = ['product_a', 'product_b', 'product_c']  # Replace with actual product IDs
recommendations = {
    'user_1': [('product_a', 4.5, 'positive'), ('product_b', 3.2, 'neutral')],
    'user_2': [('product_c', 2.5, 'negative'), ('product_a', 4.2, 'positive')],
    'user_3': [('product_b', 5.0, 'positive'), ('product_c', 3.8, 'neutral')]
}

app = Dash(__name__)

app.layout = html.Div([
    html.H1("User Recommendations"),
    dcc.Input(id='user_id', type='text', placeholder='Enter User ID'),
    html.Button('Get Recommendations', id='submit', n_clicks=0),
    html.Div(id='recommendation_output')
])

@app.callback(
    Output('recommendation_output', 'children'),
    Input('submit', 'n_clicks'),
    Input('user_id', 'value')
)
def recommend_products(n_clicks, user_id):
    if n_clicks > 0 and user_id in recommendations:
        product_recs = recommendations[user_id]
        return html.Ul([html.Li(f"{p[0]} - Rating: {p[1]} - Sentiment: {p[2]}") for p in product_recs])
    else:
        return "No recommendations available for this user."


# Example similarity data
similar_products = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D', 'E'],
    'Similarity_Score': [0.9, 0.8, 0.6, 0.4, 0.3],
    'Sentiment': [4.5, 3.8, 2.1, 4.0, 3.5]
})

fig = px.scatter(similar_products, x='Similarity_Score', y='Sentiment',
                 text='Product', title="Product Similarity and Sentiment")
app.layout.children.append(dcc.Graph(figure=fig))

similar_users = {
    'user_1': ['user_2', 'user_3'],
    'user_2': ['user_1', 'user_3'],
    'user_3': ['user_1', 'user_2']
}

@app.callback(
    Output('similar_users_output', 'children'),
    Input('user_id', 'value')
)
def show_similar_users(user_id):
    if user_id in similar_users:
        similar = similar_users[user_id]
        return html.Ul([html.Li(f"Similar User: {u}") for u in similar])
    else:
        return "No similar users found."
    

# Example sentiment data
sentiment_data = pd.DataFrame({
    'Sentiment': ['Positive', 'Neutral', 'Negative'],
    'Count': [120, 80, 50]
})

fig_sentiment = px.pie(sentiment_data, names='Sentiment', values='Count', title='Sentiment Distribution')
app.layout.children.append(dcc.Graph(figure=fig_sentiment))

# Adding filter controls
app.layout.children.append(html.Div([
    dcc.Slider(id='min_sentiment', min=1, max=5, step=0.5, value=3,
               marks={i: str(i) for i in range(1, 6)}),
    html.Label("Minimum Sentiment Score")
]))

# Update recommendations based on minimum sentiment
@app.callback(
    Output('recommendation_output', 'children'),
    Input('min_sentiment', 'value'),
    Input('user_id', 'value')
)
def recommend_filtered_products(min_sentiment, user_id):
    if user_id in recommendations:
        product_recs = [p for p in recommendations[user_id] if p[1] >= min_sentiment]
        return html.Ul([html.Li(f"{p[0]} - Rating: {p[1]} - Sentiment: {p[2]}") for p in product_recs])
    else:
        return "No recommendations available for this user."

if __name__ == '__main__':
    app.run_server(debug=True)
