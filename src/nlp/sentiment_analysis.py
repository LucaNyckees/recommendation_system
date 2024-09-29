from textblob import TextBlob
import pandas as pd


def get_sentiment(text: str) -> float:
    blob = TextBlob(text)
    return blob.sentiment.polarity


def apply_reviews_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["review_input"].apply(get_sentiment)
    return df
