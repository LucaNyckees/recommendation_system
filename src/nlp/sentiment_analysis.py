from textblob import TextBlob
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def remove_stopwords(text: str) -> str:
    return " ".join([word for word in text.split() if word not in stop_words])


def get_sentiment(text: str) -> float:
    blob = TextBlob(text)
    return blob.sentiment.polarity


def apply_reviews_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["review_input"].apply(get_sentiment)
    return df
