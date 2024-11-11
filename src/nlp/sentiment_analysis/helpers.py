from textblob import TextBlob
import pandas as pd
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
stop_words = set(stopwords.words('english'))


def remove_stopwords(text: str) -> str:
    return " ".join([word for word in text.split() if word not in stop_words])


def get_sentiment(text: str) -> float:
    blob = TextBlob(text)
    return blob.sentiment.polarity


def apply_reviews_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df["sentiment"] = df["review_input"].apply(get_sentiment)
    return df


def map_rating_to_sentiment(rating: float, min_rating: float = 1.0, max_rating: float = 5.0) -> str:
    threshold_rating = (max_rating + min_rating) / 2
    if rating < threshold_rating:
        return "negative"
    elif rating > threshold_rating:
        return "positive"
    else:
        return "neutral"
