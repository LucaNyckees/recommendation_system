from textblob import TextBlob
import pandas as pd
# from nltk.corpus import stopwords
from typing import Tuple
# import nltk
# nltk.download('stopwords')

import nltk
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
# stop_words = set(stopwords.words('english'))


def homeomorphic_interval_map(value: float, interval_in: Tuple[float, float], interval_out: Tuple[float, float]) -> float:
    a1, b1 = interval_in
    a2, b2 = interval_out
    assert a1 < b1 and a2 < b2 and (a1 <= value <= b1), f"input is pathological, got (a1,b1)=({a1},{b1}), (a2,b2)=({a2},{b2}), value={value}"
    mapped_value = a2 + ((value - a1) / (b1 - a1)) * (b2 - a2)
    return mapped_value


# def remove_stopwords(text: str) -> str:
#     return " ".join([word for word in text.split() if word not in stop_words])


def get_textblob_sentiment_rating(text: str) -> float:
    blob = TextBlob(text)
    return blob.sentiment.polarity


def map_rating_to_sentiment(rating: float, min_rating: float, max_rating: float) -> str:
    threshold_rating = (max_rating + min_rating) / 2
    delta = (max_rating - min_rating) / 5
    if rating < threshold_rating - delta / 2:
        return "negative"
    elif rating > threshold_rating + delta / 2:
        return "positive"
    else:
        return "neutral"


def apply_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates two columns 'tb_sentiment_rating' and 'tb_sentiment_category'.
    """
    # normalize user ratings to interval (0, 100)
    df["rating"] = df["rating"].apply(homeomorphic_interval_map, interval_in=(1, 5), interval_out=(0, 100))
    df["average_rating"] = df["average_rating"].apply(homeomorphic_interval_map, interval_in=(1, 5), interval_out=(0, 100))
    # extrac user sentiment category
    df["sentiment_category"] = df["rating"].apply(map_rating_to_sentiment, min_rating=0, max_rating=100)
    # extract textblob sentiment ratings
    df["tb_sentiment_rating"] = df["review_input"].apply(get_textblob_sentiment_rating)
    # normalize textblob sentiment ratings to interval (0, 100)
    df["tb_sentiment_rating"] = df["tb_sentiment_rating"].apply(homeomorphic_interval_map, interval_in=(-1, 1), interval_out=(0, 100))
    df["average_tb_sentiment_rating"] = df.groupby('parent_asin', as_index=False).agg(val=('tb_sentiment_rating', 'mean'))["val"]
    # extract textblob sentiment category
    df["tb_sentiment_category"] = df["tb_sentiment_rating"].apply(map_rating_to_sentiment, min_rating=0, max_rating=100)
    return df
