from src.nlp.sentiment_analysis.classifiers import XGBoostSentimentClassifier, RandomForestSentimentClassifier
from src.log.logger import logger


def sentiment_classifier_pipeline(category: str, embedding: str, frac: float) -> None:
    """
    Trains and stores two models per category.
    TODO : actually store models with MLFlow
    """
    logger.info(f"Running sentiment classifiers pipeline | category={category}, embedding={embedding}, frac={frac}")
    assert embedding in {"tf-idf", "bert"}, "embedding not valid"

    xgb_sentiment_classifier = XGBoostSentimentClassifier(
        category=category,
        embedding=embedding,
        frac=frac
    )
    xgb_sentiment_classifier._initialize_data()
    xgb_sentiment_classifier._train()
    xgb_sentiment_classifier._analyse()
    xgb_sentiment_classifier._make_figures()

    rf_sentiment_classifier = RandomForestSentimentClassifier(
        category=category,
        embedding=embedding,
        frac=frac
    )
    rf_sentiment_classifier._initialize_data()
    rf_sentiment_classifier._train()
    rf_sentiment_classifier._analyse()
    rf_sentiment_classifier._make_figures()

    return None
