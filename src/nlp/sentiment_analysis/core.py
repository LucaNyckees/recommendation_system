from src.nlp.sentiment_analysis.classifiers import SentimentClassifier
from src.log.logger import logger


def sentiment_classifier_pipeline(category: str, embedding: str, frac: float) -> None:
    """
    """
    logger.info(f"Running sentiment classifiers pipeline | category={category}, embedding={embedding}, frac={frac}")
    assert embedding in {"tf-idf", "bert"}, "embedding not valid"

    xgb_classifier = SentimentClassifier(
        category=category,
        embedding=embedding,
        model_class="xgb",
        frac=frac
    )
    xgb_classifier._execute()

    rf_classifier = SentimentClassifier(
        category=category,
        embedding=embedding,
        model_class="rf",
        frac=frac
    )
    rf_classifier._execute()

    return None
