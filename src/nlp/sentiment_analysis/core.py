from src.nlp.sentiment_analysis.classifiers import SentimentClassifier
from src.log.logger import logger


def sentiment_classifier_pipeline(category: str, embedding: str, frac: float) -> None:
    """
    """
    logger.info(f"Running sentiment classifiers pipeline | category={category}, embedding={embedding}, frac={frac}")
    assert embedding in {"tf-idf", "bert"}, "embedding not valid"

    classifier_xbgo = SentimentClassifier(
        category=category,
        embedding=embedding,
        model_class="xgb",
        frac=frac
    )
    classifier_xbgo._execute()

    classifier_rfst = SentimentClassifier(
        category=category,
        embedding=embedding,
        model_class="rf",
        frac=frac
    )
    classifier_rfst._execute()

    classifier_bert = SentimentClassifier(
        category=category,
        embedding=None,
        model_class="bert",
        frac=frac
    )
    classifier_bert._execute()

    return None


def sentiment_regressor_pipeline(category: str, embedding: str, frac: float) -> None:
    pass