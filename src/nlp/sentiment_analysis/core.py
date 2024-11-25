from src.nlp.sentiment_analysis.classifiers import SentimentClassifier
from src.log.logger import logger


def sentiment_classifier_pipeline(category: str, embedding: str, nb_rows: int) -> None:
    """
    """
    logger.info(f"Running sentiment classifiers pipeline | category={category}, embedding={embedding}, nb_rows={nb_rows}")
    assert embedding in {"tf-idf", "bert"}, "embedding not valid"

    classifier_xbgo = SentimentClassifier(
        category=category,
        embedding=embedding,
        model_class="xgb",
        nb_rows=nb_rows
    )
    classifier_xbgo._execute()

    classifier_rfst = SentimentClassifier(
        category=category,
        embedding=embedding,
        model_class="rf",
        nb_rows=nb_rows
    )
    classifier_rfst._execute()

    classifier_bert = SentimentClassifier(
        category=category,
        embedding=None,
        model_class="bert",
        nb_rows=nb_rows
    )
    classifier_bert._execute()

    return None


def sentiment_regressor_pipeline(category: str, embedding: str, nb_rows: int) -> None:
    pass