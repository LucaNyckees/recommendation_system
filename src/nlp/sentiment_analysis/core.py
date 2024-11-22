from src.nlp.sentiment_analysis.classifiers import SentimentClassifier
from src.log.logger import logger


def sentiment_classifier_pipeline(category: str, embedding: str, frac: float) -> None:
    """
    Trains and stores two models per category.
    TODO : actually store models with MLFlow
    """
    logger.info(f"Running sentiment classifiers pipeline | category={category}, embedding={embedding}, frac={frac}")
    assert embedding in {"tf-idf", "bert"}, "embedding not valid"

    xgb_classifier = SentimentClassifier(
        category=category,
        embedding=embedding,
        model_class="xgb",
        frac=frac
    )
    xgb_classifier._initialize_data()
    xgb_classifier._train()
    xgb_classifier._analyse()
    xgb_classifier._make_figures()

    rf_classifier = SentimentClassifier(
        category=category,
        embedding=embedding,
        model_class="rf",
        frac=frac
    )
    rf_classifier._initialize_data()
    rf_classifier._train()
    rf_classifier._analyse()
    rf_classifier._make_figures()

    return None
