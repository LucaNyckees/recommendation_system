from src.nlp.sentiment_analysis.classifiers import XGBoostSentimentClassifier, RandomForestSentimentClassifier


def sentiment_classifier_pipeline() -> None:

    xgb_sentiment_classifier = XGBoostSentimentClassifier(
        category="All_beauty",
        embedding="tf-idf",
        frac=0.01
    )
    xgb_sentiment_classifier._initialize_data()
    xgb_sentiment_classifier._train()
    xgb_sentiment_classifier._analyse()
    xgb_sentiment_classifier._make_figures()

    rf_sentiment_classifier = RandomForestSentimentClassifier(
        category="All_beauty",
        embedding="tf-idf",
        frac=0.01
    )
    rf_sentiment_classifier._initialize_data()
    rf_sentiment_classifier._train()
    rf_sentiment_classifier._analyse()
    rf_sentiment_classifier._make_figures()

    return None
