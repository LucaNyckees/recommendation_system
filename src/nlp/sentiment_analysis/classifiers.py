from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from src.paths import FIGURES_PATH
from src.processing import DataProcessor
from src.visualization import make_confusion_matrix
from src.log.logger import logger


class SentimentClassifier:
    def __init__(self, category: str, embedding: str, frac: float = 0.01) -> None:
        self.trained = False
        self.category = category
        self.embedding = embedding
        self.frac = frac

    def _initialize_data(self) -> None:
        self.data_processor = DataProcessor()
        self.data_processor._load(category=self.category, frac=self.frac)
        self.data_processor._process_reviews(clean_text=False)
        self.data_processor._embedd_reviews_and_split(embedding=self.embedding)


class XGBoostSentimentClassifier(SentimentClassifier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = "xgb-sentiment-classifier"

    def _train(self) -> None:
        xgb_classifier_args = dict(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)
        self.xgb = XGBClassifier(**xgb_classifier_args)
        logger.info(f"Building model with args {xgb_classifier_args}")
        self.xgb.fit(self.data_processor.X_train, self.data_processor.y_train.tolist())
        self.y_pred = self.xgb.predict(self.data_processor.X_test)
        self.trained = True
        logger.info("Trained model")

    def _analyse(self) -> None:
        if not self.trained:
            logger.warning("Please train your model first.")
        report = classification_report(
            y_true=self.data_processor.y_test,
            y_pred=self.y_pred,
            target_names=self.data_processor.label_encoder.classes_,
        )
        logger.info(report)

    def _make_figures(self) -> None:
        make_confusion_matrix(
            y_test=self.data_processor.y_test,
            y_pred=self.y_pred,
            classes=self.data_processor.label_encoder.classes_,
            file_path=FIGURES_PATH / self.model_name / "confusion_matrix.png"
        )


class RandomForestSentimentClassifier(SentimentClassifier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = "randomforest-sentiment-classifier"

    def _train(self) -> None:
        rf_classifier_args = dict(n_estimators=100, random_state=42)
        self.rf = RandomForestClassifier(rf_classifier_args)
        logger.info(f"Building model with args {rf_classifier_args}")
        self.rf.fit(self.data_processor.X_train, self.data_processor.y_train)
        self.y_pred = self.rf.predict(self.data_processor.X_test)
        self.trained = True
        logger.info("Trained model")

    def _analyse(self) -> None:
        if not self.trained:
            logger.warning("Please train your model first.")
            return
        report = classification_report(
            y_true=self.data_processor.y_test,
            y_pred=self.y_pred,
            target_names=self.data_processor.label_encoder.classes_,
        )
        logger.info(report)

    def _make_figures(self) -> None:
        make_confusion_matrix(
            y_test=self.data_processor.y_test,
            y_pred=self.y_pred,
            classes=self.data_processor.label_encoder.classes_,
            file_path=FIGURES_PATH / self.model_name / "confusion_matrix.png"
        )