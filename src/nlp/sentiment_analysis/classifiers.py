from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from src.paths import FIGURES_PATH
from src.processing import DataProcessor
from src.visualization import make_confusion_matrix
from src.log.logger import logger


class XGBoostSentimentClassifier:

    def __init__(self) -> None:
        self.trained = False
        self.model_name = f"xgboost-sentiment-classifier"

    def _initialize_data(self) -> None:
        self.data_processor = DataProcessor()
        self.data_processor._load(category="All_beauty", frac=0.01)
        self.data_processor._process_reviews(clean_text=False)
        self.data_processor._embedd_reviews_and_split(embedding="tf-idf")

    def _train(self) -> None:
        self.xgb = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)
        self.xgb.fit(self.data_processor.X_train, self.data_processor.y_train.tolist())
        self.y_pred = self.xgb.predict(self.data_processor.X_test)
        self.trained = True

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


class RandomForestSentimentClassifier:
    pass