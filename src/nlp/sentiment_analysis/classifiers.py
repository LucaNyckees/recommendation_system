from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import mlflow
import datetime
import os
import toml

from src.paths import FIGURES_PATH, ROOT
from src.processing import DataProcessor
from src.visualization import make_confusion_matrix
from src.log.logger import logger

with open(os.path.join(ROOT, "config.toml"), "r") as f:
    config = toml.load(f)


class SentimentClassifier:
    def __init__(self, category: str, embedding: str, model_class: str, frac: float = 0.01) -> None:
        """
        :param category: any of Amazon product categories, e.g. "All_Beauty"
        :param emebdding: the text to vectors embedding, can be "bert" or "tf-idf"
        :param model_cass: the classifier choice, can be "rf", "xgb", or "bert"
        :param frac: proportion of data to sample, e.g. 1% is frac=0.01
        """
        logger.info(f"Initiating {model_class} sentiment classification")
        self.trained = False
        self.category = category
        self.embedding = embedding
        self.model_class = model_class
        self.frac = frac
        self.model_name = f"{model_class}-sentiment-classifier-{self.category}"

    def _initialize_data(self) -> None:
        self.data_processor = DataProcessor()
        self.data_processor._load(category=self.category, frac=self.frac)
        self.data_processor._process_reviews(clean_text=False)
        self.data_processor._embedd_reviews_and_split(embedding=self.embedding)

    def _train(self) -> None:
        if self.model_class == "xgb":
            xgb_classifier_args = dict(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)
            self.xgb = XGBClassifier(**xgb_classifier_args)
            logger.info(f"Building model with args {xgb_classifier_args}")
            self.xgb.fit(self.data_processor.X_train, self.data_processor.y_train.tolist())
            self.y_pred = self.xgb.predict(self.data_processor.X_test)
        elif self.model_class == "rf":
            rf_classifier_args = dict(n_estimators=100, random_state=42)
            self.rf = RandomForestClassifier(rf_classifier_args)
            logger.info(f"Building model with args {rf_classifier_args}")
            self.rf.fit(self.data_processor.X_train, self.data_processor.y_train)
            self.y_pred = self.rf.predict(self.data_processor.X_test)
        else:
            raise ValueError(f"Invalid argument self.model_class={self.model_class}")
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

    def _execute(self) -> None:
        mlflow_run_name = f"{self.model_name}-{datetime.datetime.now()}"
        description = f"Sentiment Classification with Model {self.model_class}"
        mlflow.set_tracking_uri(f"http://{config['mlflow']['host']}:{config['mlflow']['port']}")
        tags = {"mode": "prod"}
        mlflow.end_run()
        with mlflow.start_run(run_name=mlflow_run_name, description=description, tags=tags):
            self._initialize_data()
            self._train()
            self._analyse()
            self._make_figures()
