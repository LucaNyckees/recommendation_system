from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import mlflow
import torch
import datetime
import os
import toml
import json
import random
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

from src.paths import FIGURES_PATH, ROOT, RESOURCES_PATH
from src.processing import DataProcessor
from src.visualization import make_confusion_matrix_plotly
from src.log.logger import logger

with open(os.path.join(ROOT, "config.toml"), "r") as f:
    config = toml.load(f)

with open(RESOURCES_PATH / "sentiment_classifiers_params.json") as f:
    model_params = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class SentimentClassifier:
    def __init__(self, category: str, embedding: str | None, model_class: str, frac: float = 0.01) -> None:
        """
        :param category: any of Amazon product categories, e.g. "All_Beauty"
        :param emebdding: the text to vectors embedding, can be "bert", "tf-idf" or None
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

    def _setup_model(self) -> None:
        self.training_args = model_params[self.model_class]
        mlflow.log_params(self.training_args)
        match self.model_class:
            case "xbg":
                self.model = XGBClassifier(**self.training_args)
            case "rf":
                self.model = RandomForestClassifier(**self.training_args)
            case "bert":
                torch.manual_seed(self.seed)
                random.seed(self.seed)
                self.training_args = TrainingArguments(self.training_args)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    max_length=512,
                    truncation=True,
                    clean_up_tokenization_spaces=False,
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=3,
                    id2label={1: "positive", 2: "negative", 3: "neutral"},
                    label2id={"positive": 1, "negative": 2, "neutral": 3},
                ).to(device)
        logger.info(f"Building model with args {self.training_args}")

    def _train(self) -> None:
        match self.model_class:
            case "xbg" | "rf":
                self.model.fit(self.data_processor.X_train, self.data_processor.y_train.tolist())
                self.y_pred = self.model.predict(self.data_processor.X_test)
            case "bert":
                torch.manual_seed(self.seed)
                random.seed(self.seed)
                self.trainer = Trainer(
                    model=self.model,
                    args=self.training_args,
                    train_dataset=self.train_set,
                    eval_dataset=self.eval_set,
                    tokenizer=self.tokenizer,
                )
                self.trainer.train()
        self.trained = True
        logger.info("Trained model")

    def _predict(self) -> None:
        match self.model_class:
            case "xbg" | "rf":
                self.y_pred = self.model.predict(self.data_processor.X_test)
            case "bert":
                predictions_output = self.trainer.predict(self.test_set)
                logits = predictions_output.predictions
                y_pred = logits.argmax(axis=-1)
                self.y_pred = y_pred.tolist()

    def _analyse(self) -> None:
        self.report = classification_report(
            y_true=self.data_processor.y_test,
            y_pred=self.y_pred,
            target_names=self.data_processor.label_encoder.classes_,
        )
        if self.model_class == "bert":
            self.metrics = self.trainer.evaluate()
            logger.ingo(self.metrics)
            mlflow.log_metrics(self.metrics)
        logger.info(self.report)
        mlflow.log_metrics(self.report)

    def _make_figures(self) -> None:
        fig = make_confusion_matrix_plotly(
            y_test=self.data_processor.y_test,
            y_pred=self.y_pred,
            classes=self.data_processor.label_encoder.classes_,
            file_path=FIGURES_PATH / self.model_name / "confusion_matrix.png"
        )
        mlflow.log_figure(figure=fig)

    def _push_to_hub(self) -> None:
        if self.model_class == "bert":
            hf_model_name = "LucaNyckees/amazon-bert-classifier"
            self.model.push_to_hub(hf_model_name, use_temp_dir=True)

    def _execute(self) -> None:
        mlflow_run_name = f"{self.model_name}-{datetime.datetime.now()}"
        description = f"Sentiment Classification with Model {self.model_class}"
        mlflow.set_tracking_uri(f"http://{config['mlflow']['host']}:{config['mlflow']['port']}")
        tags = {"mode": "prod"}
        mlflow.end_run()
        with mlflow.start_run(run_name=mlflow_run_name, description=description, tags=tags):
            self._initialize_data()
            self._setup_model()
            self._train()
            self._predict()
            self._analyse()
            self._make_figures()
            self._push_to_hub()

            self.artifact_path = self.model_name
            self.run_id = mlflow.active_run().info.run_id
            self.uri = f"runs:/{self.run_id}/{self.artifact_path}"

            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path=self.model_name,
                registered_model_name=self.model_name,
            )
