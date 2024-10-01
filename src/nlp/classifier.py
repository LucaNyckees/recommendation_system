import random
from datetime import datetime
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
import pandas as pd
import json

from src.paths import RESULTS_PATH, RESOURCES_PATH
from src.log.logger import logger

with open(RESOURCES_PATH / "params.json") as f:
    classifier_params = json.load(f)["classifier"]


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

id2label = {x: str(x) for x in range(1, 6)}
label2id = {str(x): x for x in range(1, 6)}


class BERTClassifier:
    def __init__(self, df: pd.DataFrame):
        for k, v in classifier_params.items():
            setattr(self, k, v)
        self.output_dir = RESULTS_PATH / f"{datetime.today().strftime('%Y-%m-%d')}"

        torch.manual_seed(self.seed)
        random.seed(self.seed)

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            num_train_epochs=self.epochs,
            seed=self.seed,
            learning_rate=self.lr,
        )

        ## Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            max_length=512,
            truncation=True,
            clean_up_tokenization_spaces=False,
        )

        def _preprocess_fn(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
            )

        ## Dataset
        dataset = Dataset.from_pandas(df)
        n = len(dataset)
        sizes = [int(0.8 * n), int(0.1 * n), n - int(0.8 * n) - int(0.1 * n)]

        gen = torch.Generator().manual_seed(self.seed_data_split)

        sets = (
            list(map(lambda x: {"text": x["text"], "label": x["label"]}, x))
            for x in torch.utils.data.random_split(dataset, sizes, generator=gen)
        )

        self.train_set, self.eval_set, self.test_set = (
            Dataset.from_list(x).map(_preprocess_fn, batched=True) for x in sets
        )

        split_dict = {
            "train_set": len(self.train_set),
            "eval_set": len(self.eval_set),
            "test_set": len(self.test_set),
        }
        logger.info(f"split : {split_dict}")

        ## Model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        ).to(device)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_set,
            eval_dataset=self.eval_set,
            tokenizer=self.tokenizer,
        )

    def train(self):
        self.trainer.train()

    def evaluate(self):
        self.metrics = self.trainer.evaluate()

    def predict(self):
        classifier = pipeline(
            task=self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            device=device,
        )
        for datapoint in tqdm(self.test_set):
            text = datapoint["text"]
            label = datapoint["label"]
            prediction = classifier(text)
            logger.info(f"Text: {text}")
            logger.info(f"True label: {label}")
            logger.info(f"Predicted label: {prediction}")

    # def push_to_hub(self):
    #     self.model.push_to_hub(self.hf_model_name, use_temp_dir=True)
    #     self.tokenizer.push_to_hub(self.task, use_temp_dir=True)
    def push_to_hub(self) -> None:
        hf_model_name = "LucaNyckees/amazon-bert-classifier"
        self.model.push_to_hub(hf_model_name, use_temp_dir=True)
