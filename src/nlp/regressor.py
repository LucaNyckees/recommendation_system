import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import plotly.express as px
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    BertPreTrainedModel,
    BertModel,
)
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import json
import os
from rich.progress import track
from src.paths import RESULTS_PATH, RESOURCES_PATH, IMAGES_PATH
from src.log.logger import logger
from src.data_loader import load_reviews
from src.processing import reviews_processing


with open(RESOURCES_PATH / "params.json") as f:
    classifier_params = json.load(f)["classifier"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

id2label = {x: str(x) for x in range(1, 6)}
label2id = {str(x): x for x in range(1, 6)}

MODEL_OUT_DIR = "../results/readabilityprize/bert_regressor"


class ReviewsDataset(Dataset):
    def __init__(self, data, maxlen, tokenizer):
        self.df = data.reset_index()
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        review = self.df.loc[index, "text"]
        target = self.df.loc[index, "target"]
        # identifier = self.df.loc[index, "id"]
        tokens = self.tokenizer.tokenize(review)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        if len(tokens) < self.maxlen:
            tokens = tokens + ["[PAD]" for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[: self.maxlen - 1] + ["[SEP]"]
        # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()

        target = torch.tensor(target, dtype=torch.float32)

        return input_ids, attention_mask, target

    def _getinfo(self, index):
        review = self.df.loc[index, "text"]
        target = self.df.loc[index, "target"]
        return review, target


class BertRegressor(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(config.hidden_size, 128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128, 128)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(128, 1)
        self.sigm = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        output = self.sigm(output)
        return output


class BertRegressorPipeline:
    def __init__(self, df: pd.DataFrame):
        for k, v in classifier_params.items():
            setattr(self, k, v)
        self.output_dir = RESULTS_PATH / f"{datetime.now().strftime('%m/%d/%Y-%H:%M:%S')}"
        self.images_dir = IMAGES_PATH / "bert-regressor" / f"{datetime.now().strftime('%m/%d/%Y-%H:%M:%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        self.df = df

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            max_length=512,
            truncation=True,
            clean_up_tokenization_spaces=False,
        )

    def _prepare_data(self, debug: bool = False):
        self.df_train, temp = train_test_split(self.df[["text", "target"]], test_size=0.4, random_state=21)
        self.df_test, self.df_validation = train_test_split(temp[["text", "target"]], test_size=0.5, random_state=21)

        split_dict = {
            "train_set": len(self.df_train),
            "eval_set": len(self.df_validation),
            "test_set": len(self.df_test),
        }
        logger.info(f"{split_dict}")
        if debug:
            self.df_train = self.df_train.sample(2)  # train on a single entry in debug mode
        self.train_set = ReviewsDataset(data=self.df_train, maxlen=self.max_len_train, tokenizer=self.tokenizer)
        self.validation_set = ReviewsDataset(
            data=self.df_validation, maxlen=self.max_len_valid, tokenizer=self.tokenizer
        )
        self.test_set = ReviewsDataset(data=self.df_test, maxlen=self.max_len_test, tokenizer=self.tokenizer)

        self.train_loader = DataLoader(dataset=self.train_set, batch_size=self.batch_size, num_workers=self.num_threads)
        self.validation_loader = DataLoader(
            dataset=self.validation_set, batch_size=self.batch_size, num_workers=self.num_threads
        )

    def _setup_model(self):
        self.model = BertRegressor.from_pretrained(
            self.model_name,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        ).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

        args_dict = dict(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            num_train_epochs=self.num_epochs,
            seed=self.seed,
            learning_rate=self.lr,
        )

        logger.info(args_dict)

        self.training_args = TrainingArguments(**args_dict)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, max_length=512, truncation=True, clean_up_tokenization_spaces=False
        )

    def evaluate(self):
        self.model.eval()
        mean_loss, count = 0, 0

        with torch.no_grad():
            for input_ids, attention_mask, target in self.validation_loader:
                input_ids, attention_mask, target = (input_ids.to(device), attention_mask.to(device), target.to(device))
                output = self.model(input_ids, attention_mask)

                mean_loss += self.criterion(torch.reshape(output, (-1,)), target.type_as(output)).item()
                #             mean_err += get_rmse(output, target)
                count += 1

        return mean_loss / count

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for input_ids, attention_mask, target in track(
                self.train_loader, description=f"epoch {epoch + 1}/{self.num_epochs}"
            ):
                self.optimizer.zero_grad()
                output = self.model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
                loss = self.criterion(torch.reshape(output, (-1,)), target.to(device).type_as(output))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            val_loss = self.evaluate()
            logger.info(f"train_loss:{train_loss/len(self.train_loader)}, val_loss:{val_loss}")

    def get_rmse(self, output, target):
        err = torch.sqrt(metrics.mean_squared_error(target, output))
        return err

    def predict(self):
        predictions = []
        targets = []
        with torch.no_grad():
            for input_ids, attention_mask, target in track(self.validation_loader, description="predicting"):
                output = self.model(input_ids.to(device), attention_mask.to(device))
                predictions += output
                targets += target.to(device)

        return targets, predictions

    def push_to_hub(self) -> None:
        hf_model_name = "LucaNyckees/amazon-bert-classifier"
        self.model.push_to_hub(hf_model_name, use_temp_dir=True)


def regressor_pipeline(category: str = "All_beauty", frac: float = 0.001, debug: bool = False) -> None:
    logger.info("data loading...")
    df = load_reviews(category=category, frac=frac)

    logger.info("data processing...")
    df = reviews_processing(df=df, clean_text=False)
    sub = df.rename(columns={"rating": "target", "review_input": "text"})[["target", "text"]]
    sub["target"] = sub["target"] / 5  # normalize product ratings to [0,1]

    logger.info("initializing model...")
    bert_pipeline = BertRegressorPipeline(df=sub)

    logger.info("preparing input data...")
    bert_pipeline._prepare_data(debug=debug)

    logger.info("model setup...")
    bert_pipeline._setup_model()

    logger.info("model training...")
    bert_pipeline.train()

    logger.info("playground...")
    targets, predictions = bert_pipeline.predict()
    fig = px.scatter(x=[t.item() for t in targets], y=[p.item() for p in predictions])
    fig.update_layout(yaxis_range=[0, 1])
    fig.write_image(bert_pipeline.images_dir / "target_v_pred.png")

    return None
