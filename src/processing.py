import pandas as pd
# import spacy  # potentially not used, could be removed from requirements.txt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from psycopg.sql import SQL

from src.log.logger import logger
from src.nlp.sentiment_analysis.helpers import apply_sentiment_analysis
from src.database.connection import connect
from src.database.db_functions import load_dataframe_from_query


class DataProcessor:

    def __init__(self) -> None:
        self.conn = connect(db_key="main")
        self.cur = self.conn.cursor()

    def _load(self, category: str, nb_rows: int = 10_000) -> None:
        """
        :param category: amazon product category, e.g. "All_Beauty"
        :param nb_rows: number of rows (i.e. number of reviews) to fetch 
        """
        query = SQL("""
            SELECT *
            FROM rs_amazon_products p
            INNER JOIN rs_amazon_reviews r ON p.parent_asin = r.parent_asin
            WHERE main_category = %(main_category)s
            LIMIT %(nb_rows)s""")
        self.df = load_dataframe_from_query(cur=self.cur, query=query, params={"main_category": category, "nb_rows": nb_rows})
        logger.info(f"loaded {len(self.df)} rows")

    # def _clean_text(text: str) -> str:
    #     nlp = spacy.load("en_core_web_sm")
    #     doc = nlp(text)
    #     cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    #     logger.info("Cleaned texts")
    #     return " ".join(cleaned_tokens)

    def _process_reviews(self, clean_text: bool) -> None:
        self.df["review_input"] = self.df["title"] + self.df["text"]
        # if clean_text:
        #     self.df["cleaned_review_input"] = self.df["review_input"].apply(self._clean_text)
        self.df = apply_sentiment_analysis(df=self.df)
        logger.info("Processed reviews")
    
    def _embedd_reviews_and_split(self, embedding: str) -> None:

        if embedding == "tf-idf":
            X = self.df["review_input"].to_list()
            y = self.df["sentiment_category"]

            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            tfidf = TfidfVectorizer(max_features=5000)
            self.X_train = tfidf.fit_transform(self.X_train)
            self.X_test = tfidf.transform(self.X_test)

        elif embedding == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased")

            def get_bert_embedding(text):
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use the embeddings from the [CLS] token
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                return cls_embedding

            embeddings = []

            for text in tqdm(self.df["review_input"], desc="Embedding reviews"):
                embeddings.append(get_bert_embedding(text).numpy())

            X = pd.DataFrame(embeddings)
            y = self.df["sentiment_category"]

            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        elif embedding is None:
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
        logger.info("Embedded reviews")

