import pandas as pd
import spacy
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
from src.nlp.sentiment_analysis.helpers import map_rating_to_sentiment
from src.database.connection import connect
from src.database.db_functions import load_dataframe_from_query


class DataProcessor:

    def __init__(self) -> None:
        self.conn = connect()
        self.cur = self.conn.cursor()

    def _load(self, category: str, frac: float = 0.01) -> None:
        """
        :param category: amazon product category, e.g. "All_Beauty"
        :param frac: fraction with wich data sampling is done
        """
        query = SQL("""
            SELECT *
            FROM rs_amazon_products p
            INNER JOIN rs_amazon_reviews r ON p.parent_asin = r.parent_asin
            WHERE main_category = %(main_category)s
            TABLESAMPLE BERNOULLI(%(proportion)s)""")
        df = load_dataframe_from_query(cur=self.cur, query=query, params={"main_category": category, "proportion": frac * 100})
        logger.info(f"loaded {len(df)} rows")

    def _clean_text(text: str) -> str:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        return " ".join(cleaned_tokens)

    def _process_reviews(self, clean_text: bool) -> None:
        self.df.rename(columns={"title_x": "review_title", "title_y": "title", "text": "review"}, inplace=True)
        self.df["review_input"] = self.df["review_title"] + self.df["review"]
        if clean_text:
            self.df["cleaned_review_input"] = self.df["review_input"].apply(self._clean_text)
        self.df["sentiment"] = self.df["rating"].apply(lambda x: map_rating_to_sentiment(rating=x))
    
    def _embedd_reviews_and_split(self, embedding: str) -> None:

        if embedding == "tf-idf":
            X = self.df["review_input"].to_list()
            y = self.df["sentiment"]

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
            y = self.df["sentiment"]

            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        else:
            raise NotImplementedError(f"Embedding {embedding} not treated, should be 'tf-idf' or 'bert'.")

