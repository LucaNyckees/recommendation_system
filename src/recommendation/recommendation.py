from typing import Literal
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.nlp.embedder import EmbedderPipeline


def recommendation_system(
    product_id: str,
    df: pd.DataFrame,
    method: Literal["tf-idf", "bert"],
    tune: bool = False,
    top_n: int = 10,
) -> pd.DataFrame:
    
    if method == "tf-idf":
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1, 1))
        matrix = vectorizer.fit_transform(df["title"])
    elif method == "bert":
        pipeline = EmbedderPipeline()
        pipeline._opt_setup()
        if tune:
            pipeline._train()
        df["embeddings"] = df["title"].apply(
            lambda text: pipeline._get_embedding(text=text, max_len=32),
        )
        matrix = np.vstack(df['embeddings'].apply(lambda x: x.squeeze(0)).values)
        
    cosine_sim = cosine_similarity(matrix)
    idx = df.index[df['asin'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    content_recommendations_idx = [i[0] for i in sim_scores[1:top_n+1]]
    recommendations = df.loc[content_recommendations_idx]
    return recommendations
