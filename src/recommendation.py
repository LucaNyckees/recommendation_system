from typing import Literal
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recommendation_system(product_id: str, df: pd.DataFrame, method: Literal["tf-idf", "bert"], top_n: int = 10) -> pd.DataFrame:
    
    if method == "tf-idf":
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1, 1))
        matrix = vectorizer.fit_transform(df['review_input'])
    elif method == "bert":
        raise NotImplementedError("BERT method is not ready yet.")
        
    cosine_sim = cosine_similarity(matrix)
    idx = df.index[df['asin'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    content_recommendations_idx = [i[0] for i in sim_scores[1:top_n+1]]
    recommendations = df.loc[content_recommendations_idx]
    return recommendations
