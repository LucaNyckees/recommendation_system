import pandas as pd
import spacy


def clean_text(text: str) -> str:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(cleaned_tokens)


def reviews_processing(df: pd.DataFrame, clean_text: bool) -> pd.DataFrame:
    df.rename(columns={"title_x": "review_title", "title_y": "title", "text": "review"}, inplace=True)
    df["review_input"] = df["review_title"] + df["review"]
    if clean_text:
        df["cleaned_review_input"] = df["review_input"].apply(clean_text)
    return df
