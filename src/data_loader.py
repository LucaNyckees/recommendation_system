import pandas as pd

from src.paths import DATA_PATH
from src.log.logger import logger


def load_reviews(category: str, frac: float = 0.01) -> pd.DataFrame:
    """
    :param category: amazon product category, e.g. "All_Beauty"
    :param frac: fraction with wich data sampling is done
    """
    file_path1 = DATA_PATH / f"{category}.jsonl"
    file_path2 = DATA_PATH / f"meta_{category}.jsonl"
    df_reviews = pd.read_json(file_path1, lines=True)
    df_meta = pd.read_json(file_path2, lines=True)
    df = pd.merge(df_reviews, df_meta, on=["parent_asin"], how="inner")
    df = df.sample(frac=frac).reset_index(drop=True)
    logger.info(f"loaded {len(df)} rows")
    return df
