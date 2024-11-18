from pydantic import BaseModel


class CategorySummary(BaseModel):
    name: str = ""
    total_price: float = 0.0
    total_nb_ratings: int = 0
