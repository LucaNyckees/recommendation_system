from pydantic import BaseModel


class CategorySummary(BaseModel):
    name: str = ""
    total_price: float = 0.0
    total_nb_ratings: int = 0


class ChatQueryInput(BaseModel):
    text: str


class ChatQueryOutput(BaseModel):
    input: str
    output: str
    intermediate_steps: list[str]
