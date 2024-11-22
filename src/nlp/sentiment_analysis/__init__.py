import typer
from typing_extensions import Annotated

app = typer.Typer(name="sentiment")


@app.command(name="classifier")
def run_sentiment_classifier_pipeline(
    category: Annotated[str, typer.Option(help="Amazon data category")] = "All_beauty",
    embedding: Annotated[str, typer.Option(help="Word embedding")] = "tf-idf",
    frac: Annotated[float, typer.Option(help="Amazon data category")] = 0.01,
    # debug: Annotated[bool, typer.Option(help="Amazon data category")] = False,
) -> None:
    
    from src.nlp.sentiment_analysis.core import sentiment_classifier_pipeline

    sentiment_classifier_pipeline(category=category, embedding=embedding, frac=frac)
