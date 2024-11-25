import typer
from typing_extensions import Annotated

app = typer.Typer(name="sentiment")


@app.command(name="classifiers")
def run_sentiment_classifier_pipeline(
    category: Annotated[str, typer.Option(help="Amazon data category")] = "All Beauty",
    embedding: Annotated[str, typer.Option(help="Word embedding")] = "tf-idf",
    nb_rows: Annotated[int, typer.Option(help="Amazon data category")] = 10_000,
) -> None:
    
    from src.nlp.sentiment_analysis.core import sentiment_classifier_pipeline

    sentiment_classifier_pipeline(
        category=category,
        embedding=embedding,
        nb_rows=nb_rows,
    )


@app.command(name="regressors")
def run_bert_regressor_pipeline(
    category: Annotated[str, typer.Option(help="Amazon data category")] = "All_Beauty",
    nb_rows: Annotated[int, typer.Option(help="Amazon data category")] = 10_000,
    debug: Annotated[bool, typer.Option(help="Amazon data category")] = False,
) -> None:
    
    from src.nlp.sentiment_analysis.core import sentiment_regressor_pipeline

    sentiment_regressor_pipeline(
        category=category,
        nb_rows=nb_rows,
        debug=debug,
    )

