import typer
from typing_extensions import Annotated

from src.nlp.regressor import regressor_pipeline

app = typer.Typer(name="nlp")


@app.command(name="reviews-regressor")
def run_bert_regressor_pipeline(
    category: Annotated[str, typer.Option(help="Amazon data category")] = "All_beauty",
    frac: Annotated[float, typer.Option(help="Amazon data category")] = 0.001,
    debug: Annotated[bool, typer.Option(help="Amazon data category")] = False,
) -> None:
    regressor_pipeline(
        category=category,
        frac=frac,
        debug=debug,
    )


@app.command(name="reviews-classifier")
def run_bert_classifier_pipeline() -> None:
    raise NotImplementedError("Not yet implemented, work still in progress.")
