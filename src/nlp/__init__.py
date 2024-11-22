import typer
from typing_extensions import Annotated

app = typer.Typer(name="nlp")


@app.command(name="reviews-regressor")
def run_bert_regressor_pipeline(
    category: Annotated[str, typer.Option(help="Amazon data category")] = "All_beauty",
    frac: Annotated[float, typer.Option(help="Amazon data category")] = 0.001,
    debug: Annotated[bool, typer.Option(help="Amazon data category")] = False,
) -> None:
    
    from src.nlp.regressor import regressor_pipeline

    regressor_pipeline(
        category=category,
        frac=frac,
        debug=debug,
    )
