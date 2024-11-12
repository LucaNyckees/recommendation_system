import typer

from src import nlp
from src.database.populate import load


app = typer.Typer()

app.add_typer(nlp.app)
app.add_typer(load.app)

if __name__ == "__main__":
    app()
