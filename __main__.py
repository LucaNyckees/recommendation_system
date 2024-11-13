import typer

from src import nlp
from src.database.populate import load, download


app = typer.Typer()

app.add_typer(nlp.app)
app.add_typer(load.app)
app.add_typer(download.app)

if __name__ == "__main__":
    app()
