import typer

from src import nlp


app = typer.Typer()

app.add_typer(nlp.app)

if __name__ == "__main__":
    app()
