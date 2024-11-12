import typer


app = typer.Typer(name="load")


@app.command(name="datasets")
def load_datasets() -> None:
    from .products import load_products
    from .reviews import load_reviews

    load_products()
    load_reviews()

    return None
