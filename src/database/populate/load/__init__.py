import typer


app = typer.Typer(name="load")


@app.command(name="datasets")
def load_datasets() -> None:
    from .products import load_products
    from .reviews import load_reviews

    load_products()
    load_reviews()

    return None


@app.command(name="generated-users")
def load_datasets() -> None:
    from .users import load_users

    load_users()

    return None
