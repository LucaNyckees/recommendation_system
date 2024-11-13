import typer


app = typer.Typer(name="download")


@app.command(name="datasets")
def download_datasets() -> None:
    from .amazon_datasets import download_amazon_datasets

    download_amazon_datasets()

    return None
