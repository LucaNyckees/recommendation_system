import asyncio
from httpx import AsyncClient
import aiofiles
from pathlib import Path
import shutil
import gzip

from src.paths import DATA_PATH
from src.log.logger import logger


def download_amazon_datasets() -> None:
    """
    Download amazone data for products and reviews.
    """
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    asyncio.run(download_datasets_coro())


async def download_datasets_coro() -> None:
    """
    We use asynchronous downloading in case later on there are multiple data sources.
    """
    async with AsyncClient(timeout=None) as client:
        await asyncio.gather(
            download_reviews(client),
            download_products(client),
        )


async def download_dataset(client: AsyncClient, url: str, path: Path) -> None:
    r = await client.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Could not download {url}")
    async with aiofiles.open(path, "wb") as f:
        await f.write(r.content)


async def download_reviews(client: AsyncClient) -> None:
    url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"
    dir = DATA_PATH / "amazon"
    dir.mkdir(exist_ok=True, parents=True)
    zip_path = dir / "All_Beauty.jsonl.gz"
    final_dir = dir
    await download_dataset(client, url, zip_path)
    logger.info("Amazon reviews downloaded.")
    with gzip.open(zip_path, 'rb') as f_in:
        with open(final_dir / "All_Beauty.jsonl", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    zip_path.unlink()


async def download_products(client: AsyncClient) -> None:
    url = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_All_Beauty.jsonl.gz"
    dir = DATA_PATH / "amazon"
    dir.mkdir(exist_ok=True, parents=True)
    zip_path = dir / "meta_All_Beauty.jsonl.gz"
    final_dir = dir
    await download_dataset(client, url, zip_path)
    logger.info("Amazon products downloaded.")
    with gzip.open(zip_path, 'rb') as f_in:
        with open(final_dir / "meta_All_Beauty.jsonl", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    zip_path.unlink()
    pass