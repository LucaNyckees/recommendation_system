from urllib.parse import quote
from toml import load

from src.paths import ROOT

def format_db_url_postgresql(user: str, password: str, host: str, port: int, name: str) -> str:
    password = quote(password, safe="$^)]")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


config = load(ROOT / "config.toml")


def get_db_url_from_tag(db_tag: str) -> str:

    db_config = config["dbs"][db_tag]

    return format_db_url_postgresql(**db_config)