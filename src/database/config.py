from urllib.parse import quote
from toml import load
from sqlalchemy import URL

from src.paths import ROOT


config = load(ROOT / "config.toml")


def format_db_url_postgresql(user: str, password: str, host: str, port: int, name: str) -> str:
    password = quote(password, safe="$^)]")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def get_db_url_from_key(db_key: str) -> str:

    db_config = config["dbs"][db_key]

    return format_db_url_postgresql(**db_config)


def format_db_url_sqlalchemy(**kwargs) -> URL:
    d = kwargs.copy()

    keys = ("user", "password", "host", "port", "name")
    cred = {}
    for k in keys:
        assert k in d, f"format_db_url_sqlalchemy must be provided with proper credentials : {keys}"
        cred[k] = d[k]
    return URL.create(
        "postgresql",
        username=cred["user"],
        password=cred["password"],
        host=cred["host"],
        database=cred["name"],
        port=cred["port"],
    )