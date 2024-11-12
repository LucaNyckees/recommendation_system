import psycopg
from psycopg import Cursor
from psycopg.sql import SQL, Composed, Identifier, Placeholder
from typing import TypeVar, TypeAlias, LiteralString, Union, Mapping, Any, Sequence
import pandas as pd


Query: TypeAlias = Union[LiteralString, bytes, SQL, Composed]
Params: TypeAlias = Union[Sequence[Any], Mapping[str, Any]]


def load_dataframe_from_query(cur: Cursor, query: Query, params: Params) -> pd.DataFrame:
    cur.execute(query=query, params=params)
    fetched = cur.fetchall()
    return pd.Dataframe(fetched)


def load_dicts_from_query(cur: Cursor, query: Query, params: Params) -> list[dict[str, Any]]:
    cur.execute(query=query, params=params)
    fetched = cur.fetchall()
    return fetched


def insert_values(cur: Cursor, table: str, values: list[dict], cols_mapping: dict[str, str]) -> None:
    """
    :param cols_mapping: dict of the form db_col: inserted_col
    """
    insert_query = SQL("INSERT INTO {table} ({db_cols}) VALUES ({inserted_cols})").format(
        table=Identifier(table),
        db_cols=SQL(", ").join(map(Identifier, cols_mapping.keys())),
        inserted_cols=SQL(", ").join(map(Placeholder, cols_mapping.values())),
        )
    cur.executemany(query=insert_query, params_seq=values)

