from typing import Any, Literal, overload

import psycopg
from psycopg.rows import dict_row

from .config import get_db_url_from_key

try:
    import numpy as np

    np_not_found = False
except ModuleNotFoundError:
    np_not_found = True

try:
    import pandas as pd
    from pandas._libs.missing import NAType

    pd_not_found = False
except ModuleNotFoundError:
    pd_not_found = True

from psycopg.abc import AdaptContext
from psycopg.adapt import Dumper
from psycopg.pq import Format
from psycopg.types.bool import BoolBinaryDumper, BoolDumper
from psycopg.types.numeric import FloatBinaryDumper, FloatDumper, IntBinaryDumper, IntDumper


class NoneDumper(Dumper):
    def dump(self, obj: Any) -> None:
        return None


class NoneBinaryDumper(NoneDumper):
    format = Format.BINARY

    def dump(self, obj: Any) -> None:
        return None


class NPIntDumper(IntDumper):
    def dump(self, obj: Any) -> bytes:
        return super().dump(int(obj))


class NPIntBinaryDumper(IntBinaryDumper):
    def dump(self, obj: Any) -> bytes:
        return super().dump(int(obj))


class NPFloatDumper(FloatDumper):
    def dump(self, obj: Any) -> bytes:
        return super().dump(float(obj))


class NPFloatBinaryDumper(FloatBinaryDumper):
    def dump(self, obj: Any) -> bytes:
        return super().dump(float(obj))


def register_adapters(context: AdaptContext) -> None:
    adapters = context.adapters
    if not np_not_found:
        adapters.register_dumper(np.bool_, BoolDumper)
        adapters.register_dumper(np.bool_, BoolBinaryDumper)
        adapters.register_dumper(np.int64, NPIntDumper)
        adapters.register_dumper(np.int64, NPIntBinaryDumper)
        adapters.register_dumper(np.float32, FloatBinaryDumper)
        adapters.register_dumper(np.float64, FloatBinaryDumper)
        adapters.register_dumper(np.int64, NPIntDumper)
        adapters.register_dumper(np.int64, NPIntBinaryDumper)
    if not pd_not_found:
        adapters.register_dumper(pd.Int32Dtype, NPIntDumper)
        adapters.register_dumper(pd.Int32Dtype, NPIntBinaryDumper)
        adapters.register_dumper(NAType, NoneDumper)
        adapters.register_dumper(NAType, NoneBinaryDumper)
        adapters.register_dumper(type(pd.NA), NoneDumper)
        adapters.register_dumper(type(pd.NA), NoneBinaryDumper)
        adapters.register_dumper(type(pd.NaT), NoneDumper)
        adapters.register_dumper(type(pd.NaT), NoneBinaryDumper)


# Clean but verbose way to type connect (bc its return type depends on the async_connection argument)
@overload
def connect(
    *,
    db_url: str | None = None,
    db_key: str | None = None,
    row_factory: Literal["dict", "realdict", "tuple"] = "dict",
    autocommit: bool = False,
    async_connection: Literal[False],
    suffix: str = "",
) -> psycopg.Connection: ...


@overload
def connect(
    *,
    db_url: str | None = None,
    db_key: str | None = None,
    row_factory: Literal["dict", "realdict", "tuple"] = "dict",
    autocommit: bool = False,
    async_connection: Literal[True],
    suffix: str = "",
) -> psycopg.AsyncConnection: ...


def connect(
    *,
    db_url: str | None = None,
    db_key: str | None = None,
    row_factory: Literal["dict", "realdict", "tuple"] = "dict",
    autocommit: bool = False,
    async_connection: bool = False,
    suffix: str = "",
) -> psycopg.Connection | psycopg.AsyncConnection:
    """Return Psycopg PostgreSQL connection"""
    match db_url, db_key:
        case None, None:
            raise ValueError("Should get a db tag or a db url.")
        case _, None:
            pass
        case None, _:
            db_url = get_db_url_from_key(db_key, suffix=suffix)
        case _, _:
            raise ValueError("Cannot state both db_url & db_key")

    match row_factory:
        case "dict":
            row_factory = dict_row
        case "realdict":
            row_factory = dict_row
        case _:
            row_factory = None

    ConnectionClass = psycopg.AsyncConnection if async_connection else psycopg.Connection

    conn = ConnectionClass.connect(db_url, row_factory=row_factory, autocommit=autocommit)
    register_adapters(conn)
    return conn