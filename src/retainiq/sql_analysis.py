"""Step 2: load cleaned data into SQLite and run the saved .sql queries."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from . import config


def load_to_sqlite(df: pd.DataFrame, db_path: Path | str = config.DB_PATH, table: str = "customers") -> None:
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


def parse_queries(sql_path: Path | str = config.SQL_FILE) -> dict[str, str]:
    """Split the .sql file into {query_name: sql} using '-- name:' markers."""
    text = Path(sql_path).read_text(encoding="utf-8")
    parts = re.split(r"^--\s*name:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    # parts = [header, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_queries(db_path: Path | str = config.DB_PATH, sql_path: Path | str = config.SQL_FILE) -> dict[str, pd.DataFrame]:
    queries = parse_queries(sql_path)
    with sqlite3.connect(db_path) as conn:
        return {name: pd.read_sql_query(sql, conn) for name, sql in queries.items()}
