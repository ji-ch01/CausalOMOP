from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "synthetic_snuh_curated.duckdb"
DEFAULT_VOCAB_DIR = PROJECT_ROOT / "data" / "vocabulary"


def db_path() -> Path:
    return Path(os.getenv("OMOP_DB_PATH", DEFAULT_DB_PATH)).resolve()


def vocab_dir() -> Path:
    return Path(os.getenv("OMOP_VOCAB_DIR", DEFAULT_VOCAB_DIR)).resolve()


def connect_write() -> duckdb.DuckDBPyConnection:
    path = db_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {path}. Set OMOP_DB_PATH or create data/synthetic_snuh_curated.duckdb."
        )
    return duckdb.connect(str(path))


def backup_database(script_name: str) -> Path:
    source = db_path()
    backup_dir = source.parent / "backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"{source.stem}.{script_name}.{stamp}{source.suffix}"
    shutil.copy2(source, backup)
    return backup


def sql_literal_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")
