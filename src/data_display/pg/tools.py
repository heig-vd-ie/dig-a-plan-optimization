#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
import psycopg2
from psycopg2 import sql
from tqdm import tqdm
from konfig import settings

# Database connection - Adjust based on your Docker env vars
DB_CONFIG = {
    "host": os.getenv("LOCAL_HOST", "localhost"),
    "port": os.getenv("SERVER_PG_PORT", "5432"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "postgres"),
}

DB_NAME = os.getenv("PG_DB", "experiments")
# ANSI Color for Orange
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


@dataclass
class Args:
    force: bool
    reset: bool
    backup: bool
    restore: str
    sync: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Manage PostgreSQL simulation data.")
    parser.add_argument(
        "--force", action="store_true", help="Re-import files even if already present"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset the database (drop and recreate)"
    )
    parser.add_argument(
        "--backup", action="store_true", help="Backup the database using pg_dump"
    )
    parser.add_argument(
        "--restore", type=str, help="Restore the database from a .sql file"
    )
    parser.add_argument("--sync", action="store_true", help="Sync folders to tables")
    ns = parser.parse_args()
    return Args(ns.force, ns.reset, ns.backup, ns.restore, ns.sync)


def get_connection():
    config = DB_CONFIG.copy()
    config["database"] = DB_NAME
    return psycopg2.connect(
        database=config["database"],
        user=config["user"],
        password=config["password"],
        host=config["host"],
        port=config["port"],
    )


def backup_db() -> None:
    out_file = f"{DB_NAME}_{os.getpid()}.sql"
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_CONFIG["password"]
    subprocess.run(
        [
            "pg_dump",
            "-h",
            DB_CONFIG["host"],
            "-p",
            DB_CONFIG["port"],
            "-U",
            DB_CONFIG["user"],
            "-f",
            out_file,
            DB_NAME,
        ],
        check=True,
        env=env,
    )
    print(f"\033[32mDatabase backed up to {out_file}\033[0m")


def sanitize_table_name(name: str) -> str:
    # Postgres tables are cleaner with underscores and lowercase
    return name.lower().replace(".", "_").replace(" ", "_").replace("-", "_")


def ensure_table(cur, table_name: str):
    cur.execute(
        sql.SQL(
            """
        CREATE TABLE IF NOT EXISTS {} (
            id SERIAL PRIMARY KEY,
            source_file TEXT UNIQUE,
            data JSONB,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
        ).format(sql.Identifier(table_name))
    )


def chunk_dict(data, chunk_size=5000):
    """
    If a dict contains a very long list, break it into a list of smaller dicts.
    """
    # Find the largest list in the dictionary to split it
    large_key = None
    for key, value in data.items():
        if isinstance(value, list) and len(value) > chunk_size:
            large_key = key
            break

    if not large_key:
        return [data]  # Can't chunk effectively, return as is

    original_list = data[large_key]
    chunks = []
    for i in range(0, len(original_list), chunk_size):
        new_chunk = data.copy()
        new_chunk[large_key] = original_list[i : i + chunk_size]
        new_chunk["_chunk_index"] = i // chunk_size
        chunks.append(new_chunk)

    return chunks


def import_file(path: Path, table_name: str, conn, force: bool) -> None:
    if path.stat().st_size == 0:
        return

    try:
        with path.open() as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"\033[93mSkipping {path}: {e}\033[0m")
        return

    # Check if the raw string size is approaching the 255MB limit
    raw_json_str = json.dumps(data)
    data_to_upload = [data]

    if len(raw_json_str) > 250_000_000:  # 250MB Safety threshold
        print(
            f"{ORANGE}File {path.name} too large ({len(raw_json_str)} bytes). Chunking...{RESET}"
        )
        data_to_upload = chunk_dict(data)

    cur = conn.cursor()
    try:
        for entry in data_to_upload:
            cur.execute(
                sql.SQL("INSERT INTO {} (source_file, data) VALUES (%s, %s)").format(
                    sql.Identifier(table_name)
                ),
                (str(path), json.dumps(entry)),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        # Specific check for the Postgres Size Error
        if "total size of jsonb object elements exceeds the maximum" in str(e):
            print(
                f"{ORANGE}Error: File {path.name} still too large after chunking. Check nested structures.{RESET}"
            )
        else:
            print(f"\033[31mError inserting {path}: {e}\033[0m")
    finally:
        cur.close()


if __name__ == "__main__":
    args = parse_args()

    # Handle Delete / Create DB logic
    if args.reset:
        conn = get_connection()
        conn.autocommit = True
        cur = conn.cursor()
        try:
            print(f"\033[33mWiping all tables in database '{DB_NAME}'...\033[0m")
            # Dropping the schema 'public' deletes all tables instantly
            cur.execute("DROP SCHEMA public CASCADE;")
            cur.execute("CREATE SCHEMA public;")
            # Grant permissions back if necessary (usually default for owner)
            cur.execute("GRANT ALL ON SCHEMA public TO public;")
            print(f"\033[32mAll tables removed from '{DB_NAME}'.\033[0m")
        except Exception as e:
            print(f"\033[31mError clearing tables: {e}\033[0m")
        finally:
            cur.close()
            conn.close()

    if args.backup:
        backup_db()

    if args.sync:
        base_dir = Path(settings.cache.outputs_expansion)
        conn = get_connection()

        for root, _, files in os.walk(base_dir):
            rel = Path(root).relative_to(base_dir)
            # Use top-level folder as table name
            run_name = (
                "base_run" if rel == Path(".") else sanitize_table_name(rel.parts[0])
            )

            with conn.cursor() as cur:
                ensure_table(cur, run_name)
            conn.commit()

            json_files = [f for f in files if f.endswith(".json")]
            for file in tqdm(json_files, desc=f"Table: {run_name}"):
                import_file(Path(root) / file, run_name, conn, args.force)

        conn.close()
        print(f"\033[32mSync complete for '{DB_NAME}'\033[0m")
