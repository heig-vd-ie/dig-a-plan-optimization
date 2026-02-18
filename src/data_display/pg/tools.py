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
    experiment: str = "all"


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
    parser.add_argument("--experiment", type=str, help="Experiment name to process")
    ns = parser.parse_args()
    return Args(ns.force, ns.reset, ns.backup, ns.restore, ns.sync, ns.experiment)


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


def chunk_sddp_response(data: list[dict], divided_by: int = 10) -> list:
    """
    Docstring for chunk_sddp_response

    :param data: Description
    :type data: dict | list
    :param chunk_size: Description
    """
    if len(data) == 0:
        return data
    final_results = []
    for i in range(divided_by):
        chunk = {}
        for key, value in data[0].items():
            chunk_size = len(value) // divided_by
            chunk[key] = value[i * chunk_size : (i + 1) * chunk_size]
        final_results.append(chunk)
    return final_results


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
                f"{ORANGE}File {path.name} too large ({len(raw_json_str)} bytes). Chunking...{RESET}"
            )
            data_to_upload = chunk_sddp_response(data_to_upload, divided_by=10)
            for idx, entry in enumerate(
                tqdm(data_to_upload, desc=f"Chunking {path.name}")
            ):
                try:
                    unique_path = f"{path}_part_{idx}"
                    cur.execute(
                        sql.SQL(
                            "INSERT INTO {} (source_file, data) VALUES (%s, %s)"
                        ).format(sql.Identifier(table_name)),
                        (unique_path, json.dumps(entry)),
                    )
                    conn.commit()
                except Exception as e2:
                    conn.rollback()
                    print(f"\033[31mError inserting {path} after chunking: {e2}\033[0m")
        else:
            print(f"\033[31mError inserting {path}: {e}\033[0m")
    finally:
        cur.close()


if __name__ == "__main__":
    args = parse_args()

    # Handle Delete / Create DB logic
    if args.reset and args.experiment == "all":
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

            # Filter based on selected experiment if not "all"
            if args.experiment != "all" and run_name != sanitize_table_name(
                args.experiment
            ):
                continue

            with conn.cursor() as cur:
                ensure_table(cur, run_name)
            conn.commit()

            json_files = [f for f in files if f.endswith(".json")]
            for file in tqdm(json_files, desc=f"Table: {run_name}"):
                import_file(Path(root) / file, run_name, conn, args.force)

        conn.close()
        print(f"\033[32mSync complete for '{DB_NAME}'\033[0m")
