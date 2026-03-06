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
from helpers import generate_log
from experiments import PROJECT_ROOT

# Database connection - Adjust based on your Docker env vars
DB_CONFIG = {
    "host": os.getenv("LOCAL_HOST", "localhost"),
    "port": os.getenv("SERVER_PG_PORT", "5432"),
    "user": os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", "postgres"),
}

DB_NAME = os.getenv("PG_DB", "experiments")
LOG = generate_log(__name__)


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
    LOG.info(f"Database backed up to {out_file}")


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
        LOG.warning(f"Skipping {path}: {e}")
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
            LOG.warning(
                f"File {path.name} too large ({len(raw_json_str)} bytes). Chunking..."
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
                    LOG.error(f"Error inserting {path} after chunking: {e2}")
        else:
            LOG.error(f"Error inserting {path}: {e}")
    finally:
        cur.close()


if __name__ == "__main__":
    args = parse_args()

    if args.experiment == "geolocations":
        from experiments.common_files_recording import record_all

        record_all()
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS geolocations;")
            ensure_table(cur, "geolocations")
            for file in (PROJECT_ROOT / settings.cache.figures / "geolocations").glob(
                "*.json"
            ):
                import_file(file, "geolocations", conn, args.force)
        conn.commit()
        exit(0)

    # Handle Delete / Create DB logic
    if args.reset and args.experiment == "all":
        conn = get_connection()
        conn.autocommit = True
        cur = conn.cursor()
        try:
            LOG.info(f"Wiping all tables in database '{DB_NAME}'...")
            # Dropping the schema 'public' deletes all tables instantly
            cur.execute("DROP SCHEMA public CASCADE;")
            cur.execute("CREATE SCHEMA public;")
            # Grant permissions back if necessary (usually default for owner)
            cur.execute("GRANT ALL ON SCHEMA public TO public;")
            LOG.info(f"All tables removed from '{DB_NAME}'")
        except Exception as e:
            LOG.error(f"Error clearing tables: {e}")
        finally:
            cur.close()
            conn.close()

    if args.experiment != "all" and args.reset:
        conn = get_connection()
        conn.autocommit = True
        cur = conn.cursor()
        table_name = sanitize_table_name(args.experiment)
        try:
            LOG.info(f"Dropping table '{table_name}'...")
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name))
            )
            LOG.info(f"Table '{table_name}' dropped from '{DB_NAME}'.")
        except Exception as e:
            LOG.error(f"Error dropping table '{table_name}': {e}")
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
            if any(
                name in str(rel)
                for name in [
                    "base_run",
                    "run_test",
                    "run_test_api",
                ]
            ) or rel == Path("."):
                continue  # Skip the base_run folder itself
            run_name = sanitize_table_name(rel.parts[0])

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
        LOG.info(f"Sync complete for '{DB_NAME}'")
