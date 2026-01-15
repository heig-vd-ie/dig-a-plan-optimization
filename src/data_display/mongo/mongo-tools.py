#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path
from pymongo import MongoClient
from pymongo.collection import Collection
from konfig import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage MongoDB data and import JSON files."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-import files even if already present",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the database before importing",
    )
    parser.add_argument(
        "--db",
        default="optimization",
        help="Database name to use (default: optimization)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup the database to a timestamped folder",
    )
    parser.add_argument(
        "--restore",
        type=str,
        metavar="BACKUP_PATH",
        help="Restore the database from a backup folder",
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Chunk large files before importing",
    )
    return parser.parse_args()


def sanitize_collection_name(name: str) -> str:
    return name.replace(".", "_").replace(" ", "_")


def import_file(path: Path, collection: Collection, force: bool) -> None:
    if path.stat().st_size == 0:
        print(f"Skipping empty file: {path}")
        return

    try:
        with path.open() as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        print(f"Skipping invalid JSON file: {path} ({e})")
        return
    except MemoryError as e:
        print(f"Skipping large file that can't fit in memory: {path} ({e})")
        return

    if isinstance(data, list):
        for doc in data:
            doc["_source_file"] = str(path)
    else:
        data["_source_file"] = str(path)

    query = {"_source_file": str(path)}
    exists = collection.find_one(query)

    if exists and not force:
        print(f"Skipping already imported file: {path}")
        return
    elif exists and force:
        collection.delete_many(query)
        print(f"Re-importing (force) file: {path}")

    try:
        if isinstance(data, list):
            collection.insert_many(data)
        else:
            collection.insert_one(data)
    except Exception as e:
        print(f"Error inserting {path}: {e}")
        return

    print(f"Imported {path} into collection '{collection.name}'")


def backup_db(db_name: str) -> None:
    out_dir = Path.home() / "mongodb_backups" / f"{db_name}_{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["mongodump", "--db", db_name, "--out", str(out_dir)], check=True)
    print(f"Database '{db_name}' backed up to {out_dir}")


def restore_db(db_name: str, backup_path: str) -> None:
    backup_path_obj = Path(backup_path)
    if not backup_path_obj.exists():
        print(f"Backup path {backup_path_obj} does not exist")
        return
    subprocess.run(
        ["mongorestore", "--db", db_name, str(backup_path_obj / db_name)], check=True
    )
    print(f"Database '{db_name}' restored from {backup_path}")


def main() -> None:
    args = parse_args()
    force, to_delete = args.force, args.delete
    db_name = args.db

    client = MongoClient(
        f"mongodb://{os.getenv('LOCAL_HOST')}:{os.getenv('SERVER_MONGODB_PORT')}"
    )
    db = client[db_name]

    if args.backup:
        backup_db(db_name)
        return

    if args.restore:
        restore_db(db_name, args.restore)
        return

    if args.chunk:
        print("Chunking large files...")
        subprocess.run([".venv/bin/python", "scripts/chunk-files.py"], check=True)
        print("Chunking completed.")
        return

    if to_delete:
        client.drop_database(db_name)
        print(f"Database '{db_name}' deleted.")
        return

    base_dir = Path(settings.cache.outputs_expansion)
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    for root, _, files in os.walk(base_dir):
        rel = Path(root).relative_to(base_dir)
        run_name = (
            "base_run" if rel == Path(".") else sanitize_collection_name(rel.parts[0])
        )
        collection = db[run_name]

        for file in files:
            if file.endswith(".json"):
                import_file(Path(root) / file, collection, force)

    print("All valid files processed.")


if __name__ == "__main__":
    main()
