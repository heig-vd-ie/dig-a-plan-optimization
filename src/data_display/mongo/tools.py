#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from pymongo import MongoClient
from pymongo.collection import Collection
from data_display.mongo.chunk import chunk_files
from konfig import settings


HOME_FOLDER = Path(os.__file__).parent.parent.parent.parent
MONGO_CACHE_DIR = ".cache/mongo"


@dataclass
class Args:
    force: bool
    delete: bool
    db: str
    backup: bool
    restore: str
    chunk: bool
    sync: bool


def parse_args() -> Args:
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
        default="expansion",
        help="Database name to use (default: expansion)",
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
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync the database with the current state of the cache (delete missing files)",
    )
    ns = parser.parse_args()
    return Args(
        force=ns.force,
        delete=ns.delete,
        db=ns.db,
        backup=ns.backup,
        restore=ns.restore,
        chunk=ns.chunk,
        sync=ns.sync,
    )


def backup_db(db_name: str) -> None:
    out_dir = HOME_FOLDER / MONGO_CACHE_DIR / f"{db_name}_{os.getpid()}"
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
        print(f"\033[31mSkipping invalid JSON file: {path} ({e})\033[0m")
        return
    except MemoryError as e:
        print(
            f"\033[31mSkipping large file that can't fit in memory: {path} ({e})\033[0m"
        )
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
        print(f"\033[31mError inserting {path}: {e}\033[0m")
        return

    print(f"\033[32mImported {path} into collection '{collection.name}'\033[0m")


if __name__ == "__main__":
    args = parse_args()
    client = MongoClient(
        f"mongodb://{os.getenv('LOCAL_HOST')}:{os.getenv('SERVER_MONGODB_PORT')}"
    )

    if args.backup:
        backup_db(args.db)

    if args.restore:
        restore_db(args.db, args.restore)

    if args.chunk:
        print("Chunking large files...")
        chunk_files()
        print("Chunking completed.")

    if args.delete:
        client.drop_database(args.db)
        print(f"Database '{args.db}' deleted.")

    if args.sync:
        base_dir = Path(settings.cache.outputs_expansion)
        if not base_dir.exists():
            print(f"Directory {base_dir} does not exist.")
        for root, _, files in os.walk(base_dir):
            rel = Path(root).relative_to(base_dir)
            run_name = (
                "base_run"
                if rel == Path(".")
                else sanitize_collection_name(rel.parts[0])
            )
            collection = client[args.db][run_name]
            for file in files:
                if file.endswith(".json"):
                    import_file(Path(root) / file, collection, args.force)
        print("All valid files processed.")
