#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from pymongo import MongoClient
from pymongo.collection import Collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import JSON files into MongoDB")
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
    return parser.parse_args()


def sanitize_collection_name(name: str) -> str:
    """Ensure collection name is valid for MongoDB."""
    return name.replace(".", "_").replace(" ", "_")


def import_file(path: Path, collection: Collection, force: bool) -> None:
    """Import a single JSON file into MongoDB."""
    if path.stat().st_size == 0:
        print(f"Skipping empty file: {path}")
        return

    try:
        with path.open() as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        print(f"Skipping invalid JSON file: {path} ({e})")
        return

    # Add metadata
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

    # Insert data
    if isinstance(data, list):
        collection.insert_many(data)
    else:
        collection.insert_one(data)

    print(f"Imported {path} into collection '{collection.name}'")


def main() -> None:
    args = parse_args()
    force, to_delete = args.force, args.delete

    # Setup MongoDB connection
    port = int(os.getenv("SERVER_MONGODB_PORT", 27017))
    client = MongoClient(f"mongodb://localhost:{port}")
    db = client.optimization

    if to_delete:
        client.drop_database(db.name)
        print(f"Database '{db.name}' deleted.")
        return

    base_dir = Path(".cache/algorithm")
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
