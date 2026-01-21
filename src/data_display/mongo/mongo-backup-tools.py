#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from pymongo import MongoClient
import msgpack
from bson import ObjectId
from datetime import datetime
from typing import Union
import random
import string


# Helper functions to serialize/deserialize BSON types
def serialize_doc(doc):
    doc = dict(doc)  # make a copy
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            doc[k] = {"__objectid__": str(v)}
        elif isinstance(v, datetime):
            doc[k] = {"__datetime__": v.isoformat()}
        elif isinstance(v, dict):
            doc[k] = serialize_doc(v)
        elif isinstance(v, list):
            doc[k] = [serialize_doc(x) if isinstance(x, dict) else x for x in v]
    return doc


def deserialize_doc(doc):
    for k, v in doc.items():
        if isinstance(v, dict):
            if "__objectid__" in v:
                doc[k] = ObjectId(v["__objectid__"])
            elif "__datetime__" in v:
                doc[k] = datetime.fromisoformat(v["__datetime__"])
            else:
                doc[k] = deserialize_doc(v)
        elif isinstance(v, list):
            doc[k] = [deserialize_doc(x) if isinstance(x, dict) else x for x in v]
    return doc


def backup_db(db_name: str, backup_dir: str) -> None:
    client = MongoClient(
        f"mongodb://{os.getenv('LOCAL_HOST')}:{os.getenv('MONGODB_PORT')}"
    )
    db = client[db_name]

    os.makedirs(backup_dir, exist_ok=True)
    # Randomized timestamp-based backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    backup_file = Path(backup_dir) / f"{db_name}_backup_{timestamp}_{rand_str}.msgpack"

    data = {}
    for col_name in db.list_collection_names():
        data[col_name] = [serialize_doc(doc) for doc in db[col_name].find({})]

    with open(backup_file, "wb") as f:
        msgpack.pack(data, f)

    print(f"Database '{db_name}' backed up to '{backup_file}'")


def get_latest_backup(backup_dir: Union[str, Path], db_name: str) -> Path:
    backup_dir = Path(backup_dir)
    backups = sorted(
        backup_dir.glob(f"{db_name}_backup_*.msgpack"), key=os.path.getmtime
    )
    if not backups:
        raise FileNotFoundError(
            f"No backup found for database '{db_name}' in {backup_dir}"
        )
    return backups[-1]  # latest file


def restore_db(db_name: str, backup_file: Union[str, Path], backup_dir: str) -> None:
    if backup_file is None:
        # pick latest backup
        backup_file = get_latest_backup(backup_dir, db_name)

    client = MongoClient(
        f"mongodb://{os.getenv('LOCAL_HOST')}:{os.getenv('MONGODB_PORT')}"
    )
    db = client[db_name]

    backup_file = Path(backup_file)
    if not backup_file.exists():
        print(f"Backup file {backup_file} does not exist")
        return

    with open(backup_file, "rb") as f:
        data = msgpack.unpack(f, raw=False)

    if not isinstance(data, dict):
        print(
            f"Backup file {backup_file} does not contain a valid dictionary. Aborting restore."
        )
        return

    for col_name, docs in data.items():
        collection = db[col_name]
        collection.delete_many({})  # clear existing data
        if docs:
            collection.insert_many([deserialize_doc(doc) for doc in docs])

    print(f"Database '{db_name}' restored from '{backup_file}'")


def main():
    parser = argparse.ArgumentParser(
        description="Backup or restore a MongoDB database using MessagePack"
    )
    parser.add_argument("--db", required=True, help="Database name")
    parser.add_argument("--backup", action="store_true", help="Backup the database")
    parser.add_argument(
        "--restore",
        nargs="?",
        const=None,
        help="Restore from a backup file (optional, uses latest if omitted)",
    )
    parser.add_argument(
        "--out", default="./mongo_backup", help="Directory to store backup"
    )
    args = parser.parse_args()

    if args.backup:
        backup_db(args.db, args.out)
    elif args.restore is not None:
        restore_db(args.db, args.restore, args.out)
    else:
        print("Specify --backup or --restore [<file>]")


if __name__ == "__main__":
    main()
