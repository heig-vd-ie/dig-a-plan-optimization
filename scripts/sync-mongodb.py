import json, os, argparse
from pymongo import MongoClient

parser = argparse.ArgumentParser()
parser.add_argument(
    "--force", action="store_true", help="Re-import files even if already present"
)
args = parser.parse_args()
force = args.force

client = MongoClient("mongodb://localhost:27017")
db = client.optimization

BASE_DIR = ".cache/algorithm"

for root, dirs, files in os.walk(BASE_DIR):
    rel = os.path.relpath(root, BASE_DIR)
    if rel == ".":
        # files directly in BASE_DIR: use a default collection name
        run_name = "base_run"
    else:
        run_name = rel.split(os.sep)[0]  # top-level folder

    # Replace invalid characters for MongoDB collection name
    run_name = run_name.replace(".", "_").replace(" ", "_")

    collection = db[run_name]

    for f in files:
        if not f.endswith(".json"):
            continue
        path = os.path.join(root, f)
        if os.path.getsize(path) == 0:
            print(f"Skipping empty file: {path}")
            continue

        try:
            with open(path) as fh:
                data = json.load(fh)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON file: {path} ({e})")
            continue

        # Add metadata
        if isinstance(data, list):
            for doc in data:
                doc["_source_file"] = path
        else:
            data["_source_file"] = path

        # Skip or force
        query = {"_source_file": path}
        exists = collection.find_one(query)
        if exists and not force:
            print(f"Skipping already imported file: {path}")
            continue
        elif exists and force:
            collection.delete_many(query)
            print(f"Re-importing (force) file: {path}")

        # Insert
        if isinstance(data, list):
            collection.insert_many(data)
        else:
            collection.insert_one(data)

        print(f"Imported {path} into collection '{run_name}'")

print("All valid files processed.")
