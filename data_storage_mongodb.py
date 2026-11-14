#!/usr/bin/env python3
"""
AUCA E-commerce Project - Resumable MongoDB Bulk Loader

Resume behavior:
- Stores checkpoints in MongoDB collection: _import_checkpoints
- If a previous run stopped before checkpoints existed, it finds the first
  missing source record by checking existing IDs in MongoDB, then continues
  from that record (works when the old import inserted records in file order).
- After each successful batch, saves the last completed source-record index.
- If interrupted during a batch, that batch is safely replayed using upserts.

Run:
    python data_storage_mongodb.py

Never use --replace unless you intentionally want to delete all target data.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from decimal import Decimal

try:
    from pymongo import MongoClient, ASCENDING, DESCENDING, ReplaceOne
    from bson.decimal128 import Decimal128
except ImportError:
    print("Missing MongoDB packages. Install with: python -m pip install pymongo")
    sys.exit(1)

try:
    import ijson
except ImportError:
    ijson = None


# -------------------- CONFIGURATION --------------------

BATCH_SIZE = 5000
DATABASE_NAME = "ecommerce_analytics"
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_COLLECTION = "_import_checkpoints"

DATA_FILES = {
    "users": ("users.json", "user_id"),
    "categories": ("categories.json", "category_id"),
    "products": ("products.json", "product_id"),
    "transactions": ("transactions.json", "transaction_id"),
}
SESSION_KEY = "session_id"
COLLECTION_NAMES = ["users", "categories", "products", "transactions", "sessions"]


# -------------------- DOCUMENT CONVERSION --------------------

def convert_decimals(value):
    if isinstance(value, Decimal):
        return Decimal128(value)
    if isinstance(value, dict):
        return {key: convert_decimals(item) for key, item in value.items()}
    if isinstance(value, list):
        return [convert_decimals(item) for item in value]
    if isinstance(value, tuple):
        return [convert_decimals(item) for item in value]
    return value


def first_non_whitespace(path, block_size=65536):
    with open(path, "r", encoding="utf-8-sig") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                return ""
            for char in chunk:
                if not char.isspace():
                    return char


def iter_json_documents(path):
    """Stream JSON arrays; also support JSON Lines and regular JSON objects/lists."""
    first = first_non_whitespace(path)

    if first == "[":
        if ijson is None:
            raise RuntimeError(
                f"{path.name} is a JSON array. Install the streaming parser with: "
                "python -m pip install ijson"
            )
        with open(path, "rb") as f:
            for document in ijson.items(f, "item"):
                if isinstance(document, dict):
                    yield document
        return

    if first == "{":
        with open(path, "r", encoding="utf-8-sig") as f:
            first_line = f.readline()

        try:
            first_obj = json.loads(first_line)
            is_jsonl = isinstance(first_obj, dict)
        except json.JSONDecodeError:
            is_jsonl = False

        if is_jsonl:
            with open(path, "r", encoding="utf-8-sig") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        document = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {path.name}, line {line_number}: {exc}"
                        ) from exc
                    if isinstance(document, dict):
                        yield document
            return

        with open(path, "r", encoding="utf-8-sig") as f:
            parsed = json.load(f)
        if isinstance(parsed, dict):
            yield parsed
        elif isinstance(parsed, list):
            for document in parsed:
                if isinstance(document, dict):
                    yield document
        return

    raise ValueError(f"Unsupported or empty JSON file: {path}")


def session_sort_key(path):
    match = re.search(r"sessions_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else 999999


def file_signature(path):
    stat = path.stat()
    return {"file_size": stat.st_size, "file_mtime_ns": stat.st_mtime_ns}


# -------------------- CHECKPOINT / RECOVERY --------------------

def checkpoint_id(collection_name, path):
    return f"{collection_name}:{path.name}"


def save_checkpoint(checkpoints, cp_id, signature, last_index, complete=False):
    checkpoints.update_one(
        {"_id": cp_id},
        {
            "$set": {
                **signature,
                "last_index": last_index,
                "complete": complete,
            }
        },
        upsert=True,
    )


def find_first_missing_index(collection, path, key_field):
    """
    Legacy recovery: no checkpoint exists yet.
    Find the first source record whose key is not already in MongoDB.
    This assumes the interrupted old import inserted records in source-file order.
    """
    source_index = 0
    batch = []

    def inspect_batch(items, start_index):
        keys = []
        for doc in items:
            key = doc.get(key_field)
            if key is None:
                raise ValueError(
                    f"{path.name}: missing required field '{key_field}' "
                    f"near source record {start_index}"
                )
            keys.append(key)

        found = set()
        cursor = collection.find(
            {key_field: {"$in": keys}},
            {key_field: 1, "_id": 0},
        )
        for row in cursor:
            found.add(row[key_field])

        for offset, key in enumerate(keys):
            if key not in found:
                return start_index + offset
        return None

    for document in iter_json_documents(path):
        batch.append(document)
        if len(batch) >= BATCH_SIZE:
            missing = inspect_batch(batch, source_index)
            if missing is not None:
                return missing
            source_index += len(batch)
            batch = []

    if batch:
        missing = inspect_batch(batch, source_index)
        if missing is not None:
            return missing
        source_index += len(batch)

    return source_index  # every source record already exists


# -------------------- IMPORT --------------------

def import_file(collection, path, key_field, checkpoints):
    cp_id = checkpoint_id(collection.name, path)
    signature = file_signature(path)
    saved = checkpoints.find_one({"_id": cp_id})

    if saved:
        if (
            saved.get("file_size") != signature["file_size"]
            or saved.get("file_mtime_ns") != signature["file_mtime_ns"]
        ):
            raise RuntimeError(
                f"{path.name} has changed since its checkpoint was saved. "
                "Do not reuse the old checkpoint with a changed source file. "
                f"Checkpoint ID: {cp_id}"
            )
        if saved.get("complete"):
            print(f"  {path.name}: already completed; skipping.")
            return {"processed": 0, "upserted": 0, "modified": 0, "skipped": 0}

        start_index = saved["last_index"] + 1
        print(f"  Resuming {path.name} at source record {start_index + 1:,}")
    else:
        print(f"  No checkpoint for {path.name}; checking existing MongoDB IDs...")
        start_index = find_first_missing_index(collection, path, key_field)
        if start_index == sum(1 for _ in iter_json_documents(path)):
            # All source IDs already exist. Mark the file complete.
            save_checkpoint(
                checkpoints, cp_id, signature, start_index - 1, complete=True
            )
            print(f"  {path.name}: all source records already exist; nothing to import.")
            return {"processed": 0, "upserted": 0, "modified": 0, "skipped": start_index}

        print(
            f"  Recovery found first missing record at {start_index + 1:,}; "
            "continuing from there."
        )

    processed = 0
    upserted = 0
    modified = 0
    skipped = 0
    batch = []
    batch_last_index = None

    def flush_batch():
        nonlocal processed, upserted, modified, batch, batch_last_index
        if not batch:
            return

        operations = []
        for document in batch:
            key_value = document.get(key_field)
            if key_value is None:
                raise ValueError(
                    f"{path.name}: missing required field '{key_field}'"
                )
            operations.append(
                ReplaceOne({key_field: key_value}, document, upsert=True)
            )

        result = collection.bulk_write(operations, ordered=False)

        # Save the checkpoint only after MongoDB confirms the batch write.
        save_checkpoint(
            checkpoints, cp_id, signature, batch_last_index, complete=False
        )

        processed += len(operations)
        upserted += result.upserted_count
        modified += result.modified_count
        print(
            f"  {path.name}: processed={processed:,} | "
            f"new={upserted:,} | updated={modified:,} | "
            f"last_source_record={batch_last_index + 1:,}",
            flush=True,
        )
        batch = []

    for source_index, raw_document in enumerate(iter_json_documents(path)):
        if source_index < start_index:
            skipped += 1
            continue

        document = convert_decimals(raw_document)
        batch.append(document)
        batch_last_index = source_index

        if len(batch) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    # Mark complete only after the entire source file has been read and written.
    final_index = source_index if "source_index" in locals() else start_index - 1
    save_checkpoint(checkpoints, cp_id, signature, final_index, complete=True)

    return {
        "processed": processed,
        "upserted": upserted,
        "modified": modified,
        "skipped": skipped,
    }


# -------------------- INDEXES --------------------

def create_index_if_field_exists(collection, field, direction=ASCENDING):
    top_level_field = field.split(".")[0]
    sample = collection.find_one({}, {top_level_field: 1})
    if sample is not None and top_level_field in sample:
        collection.create_index([(field, direction)], background=True)
        print(f"  Index created: {collection.name}.{field}")


def create_indexes(db):
    print("\nCreating indexes...")

    index_plan = {
        "users": [("user_id", ASCENDING)],
        "categories": [("category_id", ASCENDING)],
        "products": [
            ("product_id", ASCENDING),
            ("category_id", ASCENDING),
            ("subcategory_id", ASCENDING),
            ("is_active", ASCENDING),
        ],
        "transactions": [
            ("transaction_id", ASCENDING),
            ("user_id", ASCENDING),
            ("timestamp", DESCENDING),
            ("transaction_date", DESCENDING),
        ],
        "sessions": [
            ("session_id", ASCENDING),
            ("user_id", ASCENDING),
            ("start_time", DESCENDING),
            ("viewed_products", ASCENDING),
        ],
    }

    for collection_name, fields in index_plan.items():
        collection = db[collection_name]
        for field, direction in fields:
            create_index_if_field_exists(collection, field, direction)

    compound_indexes = [
        ("transactions", ["user_id", "timestamp"]),
        ("sessions", ["user_id", "start_time"]),
        ("products", ["category_id", "is_active"]),
    ]

    for collection_name, fields in compound_indexes:
        collection = db[collection_name]
        projection = {field: 1 for field in fields}
        sample = collection.find_one({}, projection)
        if sample is not None and all(field in sample for field in fields):
            collection.create_index(
                [(field, ASCENDING) for field in fields],
                background=True,
            )
            print(f"  Compound index created: {collection_name} {fields}")


# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser(description="Resumable MongoDB bulk loader")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="DANGER: drop all target collections before importing.",
    )
    args = parser.parse_args()

    required_files = {}
    for name, (filename, key_field) in DATA_FILES.items():
        path = BASE_DIR / filename
        if not path.is_file():
            print(f"Missing required file: {path}")
            sys.exit(2)
        required_files[name] = path

    session_files = sorted(BASE_DIR.glob("sessions_*.json"), key=session_sort_key)
    if not session_files:
        print("No sessions_*.json files found.")
        sys.exit(2)

    print(f"\nSession files detected: {len(session_files)}")
    for path in session_files:
        print(f"  {path.name}")

    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        db = client[DATABASE_NAME]
        checkpoints = db[CHECKPOINT_COLLECTION]

        print(f"\nMongoDB connected: {DATABASE_NAME}")
        print(f"Data directory: {BASE_DIR}")

        if args.replace:
            print("\nWARNING: --replace specified. Dropping target collections...")
            for name in COLLECTION_NAMES:
                db[name].drop()
            checkpoints.drop()

        existing = {
            name: db[name].count_documents({})
            for name in COLLECTION_NAMES
        }
        print("\nExisting MongoDB records:")
        for name, count in existing.items():
            print(f"  {name:15s}: {count:,}")

        # Preserve users, matching the behavior of the original loader.
        skip_users = existing["users"] > 0
        if skip_users:
            print("\nExisting users collection will be preserved.")
            print("users.json will NOT be imported.")

        inputs = []
        if not skip_users:
            inputs.append(("users", required_files["users"], "user_id"))

        for name in ["categories", "products", "transactions"]:
            inputs.append((name, required_files[name], DATA_FILES[name][1]))

        for path in session_files:
            inputs.append(("sessions", path, SESSION_KEY))

        totals = {
            name: {"processed": 0, "upserted": 0, "modified": 0, "skipped": 0}
            for name in COLLECTION_NAMES
        }

        for collection_name, path, key_field in inputs:
            print(f"\nImporting {path.name} -> {collection_name}")
            print(f"Matching field: {key_field}")

            stats = import_file(db[collection_name], path, key_field, checkpoints)
            for metric, value in stats.items():
                totals[collection_name][metric] += value

            print(
                f"Finished {path.name}: processed={stats['processed']:,}, "
                f"new={stats['upserted']:,}, updated={stats['modified']:,}, "
                f"skipped={stats['skipped']:,}"
            )

        create_indexes(db)

        print("\n========== IMPORT SUMMARY ==========")
        for name in COLLECTION_NAMES:
            actual = db[name].count_documents({})
            stats = totals[name]
            print(
                f"{name:15s} processed={stats['processed']:>12,} "
                f"new={stats['upserted']:>12,} "
                f"updated={stats['modified']:>12,} "
                f"skipped={stats['skipped']:>12,} "
                f"total_in_db={actual:>12,}"
            )

        print("\nImport completed successfully.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
