from __future__ import annotations

from dataclasses import dataclass

try:
    from pymongo import MongoClient
except ImportError:  # pragma: no cover - optional during local dev
    MongoClient = None


@dataclass
class MongoClientFactory:
    uri: str
    db_name: str

    def get_collection(self, collection_name: str):
        if MongoClient is None:
            raise RuntimeError("pymongo package is required for MongoDB access")
        client = MongoClient(self.uri)
        database = client[self.db_name]
        return database[collection_name]