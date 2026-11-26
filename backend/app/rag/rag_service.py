import hashlib
import importlib.util
import math
import re
from functools import lru_cache
from pathlib import Path

from app.core.config import BACKEND_ROOT, get_settings


def embed(text: str) -> list[float]:
    """Stable signed feature hashing: local lexical retrieval, not a semantic LLM."""
    vector = [0.0] * 384
    for token in re.findall(r"[a-z_][a-z_0-9]*", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        vector[int.from_bytes(digest[:4], "little") % 384] += 1 if digest[4] % 2 else -1
    norm = math.sqrt(sum(v * v for v in vector)) or 1
    return [v / norm for v in vector]


def available() -> bool:
    return importlib.util.find_spec("chromadb") is not None


@lru_cache
def collection():
    import chromadb

    path = Path(get_settings().chroma_path)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    client = chromadb.PersistentClient(
        path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        "repository_context_v1",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents(repository_id: str, documents: list[dict]) -> int:
    store = collection()
    # A guideline can also be a changed file. Index each path once, keeping the
    # latest supplied source, so Chroma receives unique chunk IDs.
    documents = list({doc["path"]: doc for doc in documents}.values())
    rows = []
    for doc in documents:
        for start in range(0, len(doc["content"]), 1400):
            text = doc["content"][start : start + 1800]
            identity = f"{repository_id}:{doc['path']}:{start}"
            rows.append(
                (
                    hashlib.sha256(identity.encode()).hexdigest(),
                    text,
                    {
                        "repository_id": repository_id,
                        "path": doc["path"],
                        "offset": start,
                    },
                )
            )
    if rows:
        # Replace only the submitted paths so shorter updates cannot leave stale chunks.
        for path in {d["path"] for d in documents}:
            store.delete(
                where={"$and": [{"repository_id": repository_id}, {"path": path}]}
            )
        store.upsert(
            ids=[r[0] for r in rows],
            documents=[r[1] for r in rows],
            embeddings=[embed(r[1]) for r in rows],
            metadatas=[r[2] for r in rows],
        )
    return len(rows)


def retrieve(repository_id: str, query: str) -> list[dict]:
    result = collection().query(
        query_embeddings=[embed(query)],
        n_results=4,
        where={"repository_id": repository_id},
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": text,
            "source": f"{meta['path']}@{meta['offset']}",
            "distance": distance,
        }
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
        if distance < 0.95
    ]
