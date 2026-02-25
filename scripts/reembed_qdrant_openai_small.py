#!/usr/bin/env python3
"""Re-embed a Qdrant collection with OpenAI text-embedding-3-small (1536d).

This script performs a safe blue/green migration:
1) Read points from source collection.
2) Recompute embeddings via OpenAI.
3) Upsert into destination collection (same ids + payload, new vectors).
4) Verify point-count parity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1536
DEFAULT_BATCH = 64
DEFAULT_MAX_CHARS = 12000


def read_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def get_openai_key() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env_vals = read_env_file(Path("/Users/aviyashchin/beca/MemOS/.env"))
    for key in ("OPENAI_API_KEY", "MOS_EMBEDDER_API_KEY"):
        value = env_vals.get(key, "")
        if value:
            return value
    return ""


def http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict:
    body = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    if not raw:
        return {}
    return json.loads(raw)


def create_destination_collection(qdrant_url: str, source: str, dest: str, dim: int) -> None:
    source_meta = http_json("GET", f"{qdrant_url}/collections/{source}")
    if source_meta.get("status") != "ok":
        raise RuntimeError(f"Unable to read source collection metadata: {source}")

    src_cfg = source_meta["result"]["config"]["params"]
    vector_distance = src_cfg["vectors"]["distance"]

    payload = {
        "vectors": {
            "size": dim,
            "distance": vector_distance,
        },
        "hnsw_config": source_meta["result"]["config"].get("hnsw_config", {}),
        "optimizers_config": source_meta["result"]["config"].get("optimizer_config", {}),
    }

    try:
        existing = http_json("GET", f"{qdrant_url}/collections/{dest}")
        if existing.get("status") == "ok":
            raise RuntimeError(
                f"Destination collection already exists: {dest}. "
                "Use a new destination name or delete it first."
            )
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    created = http_json("PUT", f"{qdrant_url}/collections/{dest}", payload)
    if created.get("status") != "ok":
        raise RuntimeError(f"Failed creating destination collection: {created}")


def iter_source_points(qdrant_url: str, source: str, limit: int):
    next_offset = None
    while True:
        payload = {
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }
        if next_offset is not None:
            payload["offset"] = next_offset
        response = http_json(
            "POST", f"{qdrant_url}/collections/{source}/points/scroll", payload, timeout=120
        )
        points = response.get("result", {}).get("points", [])
        if not points:
            break
        for point in points:
            yield point
        next_offset = response.get("result", {}).get("next_page_offset")
        if next_offset is None:
            break


def sanitize_text(payload: dict, max_chars: int) -> str:
    # Prefer canonical memory text; fall back to other text-ish payload fields.
    text = payload.get("memory") or payload.get("text") or payload.get("key") or ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        text = json.dumps(payload, ensure_ascii=False)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def embed_batch_openai(api_key: str, model: str, texts: list[str], retries: int = 5) -> list[list[float]]:
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "input": texts}
    for attempt in range(1, retries + 1):
        try:
            data = http_json("POST", url, payload, headers=headers, timeout=120)
            return [item["embedding"] for item in data["data"]]
        except urllib.error.HTTPError as e:
            # 429/5xx can be retried with exponential backoff.
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            raise
        except Exception:
            if attempt < retries:
                time.sleep(1.5 * attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def upsert_batch(qdrant_url: str, dest: str, points: list[dict]) -> None:
    payload = {"points": points}
    result = http_json("PUT", f"{qdrant_url}/collections/{dest}/points", payload, timeout=120)
    if result.get("status") != "ok":
        raise RuntimeError(f"Failed upsert to {dest}: {result}")


def collection_count(qdrant_url: str, name: str) -> int:
    result = http_json(
        "POST",
        f"{qdrant_url}/collections/{name}/points/count",
        payload={"exact": True},
        timeout=60,
    )
    return int(result["result"]["count"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="neo4j_vec_db_hnsw_v1",
        help="Source collection name",
    )
    parser.add_argument(
        "--dest",
        default="neo4j_vec_db_openai_small_1536_v1",
        help="Destination collection name",
    )
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Destination vector dimension")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Embed/upsert batch size")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Max chars per text before embedding",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = get_openai_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment or MemOS .env", file=sys.stderr)
        return 1

    print(f"[reembed] source={args.source} dest={args.dest} model={args.model} dim={args.dim}")
    create_destination_collection(args.qdrant_url, args.source, args.dest, args.dim)
    print("[reembed] destination collection created")

    total = 0
    batch_payloads: list[dict] = []
    batch_texts: list[str] = []
    started = time.time()

    def flush() -> None:
        nonlocal total, batch_payloads, batch_texts
        if not batch_texts:
            return
        vectors = embed_batch_openai(api_key, args.model, batch_texts)
        points = []
        for payload, vector in zip(batch_payloads, vectors, strict=True):
            points.append(
                {
                    "id": payload["id"],
                    "payload": payload["payload"],
                    "vector": vector,
                }
            )
        upsert_batch(args.qdrant_url, args.dest, points)
        total += len(points)
        elapsed = time.time() - started
        print(f"[reembed] copied={total} elapsed_s={elapsed:.1f}")
        batch_payloads = []
        batch_texts = []

    for point in iter_source_points(args.qdrant_url, args.source, args.batch_size):
        payload = point.get("payload", {})
        batch_payloads.append({"id": point["id"], "payload": payload})
        batch_texts.append(sanitize_text(payload, args.max_chars))
        if len(batch_texts) >= args.batch_size:
            flush()
    flush()

    source_count = collection_count(args.qdrant_url, args.source)
    dest_count = collection_count(args.qdrant_url, args.dest)
    print(f"[reembed] source_count={source_count} dest_count={dest_count}")
    if source_count != dest_count:
        print("[reembed] ERROR: count mismatch", file=sys.stderr)
        return 1

    print("[reembed] success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
