#!/usr/bin/env python3
"""Rebuild Qdrant vectors from Neo4j Memory nodes using OpenAI embeddings.

Use this for clean, source-of-truth reindexing:
1) Read all :Memory nodes from Neo4j.
2) Re-embed each node memory with OpenAI.
3) Upsert into a fresh Qdrant destination collection.
4) Verify destination count matches Neo4j count.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_NEO4J_HTTP = "http://127.0.0.1:7474/db/neo4j/tx/commit"
DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_DIM = 1536
DEFAULT_PAGE_SIZE = 400
DEFAULT_BATCH_SIZE = 64
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


def get_env_defaults() -> dict[str, str]:
    env_vals = read_env_file(Path("/Users/aviyashchin/beca/MemOS/.env"))
    defaults = {}
    defaults["OPENAI_API_KEY"] = (
        os.getenv("OPENAI_API_KEY")
        or env_vals.get("OPENAI_API_KEY")
        or env_vals.get("MOS_EMBEDDER_API_KEY")
        or ""
    )
    defaults["QDRANT_COLLECTION_NAME"] = env_vals.get("QDRANT_COLLECTION_NAME", "")
    defaults["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD") or env_vals.get("NEO4J_PASSWORD", "12345678")
    return defaults


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    body = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method=method, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    return str(value)


def sanitize_text(text: Any, max_chars: int) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def neo4j_query(
    neo4j_url: str, neo4j_user: str, neo4j_password: str, statement: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    payload = {
        "statements": [
            {
                "statement": statement,
                "parameters": params or {},
            }
        ]
    }
    basic = f"{neo4j_user}:{neo4j_password}".encode("utf-8")
    auth = "Basic " + base64.b64encode(basic).decode("ascii")
    response = http_json(
        "POST",
        neo4j_url,
        payload=payload,
        headers={"Authorization": auth},
        timeout=120,
    )
    errors = response.get("errors", [])
    if errors:
        raise RuntimeError(f"Neo4j query failed: {errors}")
    return response.get("results", [{}])[0].get("data", [])


def embed_batch_openai(api_key: str, model: str, texts: list[str], retries: int = 5) -> list[list[float]]:
    url = "https://api.openai.com/v1/embeddings"
    payload = {"model": model, "input": texts}
    headers = {"Authorization": f"Bearer {api_key}"}
    for attempt in range(1, retries + 1):
        try:
            data = http_json("POST", url, payload=payload, headers=headers, timeout=120)
            return [item["embedding"] for item in data["data"]]
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(attempt * 1.5)
                continue
            raise
        except Exception:
            if attempt < retries:
                time.sleep(attempt * 1.5)
                continue
            raise
    raise RuntimeError("unreachable")


def create_destination_collection(
    qdrant_url: str, template_collection: str, dest_collection: str, dim: int, overwrite: bool
) -> None:
    template = http_json("GET", f"{qdrant_url}/collections/{template_collection}")
    if template.get("status") != "ok":
        raise RuntimeError(f"Unable to read template collection: {template_collection}")
    try:
        existing = http_json("GET", f"{qdrant_url}/collections/{dest_collection}")
        if existing.get("status") == "ok":
            if not overwrite:
                raise RuntimeError(
                    f"Destination collection already exists: {dest_collection} (pass --overwrite to recreate)"
                )
            deleted = http_json("DELETE", f"{qdrant_url}/collections/{dest_collection}")
            if deleted.get("status") != "ok":
                raise RuntimeError(f"Failed to delete destination collection: {deleted}")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    cfg = template["result"]["config"]
    payload = {
        "vectors": {
            "size": dim,
            "distance": cfg["params"]["vectors"]["distance"],
        },
        "hnsw_config": cfg.get("hnsw_config", {}),
        "optimizers_config": cfg.get("optimizer_config", {}),
    }
    created = http_json("PUT", f"{qdrant_url}/collections/{dest_collection}", payload=payload)
    if created.get("status") != "ok":
        raise RuntimeError(f"Failed to create destination collection: {created}")


def upsert_points(qdrant_url: str, collection: str, points: list[dict[str, Any]]) -> None:
    payload = {"points": points}
    result = http_json("PUT", f"{qdrant_url}/collections/{collection}/points", payload=payload, timeout=120)
    if result.get("status") != "ok":
        raise RuntimeError(f"Qdrant upsert failed: {result}")


def collection_count(qdrant_url: str, collection: str) -> int:
    payload = {"exact": True}
    result = http_json("POST", f"{qdrant_url}/collections/{collection}/points/count", payload=payload)
    return int(result["result"]["count"])


def parse_args(defaults: dict[str, str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neo4j-url", default=DEFAULT_NEO4J_HTTP)
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=defaults["NEO4J_PASSWORD"])
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument(
        "--template-collection",
        default=defaults["QDRANT_COLLECTION_NAME"] or "neo4j_vec_db_openai_small_1536_v1",
    )
    parser.add_argument(
        "--dest-collection",
        default="neo4j_vec_db_openai_small_1536_rebuild_v2",
    )
    parser.add_argument("--openai-api-key", default=defaults["OPENAI_API_KEY"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    defaults = get_env_defaults()
    args = parse_args(defaults)
    if not args.openai_api_key:
        print("ERROR: missing OPENAI_API_KEY", file=sys.stderr)
        return 1

    total_row = neo4j_query(
        args.neo4j_url,
        args.neo4j_user,
        args.neo4j_password,
        "MATCH (n:Memory) RETURN count(n) AS c",
    )
    source_count = int(total_row[0]["row"][0]) if total_row else 0
    print(f"[reindex] source_neo4j_count={source_count}")
    if source_count == 0:
        print("[reindex] nothing to process")
        return 0

    create_destination_collection(
        args.qdrant_url,
        args.template_collection,
        args.dest_collection,
        args.dim,
        overwrite=args.overwrite,
    )
    print(f"[reindex] destination_created={args.dest_collection}")

    processed = 0
    skip = 0
    started = time.time()
    cube_counter: Counter[str] = Counter()

    while True:
        rows = neo4j_query(
            args.neo4j_url,
            args.neo4j_user,
            args.neo4j_password,
            """
            MATCH (n:Memory)
            RETURN n.id AS id, n.memory AS memory, properties(n) AS props
            ORDER BY n.id
            SKIP $skip LIMIT $limit
            """,
            params={"skip": skip, "limit": args.page_size},
        )
        if not rows:
            break

        page_payloads: list[dict[str, Any]] = []
        page_texts: list[str] = []

        for item in rows:
            row = item.get("row", [])
            if len(row) != 3:
                continue
            node_id, memory, props = row
            if not node_id or memory is None:
                continue
            safe_props = sanitize_value(props if isinstance(props, dict) else {})
            if not isinstance(safe_props, dict):
                safe_props = {"raw_props": str(safe_props)}
            safe_props.pop("embedding", None)
            safe_props.pop("id", None)
            safe_props["memory"] = memory

            cube = safe_props.get("user_name") or safe_props.get("user_id") or "unknown"
            cube_counter[str(cube)] += 1
            page_payloads.append({"id": node_id, "payload": safe_props})
            page_texts.append(sanitize_text(memory, args.max_chars))

        for i in range(0, len(page_texts), args.batch_size):
            sub_texts = page_texts[i : i + args.batch_size]
            sub_payloads = page_payloads[i : i + args.batch_size]
            vectors = embed_batch_openai(args.openai_api_key, args.model, sub_texts)
            points = []
            for meta, vector in zip(sub_payloads, vectors, strict=True):
                points.append(
                    {
                        "id": meta["id"],
                        "payload": meta["payload"],
                        "vector": vector,
                    }
                )
            upsert_points(args.qdrant_url, args.dest_collection, points)
            processed += len(points)
            elapsed = time.time() - started
            print(f"[reindex] processed={processed}/{source_count} elapsed_s={elapsed:.1f}")

        skip += len(rows)

    dest_count = collection_count(args.qdrant_url, args.dest_collection)
    print(f"[reindex] dest_count={dest_count}")
    if dest_count != source_count:
        print(f"[reindex] ERROR count mismatch source={source_count} dest={dest_count}", file=sys.stderr)
        return 1

    print("[reindex] top_cubes:")
    for cube, cnt in cube_counter.most_common(20):
        print(f"  - {cube}: {cnt}")

    print("[reindex] success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
