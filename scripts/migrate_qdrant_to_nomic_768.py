#!/usr/bin/env python3
"""Consolidate historical Qdrant collections into one canonical nomic 768d collection.

Workflow:
1) Copy missing points from 768d source collections directly.
2) Re-embed remaining missing points from 1536d collections using local embeddings API.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from typing import Any, Iterable


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_EMBED_URL = "http://127.0.0.1:8010/v1/embeddings"
DEFAULT_MODEL = "nomic-embed-text-v1.5.Q4_K_M.gguf"
DEFAULT_DEST = "neo4j_vec_db_nomic_v1_5_768_v1"
DEFAULT_768_SOURCES = [
    "neo4j_vec_db",
    "neo4j_vec_db_hnsw_v1",
    "neo4j_vec_db_nomic_768_v1",
]
DEFAULT_1536_SOURCES = [
    "neo4j_vec_db_openai_small_1536_v1",
    "neo4j_vec_db_openai_small_1536_rebuild_v2",
]


def http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
    retries: int = 4,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, method=method, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"HTTP {method} {url} failed after {retries} attempts: {last_exc}")


def id_key(point_id: Any) -> str:
    return f"{type(point_id).__name__}:{point_id}"


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def collection_dim(qdrant_url: str, collection: str) -> int:
    meta = http_json("GET", f"{qdrant_url}/collections/{collection}")
    result = meta.get("result", {})
    vectors = result.get("config", {}).get("params", {}).get("vectors", {})
    if isinstance(vectors, dict) and "size" in vectors:
        return int(vectors["size"])
    raise RuntimeError(f"Unable to read vector dimension for collection={collection}")


def collection_count(qdrant_url: str, collection: str) -> int:
    res = http_json(
        "POST",
        f"{qdrant_url}/collections/{collection}/points/count",
        payload={"exact": True},
    )
    return int(res.get("result", {}).get("count", 0))


def scroll_points(
    qdrant_url: str,
    collection: str,
    with_payload: bool,
    with_vector: bool,
    limit: int,
) -> Iterable[dict[str, Any]]:
    offset = None
    while True:
        body: dict[str, Any] = {
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if offset is not None:
            body["offset"] = offset
        res = http_json(
            "POST",
            f"{qdrant_url}/collections/{collection}/points/scroll",
            payload=body,
            timeout=180,
        )
        result = res.get("result", {})
        points = result.get("points", [])
        for point in points:
            yield point
        offset = result.get("next_page_offset")
        if not offset:
            break


def upsert_points(qdrant_url: str, collection: str, points: list[dict[str, Any]]) -> None:
    if not points:
        return
    res = http_json(
        "PUT",
        f"{qdrant_url}/collections/{collection}/points",
        payload={"points": points},
        timeout=180,
    )
    if res.get("status") != "ok":
        raise RuntimeError(f"Qdrant upsert failed for collection={collection}: {res}")


def choose_text(payload: dict[str, Any]) -> str:
    for key in ("memory", "text", "content", "summary", "key"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def embed_texts(embed_url: str, model: str, texts: list[str]) -> list[list[float]]:
    res = http_json("POST", embed_url, payload={"model": model, "input": texts}, timeout=180)
    data = res.get("data", [])
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError(f"Unexpected embedding response size: expected={len(texts)} got={len(data)}")
    return [item.get("embedding", []) for item in data]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge historical Qdrant collections into canonical nomic 768d collection."
    )
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dest", default=DEFAULT_DEST)
    parser.add_argument("--source-768", action="append", default=[])
    parser.add_argument("--source-1536", action="append", default=[])
    parser.add_argument("--scroll-batch", type=int, default=512)
    parser.add_argument("--upsert-batch", type=int, default=128)
    parser.add_argument("--embed-batch", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_768 = args.source_768 or DEFAULT_768_SOURCES
    source_1536 = args.source_1536 or DEFAULT_1536_SOURCES

    print(f"[migrate] dest={args.dest}")
    print(f"[migrate] 768_sources={source_768}")
    print(f"[migrate] 1536_sources={source_1536}")

    dest_dim = collection_dim(args.qdrant_url, args.dest)
    if dest_dim != 768:
        raise RuntimeError(f"Destination collection {args.dest} is dim={dest_dim}, expected 768")

    seen_ids: set[str] = set()
    for point in scroll_points(
        args.qdrant_url, args.dest, with_payload=False, with_vector=False, limit=args.scroll_batch
    ):
        seen_ids.add(id_key(point.get("id")))

    print(f"[migrate] initial_dest_ids={len(seen_ids)} count={collection_count(args.qdrant_url, args.dest)}")

    copied_768 = 0
    skipped_bad_vector = 0

    for src in source_768:
        if src == args.dest:
            continue
        src_dim = collection_dim(args.qdrant_url, src)
        if src_dim != 768:
            raise RuntimeError(f"Configured 768 source has dim={src_dim}: {src}")
        batch: list[dict[str, Any]] = []
        src_new = 0
        for point in scroll_points(
            args.qdrant_url, src, with_payload=True, with_vector=True, limit=args.scroll_batch
        ):
            pid = point.get("id")
            pid_key = id_key(pid)
            if pid_key in seen_ids:
                continue
            vector = point.get("vector")
            payload = point.get("payload") or {}
            if not isinstance(vector, list) or len(vector) != 768:
                skipped_bad_vector += 1
                continue
            batch.append({"id": pid, "vector": vector, "payload": payload})
            seen_ids.add(pid_key)
            src_new += 1
            copied_768 += 1
            if len(batch) >= args.upsert_batch:
                if not args.dry_run:
                    upsert_points(args.qdrant_url, args.dest, batch)
                batch = []
        if batch and not args.dry_run:
            upsert_points(args.qdrant_url, args.dest, batch)
        print(f"[migrate] copied_from_768 source={src} inserted={src_new}")

    reembedded = 0
    skipped_no_text = 0

    for src in source_1536:
        src_dim = collection_dim(args.qdrant_url, src)
        if src_dim != 1536:
            raise RuntimeError(f"Configured 1536 source has dim={src_dim}: {src}")

        staged: list[dict[str, Any]] = []
        src_new = 0
        for point in scroll_points(
            args.qdrant_url, src, with_payload=True, with_vector=False, limit=args.scroll_batch
        ):
            pid = point.get("id")
            pid_key = id_key(pid)
            if pid_key in seen_ids:
                continue
            payload = point.get("payload") or {}
            text = choose_text(payload)
            if not text:
                skipped_no_text += 1
                continue
            staged.append({"id": pid, "payload": payload, "text": text})
            seen_ids.add(pid_key)
            src_new += 1

        print(f"[migrate] reembed_candidates source={src} candidates={src_new}")
        for group in chunked(staged, args.embed_batch):
            texts = [item["text"] for item in group]
            vectors = embed_texts(args.embed_url, args.model, texts)
            points = []
            for item, vector in zip(group, vectors, strict=False):
                if not isinstance(vector, list) or len(vector) != 768:
                    continue
                points.append(
                    {
                        "id": item["id"],
                        "vector": vector,
                        "payload": item["payload"],
                    }
                )
            if points:
                if not args.dry_run:
                    upsert_points(args.qdrant_url, args.dest, points)
                reembedded += len(points)
            if reembedded and reembedded % 500 == 0:
                print(f"[migrate] reembedded_progress={reembedded}")

    final_count = collection_count(args.qdrant_url, args.dest)
    print(f"[migrate] copied_768={copied_768}")
    print(f"[migrate] reembedded_1536_to_768={reembedded}")
    print(f"[migrate] skipped_bad_vector={skipped_bad_vector}")
    print(f"[migrate] skipped_no_text={skipped_no_text}")
    print(f"[migrate] final_dest_count={final_count}")
    print(f"[migrate] expected_ids_seen={len(seen_ids)}")

    if final_count != len(seen_ids):
        print(
            f"[migrate] WARNING count mismatch final_dest_count={final_count} expected_seen={len(seen_ids)}",
            file=sys.stderr,
        )
    else:
        print("[migrate] success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
