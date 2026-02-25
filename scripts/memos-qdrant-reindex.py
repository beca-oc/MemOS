#!/usr/bin/env python3
"""Blue/green Qdrant collection reindex utility for MemOS."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


class QdrantClientHTTP:
    def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"api-key": api_key})
        self.session.headers.update({"Content-Type": "application/json"})

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self.session.request(method=method, url=url, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            body = resp.text.strip()
            raise RuntimeError(f"{method} {url} failed: HTTP {resp.status_code} - {body}")
        if not resp.text.strip():
            return {}
        try:
            return resp.json()
        except Exception as exc:
            raise RuntimeError(f"{method} {url} returned non-JSON response") from exc

    def collection_exists(self, name: str) -> bool:
        try:
            self._request("GET", f"/collections/{name}")
            return True
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                return False
            raise

    def get_collection(self, name: str) -> dict[str, Any]:
        response = self._request("GET", f"/collections/{name}")
        return response.get("result", {})

    def delete_collection(self, name: str) -> None:
        self._request("DELETE", f"/collections/{name}")

    def create_collection(
        self,
        name: str,
        vectors: dict[str, Any] | list[Any],
        hnsw_config: dict[str, Any] | None = None,
        optimizers_config: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"vectors": vectors}
        if hnsw_config:
            payload["hnsw_config"] = hnsw_config
        if optimizers_config:
            payload["optimizers_config"] = optimizers_config
        self._request("PUT", f"/collections/{name}", data=json.dumps(payload))

    def scroll_points(
        self, collection_name: str, limit: int, offset: Any | None
    ) -> tuple[list[dict[str, Any]], Any | None]:
        payload: dict[str, Any] = {
            "limit": limit,
            "with_payload": True,
            "with_vector": True,
        }
        if offset is not None:
            payload["offset"] = offset
        response = self._request(
            "POST",
            f"/collections/{collection_name}/points/scroll",
            data=json.dumps(payload),
        )
        result = response.get("result", {})
        points = result.get("points") or []
        next_offset = result.get("next_page_offset")
        return points, next_offset

    def upsert_points(self, collection_name: str, points: list[dict[str, Any]], wait: bool) -> None:
        self._request(
            "PUT",
            f"/collections/{collection_name}/points",
            params={"wait": "true" if wait else "false"},
            data=json.dumps({"points": points}),
        )

    def count_points(self, collection_name: str) -> int:
        response = self._request(
            "POST",
            f"/collections/{collection_name}/points/count",
            data=json.dumps({"exact": True}),
        )
        result = response.get("result", {})
        return int(result.get("count", 0))


def _add_if_not_none(target: dict[str, Any], key: str, value: Any | None) -> None:
    if value is not None:
        target[key] = value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reindex a Qdrant collection to a new destination collection (blue/green)."
    )
    parser.add_argument("--source", required=True, help="Source collection name")
    parser.add_argument("--dest", required=True, help="Destination collection name")
    parser.add_argument("--url", default=None, help="Qdrant base URL (e.g. http://localhost:6333)")
    parser.add_argument("--host", default="localhost", help="Qdrant host if --url is not set")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port if --url is not set")
    parser.add_argument("--api-key", default=None, help="Qdrant API key (optional)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for scroll+upsert")
    parser.add_argument(
        "--drop-dest",
        action="store_true",
        help="Delete destination collection first if it already exists",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for each upsert operation to finish",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print create payload and exit without reindexing",
    )
    parser.add_argument("--hnsw-m", type=int, default=None)
    parser.add_argument("--hnsw-ef-construct", type=int, default=None)
    parser.add_argument("--hnsw-full-scan-threshold", type=int, default=None)
    parser.add_argument("--hnsw-on-disk", choices=["true", "false"], default=None)
    parser.add_argument("--optimizer-indexing-threshold", type=int, default=None)
    parser.add_argument("--optimizer-memmap-threshold", type=int, default=None)
    parser.add_argument("--optimizer-default-segment-number", type=int, default=None)
    parser.add_argument("--optimizer-flush-interval-sec", type=int, default=None)
    parser.add_argument("--timeout-sec", type=float, default=30.0, help="HTTP timeout per request")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    base_url = args.url or f"http://{args.host}:{args.port}"
    client = QdrantClientHTTP(base_url=base_url, api_key=args.api_key, timeout=args.timeout_sec)

    source_collection = client.get_collection(args.source)
    if not source_collection:
        raise RuntimeError(f"Source collection '{args.source}' not found or has no config.")

    vectors = source_collection.get("config", {}).get("params", {}).get("vectors")
    if not vectors:
        raise RuntimeError(
            f"Unable to read vector config for source collection '{args.source}'. Aborting."
        )

    hnsw_config: dict[str, Any] = {}
    _add_if_not_none(hnsw_config, "m", args.hnsw_m)
    _add_if_not_none(hnsw_config, "ef_construct", args.hnsw_ef_construct)
    _add_if_not_none(hnsw_config, "full_scan_threshold", args.hnsw_full_scan_threshold)
    if args.hnsw_on_disk is not None:
        hnsw_config["on_disk"] = args.hnsw_on_disk == "true"

    optimizers_config: dict[str, Any] = {}
    _add_if_not_none(optimizers_config, "indexing_threshold", args.optimizer_indexing_threshold)
    _add_if_not_none(optimizers_config, "memmap_threshold", args.optimizer_memmap_threshold)
    _add_if_not_none(
        optimizers_config, "default_segment_number", args.optimizer_default_segment_number
    )
    _add_if_not_none(optimizers_config, "flush_interval_sec", args.optimizer_flush_interval_sec)

    if client.collection_exists(args.dest):
        if args.drop_dest:
            print(f"[reindex] deleting existing destination collection: {args.dest}")
            client.delete_collection(args.dest)
        else:
            raise RuntimeError(
                f"Destination collection '{args.dest}' already exists. "
                "Use --drop-dest to overwrite."
            )

    print(f"[reindex] source={args.source} dest={args.dest} url={base_url}")
    if args.dry_run:
        preview = {
            "vectors": vectors,
            "hnsw_config": hnsw_config or None,
            "optimizers_config": optimizers_config or None,
        }
        print(json.dumps(preview, indent=2))
        return 0

    print("[reindex] creating destination collection")
    client.create_collection(
        name=args.dest,
        vectors=vectors,
        hnsw_config=hnsw_config or None,
        optimizers_config=optimizers_config or None,
    )

    copied = 0
    batches = 0
    offset: Any | None = None
    while True:
        points, next_offset = client.scroll_points(
            collection_name=args.source, limit=args.batch_size, offset=offset
        )
        if not points:
            break

        client.upsert_points(collection_name=args.dest, points=points, wait=not args.no_wait)
        copied += len(points)
        batches += 1
        print(f"[reindex] batch={batches} copied={copied}")

        if next_offset is None:
            break
        offset = next_offset

    source_count = client.count_points(args.source)
    dest_count = client.count_points(args.dest)
    print(f"[reindex] source_count={source_count} dest_count={dest_count}")

    if source_count != dest_count:
        print("[reindex] ERROR: point-count mismatch after reindex", file=sys.stderr)
        return 1

    print("[reindex] success: destination collection is consistent")
    print(
        "[reindex] cutover: set QDRANT_COLLECTION_NAME to destination and restart memos-api-docker"
    )
    print(f"[reindex] rollback: set QDRANT_COLLECTION_NAME back to {args.source} and restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
