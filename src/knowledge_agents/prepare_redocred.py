from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from .redocred import load_and_sample, write_jsonl, write_manifest


SOURCE_REVISION = "ccfb54f5ddf5836027c87badda10f6dfc56efaac"
SOURCE_SHA256 = "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f"
SOURCE_URL = (
    "https://raw.githubusercontent.com/tonytan48/Re-DocRED/"
    f"{SOURCE_REVISION}/data/dev_revised.json"
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_verified(url: str, destination: str | Path, expected_sha256: str, force: bool = False) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        if sha256(path) == expected_sha256:
            return path
        raise FileExistsError(f"{path} exists with an unexpected hash; use --force to replace it.")
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        actual = sha256(temporary)
        if actual != expected_sha256:
            raise ValueError(f"Downloaded Re-DocRED hash mismatch: {actual}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, verify and freeze the official Re-DocRED evaluation subset.")
    parser.add_argument("--relation-map", default="data/docred_relations.json")
    parser.add_argument("--source", default="external/Re-DocRED/data/dev_revised.json")
    parser.add_argument("--output", default="data/redocred_eval_100.jsonl")
    parser.add_argument("--manifest", default="data/redocred_eval_100.manifest.json")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = download_verified(SOURCE_URL, args.source, SOURCE_SHA256, args.force)
    samples = load_and_sample(source, args.relation_map, args.sample_size, args.seed)
    dataset = write_jsonl(samples, args.output)
    manifest = write_manifest(
        samples,
        dataset,
        source,
        args.manifest,
        relation_map_path=args.relation_map,
        sample_size=args.sample_size,
        seed=args.seed,
        source_revision=SOURCE_REVISION,
    )
    print(f"Prepared {len(samples)} frozen Re-DocRED samples at {dataset}")
    print(f"Verified manifest: {manifest}")


if __name__ == "__main__":
    main()
