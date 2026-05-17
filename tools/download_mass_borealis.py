#!/usr/bin/env python3
"""Download MASS-SS1/SS3 EDF files from Borealis/Dataverse when access is granted.

The MASS biosignal files are restricted on Borealis. This script still discovers
the official file ids and can download them once the Dataverse account/token has
been granted access by the MASS data owners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


DATASETS = {
    "SS1": {
        "name": "MASS-SS1 Biosignals and Sleep stages",
        "doi": "doi:10.5683/SP3/OVISPE",
        "dir": "SS1",
    },
    "SS3": {
        "name": "MASS-SS3 Biosignals and Sleep stages",
        "doi": "doi:10.5683/SP3/9MYUCS",
        "dir": "SS3",
    },
}


def default_mass_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parents[1] if len(repo_root.parents) >= 2 else repo_root
    candidate = workspace_root / "04_datasets" / "MASS"
    if candidate.parent.exists():
        return candidate
    return repo_root / "datasets" / "MASS"


def fetch_json(url: str, token: str | None = None) -> dict:
    req = urllib.request.Request(url)
    if token:
        req.add_header("X-Dataverse-key", token)
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def dataset_metadata(doi: str, token: str | None = None) -> dict:
    url = "https://borealisdata.ca/api/datasets/:persistentId/?" + urllib.parse.urlencode(
        {"persistentId": doi}
    )
    return fetch_json(url, token=token)["data"]["latestVersion"]


def iter_edf_files(subset: str, metadata: dict):
    for file_entry in metadata["files"]:
        filename = file_entry["dataFile"]["filename"]
        if filename.endswith(" PSG.edf") or filename.endswith(" Base.edf"):
            yield {
                "subset": subset,
                "filename": filename,
                "id": file_entry["dataFile"]["id"],
                "size": int(file_entry["dataFile"]["filesize"]),
                "md5": file_entry["dataFile"].get("checksum", {}).get("value"),
                "restricted": bool(file_entry.get("restricted")),
            }


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(file_info: dict, output_path: Path, token: str | None, verify_md5: bool) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size == file_info["size"]:
        if not verify_md5 or file_md5(output_path).lower() == str(file_info["md5"]).lower():
            return "exists"

    url = f"https://borealisdata.ca/api/access/datafile/{file_info['id']}"
    cmd = [
        "curl",
        "--fail",
        "--location",
        "--continue-at",
        "-",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--output",
        str(output_path),
    ]
    if token:
        cmd.extend(["--header", f"X-Dataverse-key: {token}"])
    cmd.append(url)
    subprocess.run(cmd, check=True)

    if output_path.stat().st_size != file_info["size"]:
        raise RuntimeError(
            f"Size mismatch for {output_path.name}: got {output_path.stat().st_size}, "
            f"expected {file_info['size']}"
        )
    if verify_md5 and file_info["md5"]:
        observed = file_md5(output_path).lower()
        expected = str(file_info["md5"]).lower()
        if observed != expected:
            raise RuntimeError(f"MD5 mismatch for {output_path.name}: got {observed}, expected {expected}")
    return "downloaded"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MASS-SS1/SS3 EDF files from Borealis.")
    parser.add_argument("--output-root", type=Path, default=default_mass_root())
    parser.add_argument("--subsets", nargs="+", choices=sorted(DATASETS), default=["SS1", "SS3"])
    parser.add_argument("--token", default=None, help="Dataverse API token with approved MASS access.")
    parser.add_argument("--token-env", default="DATAVERSE_API_TOKEN")
    parser.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-md5", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit downloaded files per subset.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between file downloads.")
    args = parser.parse_args()

    token = args.token or os.environ.get(args.token_env)
    all_files: list[dict] = []
    manifests: dict[str, dict] = {}

    for subset in args.subsets:
        info = DATASETS[subset]
        metadata = dataset_metadata(info["doi"], token=token)
        files = list(iter_edf_files(subset, metadata))
        if args.limit:
            files = files[: args.limit]
        all_files.extend(files)
        manifests[subset] = {
            "name": info["name"],
            "doi": info["doi"],
            "version": f"{metadata.get('versionNumber')}.{metadata.get('versionMinorNumber')}",
            "termsOfAccess": metadata.get("termsOfAccess"),
            "file_count": len(files),
            "restricted_file_count": sum(1 for file in files if file["restricted"]),
            "total_size_bytes": sum(file["size"] for file in files),
            "files": files,
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "borealis_mass_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        for subset in args.subsets:
            subset_files = manifests[subset]["files"]
            total_gib = manifests[subset]["total_size_bytes"] / 1024**3
            print(f"{subset}: {len(subset_files)} files, {total_gib:.2f} GiB")
        print(f"Wrote manifest: {manifest_path}")
        return 0

    restricted = [file for file in all_files if file["restricted"]]
    if restricted and not token and not args.allow_unauthenticated:
        print(f"Wrote manifest: {manifest_path}")
        print(
            "MASS files are restricted on Borealis. Set DATAVERSE_API_TOKEN to an account "
            "with approved MASS access, or pass --allow-unauthenticated to intentionally test "
            "the unauthenticated endpoint."
        )
        return 2

    for file_info in all_files:
        subset_dir = args.output_root / DATASETS[file_info["subset"]]["dir"]
        output_path = subset_dir / file_info["filename"]
        status = download_file(file_info, output_path, token=token, verify_md5=args.verify_md5)
        print(f"{file_info['subset']} {file_info['filename']}: {status}")
        if args.sleep:
            time.sleep(args.sleep)

    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
