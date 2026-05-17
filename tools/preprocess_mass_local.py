#!/usr/bin/env python3
"""Preprocess locally downloaded MASS-SS1/SS3 EDF files into LPSGM NPZ format."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


CHANNEL_ID = {
    "F3": ("EEG F3-CLE", "EEG F3-LER"),
    "F4": ("EEG F4-CLE", "EEG F4-LER"),
    "C3": ("EEG C3-CLE", "EEG C3-LER"),
    "C4": ("EEG C4-CLE", "EEG C4-LER"),
    "O1": ("EEG O1-CLE", "EEG O1-LER"),
    "O2": ("EEG O2-CLE", "EEG O2-LER"),
    "E1": ("EOG Left Horiz",),
    "E2": ("EOG Right Horiz",),
    "Chin": (("EMG Chin1", "EMG Chin2"),),
}


def default_mass_root(repo_root: Path) -> Path:
    workspace_root = repo_root.parents[1] if len(repo_root.parents) >= 2 else repo_root
    candidate = workspace_root / "04_datasets" / "MASS"
    if candidate.parent.exists():
        return candidate
    return repo_root / "datasets" / "MASS"


def load_mass_module(repo_root: Path):
    preprocess_dir = repo_root / "preprocess"
    sys.path.insert(0, str(preprocess_dir))
    module_path = preprocess_dir / "MASS-SS1-SS3.py"
    spec = importlib.util.spec_from_file_location("lpsgm_mass_preprocess", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subject_ids(raw_dir: Path) -> list[str]:
    ids = set()
    for edf_path in raw_dir.glob("*.edf"):
        ids.add(edf_path.name.split(" ")[0])
    return sorted(ids)


def preprocess_subset(module, subset: str, raw_dir: Path, dst_root: Path, limit: int | None):
    module.channel_id = CHANNEL_ID
    module.resample_rate = 100
    module.src_root = str(raw_dir)
    module.dst_root = str(dst_root)
    module.SUB_REMOVE = []

    inputs = []
    for sub_id in subject_ids(raw_dir):
        sig_path = raw_dir / f"{sub_id} PSG.edf"
        ano_path = raw_dir / f"{sub_id} Base.edf"
        if sig_path.exists() and ano_path.exists():
            inputs.append((sub_id, str(sig_path), str(ano_path)))
    if limit:
        inputs = inputs[:limit]

    if not inputs:
        raise FileNotFoundError(f"No complete MASS {subset} PSG/Base EDF pairs found under {raw_dir}")

    for args in tqdm(inputs, desc=f"Preprocess MASS-{subset}"):
        module.process_recording(*args)
    module.formatting_check(str(dst_root))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Preprocess MASS EDF files for LPSGM.")
    parser.add_argument("--mass-root", type=Path, default=default_mass_root(repo_root))
    parser.add_argument("--output-root", type=Path, default=repo_root / "data")
    parser.add_argument("--subsets", nargs="+", choices=["SS1", "SS3"], default=["SS1", "SS3"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    module = load_mass_module(repo_root)
    for subset in args.subsets:
        raw_dir = args.mass_root / subset
        dst_root = args.output_root / f"MASS-{subset}"
        if args.overwrite:
            shutil.rmtree(dst_root, ignore_errors=True)
        dst_root.mkdir(parents=True, exist_ok=True)
        preprocess_subset(module, subset, raw_dir, dst_root, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
