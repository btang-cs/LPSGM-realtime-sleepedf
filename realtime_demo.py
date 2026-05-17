# -*- coding: utf-8 -*-
"""
Command-line demo for causal real-time LPSGM sleep staging.

Examples
--------
Replay a preprocessed NPZ with channel arrays shaped (N, 3000):

    python realtime_demo.py --processed-npz sample.npz --weights weights/ched32_seqed64_ch9_seql20_block6.pth

Replay an EDF through the original offline EDF preprocessing, then feed epochs
one by one into the causal real-time stager:

    python realtime_demo.py --edf subject.edf --channel-map-json channel_map.json --weights weights/model.pth

Use ``--random-weights`` only to smoke-test the data path when a checkpoint is
not available. The resulting stages are not clinically meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from realtime_inference import (
    RealtimeLPSGMSleepStager,
    predict_processed_recording_realtime,
)
from web_demo.utils import load_sig, pre_process


DEFAULT_CHANNEL_MAP = {
    "F3": (("F3", "M2"), "F3", "F3-M2", "F3-A2"),
    "F4": (("F4", "M1"), "F4", "F4-M1", "F4-A1"),
    "C3": (("C3", "M2"), "C3", "C3-M2", "C3-A2"),
    "C4": (("C4", "M1"), "C4", "C4-M1", "C4-A1"),
    "O1": (("O1", "M2"), "O1", "O1-M2", "O1-A2"),
    "O2": (("O2", "M1"), "O2", "O2-M1", "O2-A1"),
    "E1": (("E1", "M2"), "E1", "E1-M2", "LOC-M2"),
    "E2": (("E2", "M1"), "E2", "E2-M1", "ROC-M1"),
    "Chin": ("Chin", "EMG Chin", "EMG submental"),
}


def _json_option_to_tuple(value: Any):
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(item, str) for item in value):
            return tuple(value)
        return tuple(_json_option_to_tuple(item) for item in value)
    return value


def load_channel_map(path: str | Path | None):
    if path is None:
        return DEFAULT_CHANNEL_MAP
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    channel_map = {}
    for target, options in raw.items():
        if isinstance(options, str):
            options = [options]
        channel_map[target] = tuple(_json_option_to_tuple(option) for option in options)
    return channel_map


def load_processed_npz(path: str | Path) -> dict[str, np.ndarray]:
    npz = np.load(path)
    sig_dict = {}
    for key in npz.files:
        arr = np.asarray(npz[key])
        if arr.ndim == 2 and arr.shape[1] == 3000:
            sig_dict[key] = arr.astype(np.float32)
    if not sig_dict:
        raise ValueError(f"No channel arrays shaped (N, 3000) found in {path}")
    return sig_dict


def load_processed_edf(path: str | Path, channel_map_path: str | Path | None, resample_rate: int, notch: bool):
    channel_map = load_channel_map(channel_map_path)
    _, raw_sig = load_sig(str(path), channel_map)
    if not raw_sig:
        raise ValueError("No configured channels could be loaded from the EDF file")
    return pre_process(raw_sig, resample_rate=resample_rate, notch=notch)


def write_predictions(path: str | Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Causal real-time LPSGM sleep-staging demo")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--processed-npz", type=str, help="NPZ containing preprocessed channel arrays shaped (N, 3000)")
    src.add_argument("--edf", type=str, help="EDF file to preprocess and replay epoch by epoch")
    parser.add_argument("--channel-map-json", type=str, default=None, help="JSON channel map for EDF loading")
    parser.add_argument("--weights", type=str, default="weights/ched32_seqed64_ch9_seql20_block6.pth")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, cuda:0, or mps")
    parser.add_argument("--min-context", type=int, default=1, help="Return predictions after this many epochs")
    parser.add_argument("--max-epochs", type=int, default=None, help="Limit replayed epochs for quick tests")
    parser.add_argument("--resample-rate", type=int, default=100)
    parser.add_argument("--notch", action="store_true")
    parser.add_argument("--random-weights", action="store_true", help="Smoke-test only; predictions are meaningless")
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args()

    if args.processed_npz:
        sig_dict = load_processed_npz(args.processed_npz)
    else:
        sig_dict = load_processed_edf(args.edf, args.channel_map_json, args.resample_rate, args.notch)

    if args.max_epochs is not None:
        sig_dict = {ch: sig[: args.max_epochs] for ch, sig in sig_dict.items()}

    stager = RealtimeLPSGMSleepStager(
        weights=args.weights,
        device=args.device,
        channel_order=list(sig_dict.keys()),
        min_context=args.min_context,
        random_weights=args.random_weights,
    )
    predictions = predict_processed_recording_realtime(sig_dict, stager)

    rows = []
    for item in predictions:
        if item is None:
            continue
        row = item.to_dict()
        rows.append(row)
        print(
            f"epoch={row['epoch_index']:04d} stage={row['stage']} "
            f"context={row['context_size']} warmup={row['is_warmup']}"
        )

    if args.output_csv:
        write_predictions(args.output_csv, rows)
        print(f"Saved {len(rows)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
