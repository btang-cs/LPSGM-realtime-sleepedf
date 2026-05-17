# -*- coding: utf-8 -*-
"""
Batch evaluation for causal real-time LPSGM sleep staging.

This script evaluates already preprocessed LPSGM NPZ datasets or a small EDF
manifest. Unlike the original offline inference path, each epoch prediction only
uses current and past epochs.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from realtime_demo import load_channel_map
from realtime_inference import build_realtime_args, load_lpsgm_model
from web_demo.inference_backend import CHANNEL_TO_INDEX
from web_demo.utils import load_sig, pre_process


STAGE_LABELS = ["W", "N1", "N2", "N3", "R"]
INDEX_TO_STAGE = {idx: stage for idx, stage in enumerate(STAGE_LABELS)}
STAGE_TO_INDEX = {stage: idx for idx, stage in INDEX_TO_STAGE.items()}


@dataclass
class DatasetResult:
    dataset: str
    status: str
    reason: str
    subjects: int = 0
    epochs_total: int = 0
    epochs_evaluated_all: int = 0
    epochs_evaluated_after_warmup: int = 0
    accuracy_all: float | None = None
    accuracy_after_warmup: float | None = None
    balanced_accuracy_all: float | None = None
    macro_f1_all: float | None = None
    weighted_f1_all: float | None = None
    kappa_all: float | None = None
    confusion_matrix_all: list[list[int]] | None = None
    classification_report_all: str | None = None


def _load_checkpoint_model(weights: str, device: str | None):
    args = build_realtime_args(weights=weights)
    model, resolved_device = load_lpsgm_model(args, device=device)
    return args, model, resolved_device


def _load_subject_npz(subject_dir: Path, selected_channels: Sequence[str] | None = None):
    sig_segments: list[dict[str, np.ndarray]] = []
    label_segments: list[np.ndarray] = []
    channel_segments: list[list[str]] = []

    for seq_path in sorted(subject_dir.glob("*.npz")):
        npz = np.load(seq_path)
        if "Hypnogram" not in npz.files:
            continue

        channels = [ch for ch in CHANNEL_TO_INDEX if ch in npz.files]
        if selected_channels:
            channels = [ch for ch in selected_channels if ch in channels]
        if not channels:
            continue

        labels = npz["Hypnogram"].astype(np.int64)
        signals = {ch: np.asarray(npz[ch], dtype=np.float32) for ch in channels}
        n_epochs = min([labels.shape[0], *[sig.shape[0] for sig in signals.values()]])
        if n_epochs <= 0:
            continue
        sig_segments.append({ch: sig[:n_epochs] for ch, sig in signals.items()})
        label_segments.append(labels[:n_epochs])
        channel_segments.append(channels)

    return sig_segments, label_segments, channel_segments


def _causal_windows_for_batch(
    sig: np.ndarray,
    ch_id: np.ndarray,
    epoch_indices: np.ndarray,
    seq_len: int,
):
    channel_count = len(ch_id)
    epoch_samples = sig.shape[-1]
    batch_size = len(epoch_indices)

    seq = np.zeros((batch_size, seq_len, channel_count, epoch_samples), dtype=np.float32)
    mask = np.ones((batch_size, seq_len, channel_count), dtype=np.bool_)
    for batch_i, epoch_i in enumerate(epoch_indices):
        start = max(0, int(epoch_i) - seq_len + 1)
        context = sig[start : int(epoch_i) + 1]
        context_size = context.shape[0]
        seq[batch_i, seq_len - context_size :] = context
        mask[batch_i, seq_len - context_size :] = False

    seq_idx = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len, 1)
    seq_idx = np.tile(seq_idx, (batch_size, 1, channel_count)).reshape(batch_size, seq_len * channel_count)
    ch_idx = ch_id.reshape(1, 1, channel_count)
    ch_idx = np.tile(ch_idx, (batch_size, seq_len, 1)).reshape(batch_size, seq_len * channel_count)

    return (
        seq.reshape(batch_size, seq_len * channel_count, epoch_samples),
        mask.reshape(batch_size, seq_len * channel_count),
        ch_idx,
        seq_idx,
    )


@torch.no_grad()
def predict_realtime_causal(
    sig_dict: Mapping[str, np.ndarray],
    model,
    model_args: SimpleNamespace,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    channels = [ch for ch in CHANNEL_TO_INDEX if ch in sig_dict]
    if not channels:
        raise ValueError("No supported LPSGM channels available")
    epoch_counts = {ch: np.asarray(sig_dict[ch]).shape[0] for ch in channels}
    if len(set(epoch_counts.values())) != 1:
        raise ValueError(f"Channel epoch counts differ: {epoch_counts}")

    ch_id = np.array([CHANNEL_TO_INDEX[ch] for ch in channels], dtype=np.int64)
    sig = np.stack([np.asarray(sig_dict[ch], dtype=np.float32) for ch in channels], axis=1)
    n_epochs = sig.shape[0]
    pred = np.empty(n_epochs, dtype=np.int64)

    model.eval()
    batch_num = ceil(n_epochs / batch_size)
    for batch_i in range(batch_num):
        epoch_indices = np.arange(batch_i * batch_size, min(n_epochs, (batch_i + 1) * batch_size))
        seq_np, mask_np, ch_idx_np, seq_idx_np = _causal_windows_for_batch(
            sig=sig,
            ch_id=ch_id,
            epoch_indices=epoch_indices,
            seq_len=model_args.seq_len,
        )
        seq_t = torch.as_tensor(seq_np, dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        ch_idx_t = torch.as_tensor(ch_idx_np, dtype=torch.int64, device=device)
        seq_idx_t = torch.as_tensor(seq_idx_np, dtype=torch.int64, device=device)
        logits_t = model(seq_t, mask_t, ch_idx_t, seq_idx_t, None)
        pred[epoch_indices] = torch.argmax(logits_t[:, model_args.seq_len - 1], dim=-1).cpu().numpy()

    return pred


def _safe_metric(fn, y_true, y_pred, **kwargs):
    try:
        return float(fn(y_true, y_pred, **kwargs))
    except ValueError:
        return None


def _metrics(
    dataset: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seq_len: int,
    subjects: int,
    warmup_mask: np.ndarray | None = None,
) -> DatasetResult:
    valid = np.isin(y_true, np.arange(len(STAGE_LABELS)))
    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]

    if warmup_mask is None:
        after_warmup = np.zeros_like(valid, dtype=bool)
        after_warmup[seq_len - 1 :] = True
    else:
        after_warmup = warmup_mask.astype(bool, copy=False)
    after_warmup &= valid

    labels = list(range(len(STAGE_LABELS)))
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=labels)
    report = classification_report(
        y_true_valid,
        y_pred_valid,
        labels=labels,
        target_names=STAGE_LABELS,
        zero_division=0,
    )
    return DatasetResult(
        dataset=dataset,
        status="evaluated",
        reason="ok",
        subjects=subjects,
        epochs_total=int(len(y_true)),
        epochs_evaluated_all=int(len(y_true_valid)),
        epochs_evaluated_after_warmup=int(after_warmup.sum()),
        accuracy_all=_safe_metric(accuracy_score, y_true_valid, y_pred_valid),
        accuracy_after_warmup=_safe_metric(accuracy_score, y_true[after_warmup], y_pred[after_warmup]),
        balanced_accuracy_all=_safe_metric(balanced_accuracy_score, y_true_valid, y_pred_valid),
        macro_f1_all=_safe_metric(f1_score, y_true_valid, y_pred_valid, labels=labels, average="macro", zero_division=0),
        weighted_f1_all=_safe_metric(f1_score, y_true_valid, y_pred_valid, labels=labels, average="weighted", zero_division=0),
        kappa_all=_safe_metric(cohen_kappa_score, y_true_valid, y_pred_valid),
        confusion_matrix_all=cm.astype(int).tolist(),
        classification_report_all=report,
    )


def evaluate_npz_dataset(
    dataset_root: Path,
    dataset_name: str,
    model,
    model_args: SimpleNamespace,
    device: torch.device,
    batch_size: int,
    channels: Sequence[str] | None,
):
    if not dataset_root.exists():
        return DatasetResult(dataset_name, "missing", f"Dataset directory not found: {dataset_root}")

    subject_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if not subject_dirs:
        return DatasetResult(dataset_name, "missing", f"No subject directories found under {dataset_root}")

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_warmup: list[np.ndarray] = []
    evaluated_subjects = 0

    for subject_dir in tqdm(subject_dirs, desc=f"Realtime eval {dataset_name}"):
        sig_segments, label_segments, _ = _load_subject_npz(subject_dir, channels)
        subject_true = []
        subject_pred = []
        subject_warmup = []
        for sig_dict, labels in zip(sig_segments, label_segments):
            pred = predict_realtime_causal(sig_dict, model, model_args, device, batch_size)
            n = min(len(labels), len(pred))
            subject_true.append(labels[:n])
            subject_pred.append(pred[:n])
            warmup = np.zeros(n, dtype=bool)
            warmup[model_args.seq_len - 1 :] = True
            subject_warmup.append(warmup)
        if subject_true:
            all_true.append(np.concatenate(subject_true))
            all_pred.append(np.concatenate(subject_pred))
            all_warmup.append(np.concatenate(subject_warmup))
            evaluated_subjects += 1

    if not all_true:
        return DatasetResult(dataset_name, "missing", f"No evaluable NPZ files with Hypnogram and channels in {dataset_root}")

    return _metrics(
        dataset_name,
        np.concatenate(all_true),
        np.concatenate(all_pred),
        model_args.seq_len,
        evaluated_subjects,
        warmup_mask=np.concatenate(all_warmup),
    )


def _sleepedf_labels(hypnogram_path: Path, n_epochs: int) -> np.ndarray:
    import mne

    ann = mne.read_annotations(str(hypnogram_path))
    labels = np.full(n_epochs, 9, dtype=np.int64)
    stage_map = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        stage = stage_map.get(str(desc))
        if stage is None:
            continue
        start = max(0, int(round(float(onset) / 30.0)))
        end = min(n_epochs, int(round((float(onset) + float(duration)) / 30.0)))
        labels[start:end] = stage
    return labels


def evaluate_edf_item(
    item: dict,
    model,
    model_args: SimpleNamespace,
    device: torch.device,
    batch_size: int,
) -> DatasetResult:
    psg_path = Path(item["psg"])
    hypnogram_path = Path(item["hypnogram"])
    channel_map = load_channel_map(item.get("channel_map_json"))
    if "channel_map" in item:
        channel_map = item["channel_map"]
    if not psg_path.exists() or not hypnogram_path.exists():
        return DatasetResult(item["name"], "missing", f"Missing PSG or hypnogram file for {item['name']}")

    _, raw = load_sig(str(psg_path), channel_map)
    processed = pre_process(raw, resample_rate=100, notch=bool(item.get("notch", False)))
    n_epochs = min(np.asarray(sig).shape[0] for sig in processed.values())
    processed = {ch: np.asarray(sig[:n_epochs], dtype=np.float32) for ch, sig in processed.items()}
    labels = _sleepedf_labels(hypnogram_path, n_epochs)
    pred = predict_realtime_causal(processed, model, model_args, device, batch_size)
    warmup = np.zeros(n_epochs, dtype=bool)
    warmup[model_args.seq_len - 1 :] = True
    return _metrics(item["name"], labels, pred, model_args.seq_len, subjects=1, warmup_mask=warmup)


def write_outputs(results: list[DatasetResult], output_dir: Path, manifest: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "realtime_eval_results.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "manifest": manifest,
                "results": [asdict(result) for result in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = output_dir / "realtime_eval_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "dataset",
            "status",
            "reason",
            "subjects",
            "epochs_evaluated_all",
            "accuracy_all",
            "accuracy_after_warmup",
            "balanced_accuracy_all",
            "macro_f1_all",
            "weighted_f1_all",
            "kappa_all",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({field: row.get(field) for field in fields})

    md_path = output_dir / "realtime_eval_report.md"
    rows = []
    for result in results:
        rows.append(
            "| {dataset} | {status} | {subjects} | {epochs} | {acc} | {mf1} | {kappa} | {reason} |".format(
                dataset=result.dataset,
                status=result.status,
                subjects=result.subjects,
                epochs=result.epochs_evaluated_all,
                acc="" if result.accuracy_all is None else f"{result.accuracy_all:.4f}",
                mf1="" if result.macro_f1_all is None else f"{result.macro_f1_all:.4f}",
                kappa="" if result.kappa_all is None else f"{result.kappa_all:.4f}",
                reason=result.reason.replace("|", "/"),
            )
        )

    md = f"""# LPSGM Real-time Evaluation Report

## Scope

This report evaluates LPSGM with a causal real-time protocol: each epoch uses only current and past epochs. It does not use the original offline future-window voting.

## Summary

| Dataset | Status | Subjects | Epochs | Accuracy | Macro F1 | Kappa | Note |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Official Dataset Availability

The repository identifies HANG7 and SYSU as the primary official private sleep-staging test centers. Their raw data paths in the preprocessing scripts point to the authors' private storage and are not included in this GitHub checkout. Missing rows above mean the metric was not computed locally because the data files are absent.

## Artifacts

- JSON: `{json_path}`
- CSV: `{csv_path}`
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate LPSGM with causal real-time windows")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--weights", type=str, default="weights/ched32_seqed64_ch9_seql20_block6.pth")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_args, model, device = _load_checkpoint_model(args.weights, args.device)

    results: list[DatasetResult] = []
    for item in manifest.get("npz_datasets", []):
        results.append(
            evaluate_npz_dataset(
                dataset_root=Path(item["root"]),
                dataset_name=item["name"],
                model=model,
                model_args=model_args,
                device=device,
                batch_size=args.batch_size,
                channels=item.get("channels"),
            )
        )
    for item in manifest.get("edf_items", []):
        results.append(evaluate_edf_item(item, model, model_args, device, args.batch_size))

    json_path, csv_path, md_path = write_outputs(results, Path(args.output_dir), manifest)
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Report: {md_path}")
    for result in results:
        if result.accuracy_all is None:
            print(f"{result.dataset}: {result.status} - {result.reason}")
        else:
            print(f"{result.dataset}: accuracy={result.accuracy_all:.4f}, macro_f1={result.macro_f1_all:.4f}, kappa={result.kappa_all:.4f}")


if __name__ == "__main__":
    main()
