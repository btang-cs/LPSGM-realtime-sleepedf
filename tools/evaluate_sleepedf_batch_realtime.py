#!/usr/bin/env python3
"""Batch real-time causal LPSGM evaluation on Sleep-EDF sleep-cassette."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", message="Channels contain different highpass filters.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Channels contain different lowpass filters.*", category=RuntimeWarning)

import mne
import numpy as np
import torch
from mne.datasets.sleep_physionet import age
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from realtime_eval import predict_realtime_causal
from realtime_inference import build_realtime_args, load_lpsgm_model
from web_demo.utils import load_sig, pre_process


STAGE_LABELS = ["W", "N1", "N2", "N3", "R"]
LABELS = list(range(len(STAGE_LABELS)))
CHANNEL_MAP = {
    "C3": ("EEG Fpz-Cz",),
    "O1": ("EEG Pz-Oz",),
    "E1": ("EOG horizontal",),
    "Chin": ("EMG submental",),
}


@dataclass
class RecordResult:
    recording: str
    subject: int
    night: int
    status: str
    reason: str
    epochs_total: int = 0
    epochs_evaluated: int = 0
    epochs_after_warmup: int = 0
    accuracy: float | None = None
    accuracy_after_warmup: float | None = None
    non_wake_accuracy: float | None = None
    balanced_accuracy: float | None = None
    macro_f1: float | None = None
    weighted_f1: float | None = None
    kappa: float | None = None
    runtime_seconds: float | None = None
    confusion_matrix: list[list[int]] | None = None


def safe_metric(fn, y_true, y_pred, **kwargs):
    try:
        return float(fn(y_true, y_pred, **kwargs))
    except ValueError:
        return None


def sleepedf_records(data_dir: Path) -> list[dict]:
    rows_by_key: dict[tuple[int, int], dict] = {}
    with open(age.AGE_SLEEP_RECORDS, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (int(row["subject"]), int(row["night"]))
            rec = rows_by_key.setdefault(
                key,
                {
                    "subject": int(row["subject"]),
                    "night": int(row["night"]),
                    "age": row["age"],
                    "sex": row["sex"],
                    "lights_off": row["lights off"],
                },
            )
            if row["record type"] == "PSG":
                rec["psg"] = str(data_dir / row["fname"])
                rec["recording"] = row["fname"].replace("-PSG.edf", "")
            elif row["record type"] == "Hypnogram":
                rec["hypnogram"] = str(data_dir / row["fname"])
    return [rec for _, rec in sorted(rows_by_key.items()) if "psg" in rec and "hypnogram" in rec]


def sleepedf_labels(hypnogram_path: Path, n_epochs: int) -> np.ndarray:
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


def metrics_for_record(
    recording: str,
    subject: int,
    night: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    warmup_mask: np.ndarray,
    runtime_seconds: float,
) -> RecordResult:
    valid = np.isin(y_true, LABELS)
    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]
    after_warmup = valid & warmup_mask
    non_wake = valid & (y_true != 0)

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=LABELS)
    return RecordResult(
        recording=recording,
        subject=subject,
        night=night,
        status="evaluated",
        reason="ok",
        epochs_total=int(len(y_true)),
        epochs_evaluated=int(valid.sum()),
        epochs_after_warmup=int(after_warmup.sum()),
        accuracy=safe_metric(accuracy_score, y_true_valid, y_pred_valid),
        accuracy_after_warmup=safe_metric(accuracy_score, y_true[after_warmup], y_pred[after_warmup]),
        non_wake_accuracy=safe_metric(accuracy_score, y_true[non_wake], y_pred[non_wake]),
        balanced_accuracy=safe_metric(balanced_accuracy_score, y_true_valid, y_pred_valid),
        macro_f1=safe_metric(f1_score, y_true_valid, y_pred_valid, labels=LABELS, average="macro", zero_division=0),
        weighted_f1=safe_metric(
            f1_score, y_true_valid, y_pred_valid, labels=LABELS, average="weighted", zero_division=0
        ),
        kappa=safe_metric(cohen_kappa_score, y_true_valid, y_pred_valid),
        runtime_seconds=float(runtime_seconds),
        confusion_matrix=cm.astype(int).tolist(),
    )


def evaluate_record(record: dict, model, model_args, device: torch.device, batch_size: int):
    psg_path = Path(record["psg"])
    hypnogram_path = Path(record["hypnogram"])
    if not psg_path.exists() or not hypnogram_path.exists():
        return None, None, None, RecordResult(
            recording=record.get("recording", psg_path.stem),
            subject=record["subject"],
            night=record["night"],
            status="missing",
            reason="missing PSG or Hypnogram file",
        )

    started = time.time()
    _, raw = load_sig(str(psg_path), CHANNEL_MAP)
    processed = pre_process(raw, resample_rate=100, notch=False)
    n_epochs = min(np.asarray(sig).shape[0] for sig in processed.values())
    processed = {ch: np.asarray(sig[:n_epochs], dtype=np.float32) for ch, sig in processed.items()}
    y_true = sleepedf_labels(hypnogram_path, n_epochs)
    y_pred = predict_realtime_causal(processed, model, model_args, device, batch_size)
    warmup_mask = np.zeros(n_epochs, dtype=bool)
    warmup_mask[model_args.seq_len - 1 :] = True
    result = metrics_for_record(
        record["recording"],
        record["subject"],
        record["night"],
        y_true,
        y_pred,
        warmup_mask,
        time.time() - started,
    )
    return y_true, y_pred, warmup_mask, result


def aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray, warmup_mask: np.ndarray) -> dict:
    valid = np.isin(y_true, LABELS)
    after_warmup = valid & warmup_mask
    non_wake = valid & (y_true != 0)
    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=LABELS)
    return {
        "epochs_total": int(len(y_true)),
        "epochs_evaluated": int(valid.sum()),
        "epochs_after_warmup": int(after_warmup.sum()),
        "accuracy": safe_metric(accuracy_score, y_true_valid, y_pred_valid),
        "accuracy_after_warmup": safe_metric(accuracy_score, y_true[after_warmup], y_pred[after_warmup]),
        "non_wake_accuracy": safe_metric(accuracy_score, y_true[non_wake], y_pred[non_wake]),
        "balanced_accuracy": safe_metric(balanced_accuracy_score, y_true_valid, y_pred_valid),
        "macro_f1": safe_metric(f1_score, y_true_valid, y_pred_valid, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": safe_metric(
            f1_score, y_true_valid, y_pred_valid, labels=LABELS, average="weighted", zero_division=0
        ),
        "kappa": safe_metric(cohen_kappa_score, y_true_valid, y_pred_valid),
        "label_order": STAGE_LABELS,
        "true_distribution": {STAGE_LABELS[i]: int(np.sum(y_true_valid == i)) for i in LABELS},
        "pred_distribution": {STAGE_LABELS[i]: int(np.sum(y_pred_valid == i)) for i in LABELS},
        "confusion_matrix": cm.astype(int).tolist(),
        "classification_report": classification_report(
            y_true_valid,
            y_pred_valid,
            labels=LABELS,
            target_names=STAGE_LABELS,
            zero_division=0,
        ),
    }


def pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}%"


def write_report(output_dir: Path, summary: dict, record_results: list[RecordResult], paths: dict):
    cm = summary["confusion_matrix"]
    cm_rows = []
    for label, row in zip(STAGE_LABELS, cm):
        cm_rows.append("| {label} | {vals} |".format(label=label, vals=" | ".join(str(v) for v in row)))

    report = f"""# LPSGM Sleep-EDF 批量实时睡眠分期测试报告

生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}

## 1. 结论

本次对 Sleep-EDF Expanded sleep-cassette 可用公开样本执行 LPSGM 实时因果推理评估。每个 epoch 的预测只使用当前 epoch 和最多前 19 个历史 epoch，不使用未来 epoch，也不使用离线整晚窗口投票。

| 指标 | 结果 |
|---|---:|
| 成功评估 recording 数 | {summary["records_evaluated"]} / {summary["records_total"]} |
| 全记录整体准确率 | {pct(summary["accuracy"])} |
| 20-epoch warmup 后准确率 | {pct(summary["accuracy_after_warmup"])} |
| 非 Wake 睡眠期准确率 | {pct(summary["non_wake_accuracy"])} |
| Balanced accuracy | {pct(summary["balanced_accuracy"])} |
| Macro-F1 | {summary["macro_f1"]:.4f} |
| Weighted-F1 | {summary["weighted_f1"]:.4f} |
| Cohen's kappa | {summary["kappa"]:.4f} |
| 评估 epoch 数 | {summary["epochs_evaluated"]} |
| 评估总耗时（含读取/预处理/推理） | {summary["runtime_seconds"]:.2f} s |
| 平均每 epoch 评估时间 | {1000 * summary["runtime_seconds"] / max(summary["epochs_evaluated"], 1):.2f} ms |
| 评估吞吐 | {summary["epochs_evaluated"] / max(summary["runtime_seconds"], 1e-9):.2f} epochs/s |

说明：Sleep-EDF 中 Wake epoch 占比较高，因此整体准确率需要结合非 Wake 准确率、balanced accuracy 和 macro-F1 一起看。

## 2. 数据与任务设定

| 项目 | 值 |
|---|---|
| 数据集 | Sleep-EDF Expanded sleep-cassette |
| 数据目录 | `{paths["data_dir"]}` |
| PSG/Hypnogram 来源 | PhysioNet Sleep-EDF v1.0.0 |
| recording 数 | {summary["records_total"]} |
| epoch 长度 | 30 s |
| LPSGM 上下文长度 | 20 epochs |
| 类别顺序 | W, N1, N2, N3, R |

Sleep-EDF 的 `Sleep stage 3` 和 `Sleep stage 4` 均合并为 N3 类。

## 3. 实时推理流程

1. 读取每条 PSG 与对应 hypnogram。
2. 通过工程映射取 Sleep-EDF 中可用的 4 个通道：`C3 <- EEG Fpz-Cz`，`O1 <- EEG Pz-Oz`，`E1 <- EOG horizontal`，`Chin <- EMG submental`。
3. 对 EEG/EOG 做 0.3-35 Hz 滤波，对 Chin EMG 做 10 Hz 高通滤波。
4. 重采样到 100 Hz，并切分为 30 秒 epoch。
5. 第 `t` 个 epoch 只使用第 `t` 个 epoch 和最多前 19 个历史 epoch。
6. 开头不足 20 个 epoch 的位置用 padding 和 mask 处理。

## 4. 类别分布

真实类别分布：

| 类别 | epoch 数 |
|---|---:|
{chr(10).join(f"| {k} | {v} |" for k, v in summary["true_distribution"].items())}

预测类别分布：

| 类别 | epoch 数 |
|---|---:|
{chr(10).join(f"| {k} | {v} |" for k, v in summary["pred_distribution"].items())}

## 5. 混淆矩阵

行是真实标签，列是预测标签。

| True \\ Pred | W | N1 | N2 | N3 | R |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cm_rows)}

## 6. 局限

1. 这不是 LPSGM 官方测试集结果，而是公开 Sleep-EDF 上的本地实时协议验证。
2. Sleep-EDF 通道与 LPSGM 官方训练/测试 montage 不完全一致，本报告使用工程映射。
3. LPSGM 论文中的离线评估可使用未来窗口和投票，本报告为实时因果协议，不能直接与论文离线准确率横向比较。
4. 评估使用全记录预处理统计量；真正部署在线系统时，还应进一步验证在线归一化策略。

## 7. 输出文件

| 文件 | 内容 |
|---|---|
| `{paths["metrics_json"]}` | 汇总指标、混淆矩阵、分类报告 |
| `{paths["records_csv"]}` | 每条 recording 的指标 |
| `{paths["epochs_csv"]}` | 每个 epoch 的真实标签、预测标签、warmup 标记 |
| `{paths["manifest_json"]}` | 本次 Sleep-EDF 文件清单 |
"""
    report_path = output_dir / "lpsgm_realtime_sleepedf_batch_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LPSGM on all local Sleep-EDF sleep-cassette records.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/Volumes/Toms-Shield/CodexResearch/04_datasets/sleep-edf/physionet-sleep-data"),
    )
    parser.add_argument("--weights", default="weights/ched32_seqed64_ch9_seql20_block6.pth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/Volumes/Toms-Shield/CodexResearch/06_outputs/lpsgm_sleepedf_batch"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = sleepedf_records(args.data_dir)
    if args.limit:
        records = records[: args.limit]

    manifest_path = args.output_dir / "sleepedf_batch_manifest.json"
    manifest_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")

    model_args = build_realtime_args(weights=args.weights)
    model, device = load_lpsgm_model(model_args, device=args.device)

    record_results: list[RecordResult] = []
    true_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    warmup_all: list[np.ndarray] = []
    started_all = time.time()

    epochs_csv = args.output_dir / "lpsgm_realtime_sleepedf_batch_epochs.csv"
    with epochs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recording", "subject", "night", "epoch", "true_stage", "pred_stage", "is_after_warmup"])
        for record in tqdm(records, desc="Realtime Sleep-EDF batch"):
            y_true, y_pred, warmup_mask, result = evaluate_record(record, model, model_args, device, args.batch_size)
            record_results.append(result)
            if y_true is None or y_pred is None or warmup_mask is None:
                continue
            true_all.append(y_true)
            pred_all.append(y_pred)
            warmup_all.append(warmup_mask)
            for epoch_i, (true_i, pred_i, warm_i) in enumerate(zip(y_true, y_pred, warmup_mask)):
                true_stage = STAGE_LABELS[int(true_i)] if int(true_i) in LABELS else "UNK"
                pred_stage = STAGE_LABELS[int(pred_i)] if int(pred_i) in LABELS else "UNK"
                writer.writerow([record["recording"], record["subject"], record["night"], epoch_i, true_stage, pred_stage, bool(warm_i)])

    if not true_all:
        raise RuntimeError("No Sleep-EDF records were evaluated.")

    summary = aggregate_metrics(np.concatenate(true_all), np.concatenate(pred_all), np.concatenate(warmup_all))
    summary.update(
        {
            "records_total": len(records),
            "records_evaluated": sum(1 for result in record_results if result.status == "evaluated"),
            "runtime_seconds": time.time() - started_all,
            "weights": str(Path(args.weights).resolve()),
            "device": str(device),
            "protocol": "realtime_causal_current_epoch_uses_only_current_and_past_epochs_no_future_voting",
            "channel_map": CHANNEL_MAP,
        }
    )

    records_csv = args.output_dir / "lpsgm_realtime_sleepedf_batch_records.csv"
    with records_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(record_results[0]).keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in record_results:
            writer.writerow(asdict(result))

    metrics_json = args.output_dir / "lpsgm_realtime_sleepedf_batch_metrics.json"
    metrics_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "records": [asdict(result) for result in record_results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = write_report(
        args.output_dir,
        summary,
        record_results,
        {
            "data_dir": str(args.data_dir),
            "metrics_json": str(metrics_json),
            "records_csv": str(records_csv),
            "epochs_csv": str(epochs_csv),
            "manifest_json": str(manifest_path),
        },
    )

    print(f"Report: {report_path}")
    print(f"Metrics: {metrics_json}")
    print(f"Records: {records_csv}")
    print(f"Epochs: {epochs_csv}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Balanced accuracy: {summary['balanced_accuracy']:.4f}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
