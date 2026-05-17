# -*- coding: utf-8 -*-
"""
Realtime sleep-staging utilities for LPSGM.

The original inference path in ``web_demo/inference_backend.py`` is offline: it
slides a 20-epoch window over the whole night and votes over overlapping
windows. That gives good full-recording predictions, but it also lets an epoch
benefit from future epochs.

This module exposes a causal interface for real-time use. It keeps a rolling
history of the latest ``seq_len`` epochs, feeds that single window to LPSGM, and
returns the prediction for the newest epoch only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from model.model import LPSGM
from web_demo.inference_backend import CHANNEL_TO_INDEX, STAGE_TO_INDEX, args as DefaultArgs


INDEX_TO_STAGE = dict(STAGE_TO_INDEX)
STAGE_TO_PROB_KEY = {idx: f"p_{stage}" for idx, stage in INDEX_TO_STAGE.items()}


@dataclass(frozen=True)
class RealtimePrediction:
    """Prediction returned for one newly received 30-second epoch."""

    epoch_index: int
    stage_index: int
    stage: str
    probabilities: dict[str, float]
    context_size: int
    is_warmup: bool

    def to_dict(self) -> dict[str, int | float | str | bool]:
        row: dict[str, int | float | str | bool] = {
            "epoch_index": self.epoch_index,
            "stage_index": self.stage_index,
            "stage": self.stage,
            "context_size": self.context_size,
            "is_warmup": self.is_warmup,
        }
        row.update(self.probabilities)
        return row


def build_realtime_args(weights: str | Path | None = None, **overrides) -> SimpleNamespace:
    """Create a mutable args namespace compatible with ``model.model.LPSGM``."""

    keys = [
        "architecture",
        "ch_num",
        "ch_emb_dim",
        "seq_emb_dim",
        "seq_len",
        "num_transformer_blocks",
        "transformer_num_heads",
        "transformer_dropout",
        "transformer_attn_dropout",
        "epoch_encoder_dropout",
        "batch_size",
        "clamp_value",
        "weights",
    ]
    values = {key: getattr(DefaultArgs, key) for key in keys}
    if weights is not None:
        values["weights"] = str(weights)
    values.update(overrides)
    return SimpleNamespace(**values)


def load_lpsgm_model(
    model_args: SimpleNamespace,
    device: str | torch.device | None = None,
    *,
    random_weights: bool = False,
) -> tuple[nn.Module, torch.device]:
    """Instantiate LPSGM and load the configured checkpoint once."""

    resolved_device = torch.device(device) if isinstance(device, str) else (
        device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = LPSGM(model_args)

    if not random_weights:
        weights = Path(model_args.weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {weights}. Download the pretrained LPSGM "
                "weights and pass --weights, or use random_weights=True only for "
                "pipeline smoke tests."
            )

        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        cleaned_state = {}
        for key, value in state.items():
            cleaned_state[key[7:] if key.startswith("module.") else key] = value
        model.load_state_dict(cleaned_state)

    if resolved_device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(resolved_device)
    model.eval()
    return model, resolved_device


class RealtimeLPSGMSleepStager:
    """
    Causal real-time LPSGM wrapper.

    Parameters
    ----------
    min_context:
        Number of epochs required before returning predictions. Set this to
        ``model_args.seq_len`` to suppress the first warmup predictions.
    channel_order:
        Fixed LPSGM channel names to read from each epoch. When omitted, the
        first epoch determines the available channels in canonical LPSGM order.
    random_weights:
        Keeps the pipeline runnable without a checkpoint. Predictions are not
        meaningful in this mode.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        *,
        model_args: SimpleNamespace | None = None,
        model: nn.Module | None = None,
        device: str | torch.device | None = None,
        channel_order: Sequence[str] | None = None,
        epoch_samples: int = 3000,
        min_context: int = 1,
        random_weights: bool = False,
    ):
        self.args = model_args or build_realtime_args(weights)
        self.epoch_samples = int(epoch_samples)
        self.seq_len = int(self.args.seq_len)
        self.min_context = int(min_context)
        if self.min_context < 1 or self.min_context > self.seq_len:
            raise ValueError(f"min_context must be in [1, {self.seq_len}], got {min_context}")

        if model is None:
            self.model, self.device = load_lpsgm_model(
                self.args,
                device=device,
                random_weights=random_weights,
            )
        else:
            self.device = torch.device(device) if isinstance(device, str) else (
                device or next(model.parameters()).device
            )
            self.model = model.to(self.device).eval()

        self.channel_order = self._validate_channel_order(channel_order) if channel_order else None
        self.ch_id: np.ndarray | None = (
            np.array([CHANNEL_TO_INDEX[ch] for ch in self.channel_order], dtype=np.int64)
            if self.channel_order
            else None
        )
        self.history: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self.epoch_index = -1

    @staticmethod
    def _validate_channel_order(channel_order: Sequence[str]) -> tuple[str, ...]:
        unknown = [ch for ch in channel_order if ch not in CHANNEL_TO_INDEX]
        if unknown:
            raise ValueError(f"Unknown LPSGM channel(s): {unknown}")
        if len(set(channel_order)) != len(channel_order):
            raise ValueError(f"channel_order contains duplicates: {channel_order}")
        return tuple(channel_order)

    def reset(self) -> None:
        self.history.clear()
        self.epoch_index = -1

    def _init_channels_from_epoch(self, epoch: Mapping[str, np.ndarray]) -> None:
        channels = [ch for ch in CHANNEL_TO_INDEX if ch in epoch]
        if not channels:
            raise ValueError(
                "No supported LPSGM channels found in epoch. Expected one or more of "
                f"{list(CHANNEL_TO_INDEX)}."
            )
        self.channel_order = tuple(channels)
        self.ch_id = np.array([CHANNEL_TO_INDEX[ch] for ch in self.channel_order], dtype=np.int64)

    def _epoch_to_array(self, epoch: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.channel_order is None:
            self._init_channels_from_epoch(epoch)
        assert self.channel_order is not None

        missing = [ch for ch in self.channel_order if ch not in epoch]
        if missing:
            raise ValueError(f"Epoch is missing configured channel(s): {missing}")

        arrays = []
        for ch in self.channel_order:
            arr = np.asarray(epoch[ch], dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != self.epoch_samples:
                raise ValueError(
                    f"Channel {ch} must be a 1D array with {self.epoch_samples} samples, "
                    f"got shape {arr.shape}"
                )
            arrays.append(arr)

        return np.stack(arrays, axis=0)

    def _build_causal_window(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        assert self.ch_id is not None

        context_size = len(self.history)
        channel_count = len(self.ch_id)
        start = self.seq_len - context_size

        seq = np.zeros((self.seq_len, channel_count, self.epoch_samples), dtype=np.float32)
        mask = np.ones((self.seq_len, channel_count), dtype=np.bool_)

        if context_size:
            seq[start:] = np.stack(list(self.history), axis=0)
            mask[start:] = False

        seq_idx = np.arange(self.seq_len, dtype=np.int64).reshape(self.seq_len, 1)
        seq_idx = np.tile(seq_idx, (1, channel_count))
        ch_idx = np.tile(self.ch_id.reshape(1, channel_count), (self.seq_len, 1))

        return (
            seq.reshape(self.seq_len * channel_count, self.epoch_samples),
            mask.reshape(self.seq_len * channel_count),
            ch_idx.reshape(self.seq_len * channel_count),
            seq_idx.reshape(self.seq_len * channel_count),
            self.seq_len - 1,
        )

    @torch.no_grad()
    def push_epoch(self, epoch: Mapping[str, np.ndarray]) -> RealtimePrediction | None:
        """
        Add one preprocessed 30-second epoch and return its causal prediction.

        ``epoch`` maps LPSGM channel names such as ``C3`` or ``E1`` to arrays of
        length 3000, already filtered, resampled to 100 Hz, and normalized.
        """

        epoch_array = self._epoch_to_array(epoch)
        self.history.append(epoch_array)
        self.epoch_index += 1

        context_size = len(self.history)
        if context_size < self.min_context:
            return None

        seq_np, mask_np, ch_idx_np, seq_idx_np, output_pos = self._build_causal_window()

        seq_t = torch.as_tensor(seq_np[None, ...], dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask_np[None, ...], dtype=torch.bool, device=self.device)
        ch_idx_t = torch.as_tensor(ch_idx_np[None, ...], dtype=torch.int64, device=self.device)
        seq_idx_t = torch.as_tensor(seq_idx_np[None, ...], dtype=torch.int64, device=self.device)

        logits = self.model(seq_t, mask_t, ch_idx_t, seq_idx_t, None)
        current_logits = logits[0, output_pos]
        probs = torch.softmax(current_logits, dim=-1).detach().cpu().numpy()
        stage_index = int(np.argmax(probs))

        return RealtimePrediction(
            epoch_index=self.epoch_index,
            stage_index=stage_index,
            stage=INDEX_TO_STAGE[stage_index],
            probabilities={STAGE_TO_PROB_KEY[i]: float(probs[i]) for i in range(len(probs))},
            context_size=context_size,
            is_warmup=context_size < self.seq_len,
        )


def predict_processed_recording_realtime(
    sig_dict: Mapping[str, np.ndarray],
    stager: RealtimeLPSGMSleepStager,
) -> list[RealtimePrediction | None]:
    """
    Replay an already preprocessed recording through the real-time interface.

    ``sig_dict`` values must have shape ``(num_epochs, 3000)``.
    """

    if not sig_dict:
        raise ValueError("sig_dict is empty")

    epoch_counts = {ch: np.asarray(sig).shape[0] for ch, sig in sig_dict.items()}
    if len(set(epoch_counts.values())) != 1:
        raise ValueError(f"All channels must have the same epoch count, got {epoch_counts}")

    num_epochs = next(iter(epoch_counts.values()))
    predictions: list[RealtimePrediction | None] = []
    for epoch_i in range(num_epochs):
        epoch = {ch: np.asarray(sig)[epoch_i] for ch, sig in sig_dict.items()}
        predictions.append(stager.push_epoch(epoch))
    return predictions


class RealtimePSGPreprocessor:
    """
    Stateful raw-sample preprocessor for live PSG streams.

    This is intentionally conservative: it performs causal IIR filtering,
    accumulates complete 30-second epochs, resamples each complete epoch to
    100 Hz, and normalizes with either running or per-epoch statistics.
    """

    def __init__(
        self,
        channels: Iterable[str],
        sample_rate: float | Mapping[str, float],
        *,
        resample_rate: int = 100,
        epoch_seconds: int = 30,
        notch: bool = False,
        normalization: str = "running",
    ):
        self.channels = tuple(channels)
        unknown = [ch for ch in self.channels if ch not in CHANNEL_TO_INDEX]
        if unknown:
            raise ValueError(f"Unknown LPSGM channel(s): {unknown}")

        if isinstance(sample_rate, Mapping):
            self.sample_rates = {ch: float(sample_rate[ch]) for ch in self.channels}
        else:
            self.sample_rates = {ch: float(sample_rate) for ch in self.channels}
        invalid_rates = {ch: sr for ch, sr in self.sample_rates.items() if sr <= 0}
        if invalid_rates:
            raise ValueError(f"Sample rates must be positive, got {invalid_rates}")

        self.resample_rate = int(resample_rate)
        self.epoch_seconds = int(epoch_seconds)
        self.notch = bool(notch)
        self.normalization = normalization
        if normalization not in {"running", "epoch", "none"}:
            raise ValueError("normalization must be one of: running, epoch, none")

        self.buffers: MutableMapping[str, np.ndarray] = {
            ch: np.empty(0, dtype=np.float32) for ch in self.channels
        }
        self.filter_states = {ch: self._make_filter_chain(ch, self.sample_rates[ch]) for ch in self.channels}
        self.running_stats = {
            ch: {"n": 0, "mean": 0.0, "m2": 0.0} for ch in self.channels
        }

    def _make_filter_chain(self, ch: str, sample_rate: float):
        from scipy import signal

        filters = []
        nyquist = sample_rate / 2
        if ch in {"F3", "F4", "C3", "C4", "O1", "O2", "E1", "E2"}:
            high = min(35.0, nyquist * 0.95)
            sos = signal.butter(4, [0.3, high], btype="bandpass", fs=sample_rate, output="sos")
            filters.append({"sos": sos, "zi": None})
        elif ch == "Chin":
            highpass = min(10.0, nyquist * 0.95)
            sos = signal.butter(4, highpass, btype="highpass", fs=sample_rate, output="sos")
            filters.append({"sos": sos, "zi": None})

        if self.notch and nyquist > 50:
            b, a = signal.iirnotch(w0=50, Q=20, fs=sample_rate)
            filters.append({"sos": signal.tf2sos(b, a), "zi": None})

        return filters

    def _filter_chunk(self, ch: str, x: np.ndarray) -> np.ndarray:
        from scipy import signal

        y = x.astype(np.float32, copy=False)
        for entry in self.filter_states[ch]:
            sos = entry["sos"]
            if entry["zi"] is None:
                entry["zi"] = signal.sosfilt_zi(sos) * float(y[0])
            y, entry["zi"] = signal.sosfilt(sos, y, zi=entry["zi"])
        return y.astype(np.float32, copy=False)

    def _resample_epoch(self, ch: str, epoch: np.ndarray) -> np.ndarray:
        from scipy import signal

        sample_rate = self.sample_rates[ch]
        target_len = self.epoch_seconds * self.resample_rate
        if abs(sample_rate - self.resample_rate) < 1e-6:
            return epoch.astype(np.float32, copy=False)

        if float(sample_rate).is_integer():
            sr_int = int(round(sample_rate))
            frac = Fraction(self.resample_rate, sr_int)
            resampled = signal.resample_poly(epoch, frac.numerator, frac.denominator)
        else:
            resampled = signal.resample(epoch, target_len)

        if len(resampled) > target_len:
            resampled = resampled[:target_len]
        elif len(resampled) < target_len:
            resampled = np.pad(resampled, (0, target_len - len(resampled)))
        return resampled.astype(np.float32, copy=False)

    def _update_running_stats(self, ch: str, x: np.ndarray) -> tuple[float, float]:
        stats = self.running_stats[ch]
        batch_n = int(x.size)
        batch_mean = float(np.mean(x))
        batch_m2 = float(np.sum((x - batch_mean) ** 2))

        if stats["n"] == 0:
            stats["n"] = batch_n
            stats["mean"] = batch_mean
            stats["m2"] = batch_m2
        else:
            n_a = stats["n"]
            n_b = batch_n
            delta = batch_mean - stats["mean"]
            n = n_a + n_b
            stats["mean"] += delta * n_b / n
            stats["m2"] += batch_m2 + delta * delta * n_a * n_b / n
            stats["n"] = n

        variance = stats["m2"] / max(stats["n"] - 1, 1)
        return stats["mean"], float(np.sqrt(max(variance, 1e-8)))

    def _normalize_epoch(self, ch: str, epoch: np.ndarray) -> np.ndarray:
        if self.normalization == "none":
            return epoch.astype(np.float32, copy=False)
        if self.normalization == "epoch":
            mean = float(np.mean(epoch))
            std = float(np.std(epoch))
        else:
            mean, std = self._update_running_stats(ch, epoch)
        return ((epoch - mean) / max(std, 1e-6)).astype(np.float32, copy=False)

    def push_samples(self, samples: Mapping[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
        """
        Push a raw-sample chunk and return zero or more completed epochs.

        ``samples`` maps each configured channel to a one-dimensional chunk. All
        chunks do not need to have the same length, but each channel must advance
        according to its configured sample rate.
        """

        for ch in self.channels:
            if ch not in samples:
                raise ValueError(f"Missing raw sample chunk for channel {ch}")
            chunk = np.asarray(samples[ch], dtype=np.float32)
            if chunk.ndim != 1:
                raise ValueError(f"Raw samples for {ch} must be 1D, got {chunk.shape}")
            if chunk.size == 0:
                continue
            filtered = self._filter_chunk(ch, chunk)
            self.buffers[ch] = np.concatenate([self.buffers[ch], filtered])

        completed: list[dict[str, np.ndarray]] = []
        needed = {
            ch: int(round(self.sample_rates[ch] * self.epoch_seconds))
            for ch in self.channels
        }

        while all(self.buffers[ch].size >= needed[ch] for ch in self.channels):
            epoch: dict[str, np.ndarray] = {}
            for ch in self.channels:
                raw_epoch = self.buffers[ch][: needed[ch]]
                self.buffers[ch] = self.buffers[ch][needed[ch] :]
                resampled = self._resample_epoch(ch, raw_epoch)
                epoch[ch] = self._normalize_epoch(ch, resampled)
            completed.append(epoch)

        return completed


class RealtimeLPSGMPipeline:
    """Combine streaming preprocessing and LPSGM staging."""

    def __init__(self, preprocessor: RealtimePSGPreprocessor, stager: RealtimeLPSGMSleepStager):
        self.preprocessor = preprocessor
        self.stager = stager

    def push_samples(self, samples: Mapping[str, np.ndarray]) -> list[RealtimePrediction | None]:
        epochs = self.preprocessor.push_samples(samples)
        return [self.stager.push_epoch(epoch) for epoch in epochs]
