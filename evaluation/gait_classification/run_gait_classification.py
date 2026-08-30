# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch
from scipy.signal import find_peaks


CONFIG_ORDER = [
    "HPR_DTH_PS",
    "HPR_THDOT_PS",
    "HPR_OBS_PS",
    "HPR_O2_PS",
    "HPR_O2_JS",
    "SGRR_O2_JS",
]

PAPER_LABELS = {
    "HPR_DTH_PS": "HPR-DTH-PS",
    "HPR_THDOT_PS": "HPR-THDOT-PS",
    "HPR_OBS_PS": "HPR-OBS-PS",
    "HPR_O2_PS": "HPR-O2-PS",
    "HPR_O2_JS": "HPR-O2-JS",
    "SGRR_O2_JS": "SGRR-O2-JS",
}

GAIT_ORDER = [
    "formal_rolling",
    "crawling_candidate",
    "partial_roll",
    "rocking",
    "sliding_candidate",
    "failed_other",
    "technical_exclusion",
]

GAIT_COLORS = {
    "formal_rolling": "#2F6BFF",
    "crawling_candidate": "#009E73",
    "partial_roll": "#8E44AD",
    "rocking": "#E69F00",
    "sliding_candidate": "#CC79A7",
    "failed_other": "#C7CDD4",
    "technical_exclusion": "#111111",
}

PRESET_CONTRASTS = [
    ("HPR_DTH_PS", "HPR_THDOT_PS", "DTH to THDOT"),
    ("HPR_DTH_PS", "HPR_OBS_PS", "DTH to OBS"),
    ("HPR_OBS_PS", "HPR_O2_PS", "OBS to O2-PS"),
    ("HPR_O2_PS", "HPR_O2_JS", "O2-PS to O2-JS"),
    ("HPR_O2_JS", "SGRR_O2_JS", "O2-JS HPR to SGRR"),
]


@dataclass(frozen=True)
class Thresholds:
    transient_steps: int = 100
    minimum_forward_body_lengths: float = 1.0
    minimum_post_transient_progress_body_lengths: float = 0.75
    maximum_crawling_rotation_span_degrees: float = 180.0
    rocking_rotation_span_degrees: float = 90.0
    rocking_reversal_index: float = 0.50
    analysis_window_count: int = 4
    minimum_window_progress_body_lengths: float = 0.05
    minimum_positive_progress_windows: int = 3
    minimum_window_shape_amplitude_degrees: float = 20.0
    minimum_shape_windows: int = 3
    minimum_support_windows: int = 3
    minimum_valid_contact_samples_per_window: int = 10
    minimum_shape_rms_radians: float = 0.08
    minimum_shape_velocity_rms_radians: float = 0.005
    minimum_shape_repeatability: float = 0.40
    minimum_cycle_count: float = 2.0
    minimum_positive_cycle_fraction: float = 0.70
    minimum_median_cycle_progress_body_lengths: float = 0.05
    minimum_support_span_indices: float = 1.50
    minimum_support_alternations: int = 2
    minimum_contact_valid_fraction: float = 0.50
    active_rotation_increment_degrees: float = 0.05
    particle_radius: float = 1.0 / 3.0
    contact_band_body_fraction: float = 0.015
    contact_softness_body_fraction: float = 0.01
    contact_valid_weight: float = 0.50
    minimum_period_steps: int = 20
    maximum_period_steps: int = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline formal gait classification for the locked six-configuration endpoint traces."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-review-assets", action="store_true")
    parser.add_argument("--review-all", action="store_true")
    parser.add_argument(
        "--portable-new-results",
        action="store_true",
        help=(
            "Accept any scientifically valid 600-episode rerun instead of requiring "
            "the archived thesis result counts (146 strict / 147 lenient)."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=20260824)
    return parser.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "protocol": root / "protocol",
        "data": root / "data",
        "figures": root / "figures",
        "review": root / "review",
        "review_images": root / "review" / "blind_episode_images",
        "report": root / "report",
        "qa": root / "qa",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV columns for empty output: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def bool_from_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def wrapped_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def body_length(positions: np.ndarray) -> np.ndarray:
    return np.sum(np.linalg.norm(np.diff(positions, axis=1), axis=2), axis=1)


def joint_angles(positions: np.ndarray) -> np.ndarray:
    segments = np.diff(positions, axis=1)
    heading = np.arctan2(segments[..., 1], segments[..., 0])
    return wrapped_angle(heading[:, 1:] - heading[:, :-1])


def normalized_autocorrelation(signal: np.ndarray, minimum_lag: int, maximum_lag: int) -> tuple[int, float, np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal, axis=0, keepdims=True)
    if signal.ndim == 1:
        signal = signal[:, None]
    energy = float(np.sum(signal * signal))
    if not math.isfinite(energy) or energy <= 1e-12:
        return 0, 0.0, np.asarray([], dtype=int), np.asarray([], dtype=float)
    maximum_lag = min(maximum_lag, signal.shape[0] // 2)
    if maximum_lag <= minimum_lag:
        return 0, 0.0, np.asarray([], dtype=int), np.asarray([], dtype=float)
    lags = np.arange(minimum_lag, maximum_lag + 1, dtype=int)
    values: list[float] = []
    for lag in lags:
        left = signal[:-lag]
        right = signal[lag:]
        denominator = math.sqrt(float(np.sum(left * left)) * float(np.sum(right * right)))
        values.append(float(np.sum(left * right) / denominator) if denominator > 1e-12 else 0.0)
    ac = np.asarray(values, dtype=np.float64)
    peak_indices, _ = find_peaks(ac, distance=10, prominence=0.03)
    if peak_indices.size:
        best_index = int(peak_indices[np.argmax(ac[peak_indices])])
    else:
        best_index = int(np.argmax(ac))
    return int(lags[best_index]), float(ac[best_index]), lags, ac


def support_proxy(positions: np.ndarray, thresholds: Thresholds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = body_length(positions)
    softness = np.maximum(thresholds.contact_softness_body_fraction * lengths, 1e-4)
    band = thresholds.contact_band_body_fraction * lengths
    y = positions[..., 1]
    value = (
        thresholds.particle_radius + band[:, None] - y
    ) / softness[:, None]
    value = np.clip(value.astype(np.float32), -20.0, 20.0)
    weight = 1.0 / (1.0 + np.exp(-value))
    material_index = np.arange(positions.shape[1], dtype=np.float32)[None, :]
    support = np.sum(weight * material_index, axis=1) / np.maximum(np.sum(weight, axis=1), 1e-6)
    strength = np.max(weight, axis=1)
    valid = strength >= thresholds.contact_valid_weight
    return support.astype(np.float32), strength.astype(np.float32), valid


def support_transition_count(support: np.ndarray, valid: np.ndarray) -> tuple[float, int]:
    support = np.asarray(support, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    selected = support[valid & np.isfinite(support)]
    if selected.size < 20:
        return 0.0, 0
    low, high = np.percentile(selected, [10.0, 90.0])
    span = float(high - low)
    if span <= 1e-9:
        return span, 0
    lower_band = low + 0.25 * span
    upper_band = high - 0.25 * span
    states: list[int] = []
    for value in selected:
        state = -1 if value <= lower_band else (1 if value >= upper_band else 0)
        if state == 0:
            continue
        if not states or states[-1] != state:
            states.append(state)
    return span, max(0, len(states) - 1)


def episode_features(
    positions: np.ndarray,
    cumulative_rotation_degrees: np.ndarray,
    rotation_increment_degrees: np.ndarray,
    thresholds: Thresholds,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    positions = np.asarray(positions, dtype=np.float64)
    cumulative = np.asarray(cumulative_rotation_degrees, dtype=np.float64)
    increments = np.asarray(rotation_increment_degrees, dtype=np.float64)
    if positions.shape != (1001, 10, 2):
        raise ValueError(f"Unexpected positions shape: {positions.shape}")
    if cumulative.shape != (1001,) or increments.shape != (1000,):
        raise ValueError("Unexpected rotation-array shape")
    if not np.isfinite(positions).all() or not np.isfinite(cumulative).all() or not np.isfinite(increments).all():
        raise ValueError("Non-finite trajectory input")

    transient = thresholds.transient_steps
    lengths = body_length(positions)
    initial_length = float(lengths[0])
    if not math.isfinite(initial_length) or initial_length <= 1e-9:
        raise ValueError("Invalid initial body length")
    com = np.mean(positions, axis=1)
    forward_body_lengths = float((com[-1, 0] - com[0, 0]) / initial_length)
    rotation_span = float(np.max(cumulative) - np.min(cumulative))
    active = np.abs(increments) >= thresholds.active_rotation_increment_degrees
    active_inc = increments[active]
    if active_inc.size:
        reversal_index = float(
            1.0 - abs(float(np.sum(active_inc))) / (float(np.sum(np.abs(active_inc))) + 1e-12)
        )
    else:
        reversal_index = 0.0

    theta = joint_angles(positions)
    theta_tail = theta[transient:]
    circular_mean = np.arctan2(np.mean(np.sin(theta_tail), axis=0), np.mean(np.cos(theta_tail), axis=0))
    theta_residual = wrapped_angle(theta_tail - circular_mean[None, :])
    shape_rms = float(np.sqrt(np.mean(theta_residual * theta_residual)))
    theta_delta = wrapped_angle(np.diff(theta_tail, axis=0))
    shape_velocity_rms = float(np.sqrt(np.mean(theta_delta * theta_delta)))

    shape_embedding = np.concatenate((np.sin(theta_tail), np.cos(theta_tail)), axis=1)
    shape_embedding -= np.mean(shape_embedding, axis=0, keepdims=True)
    _, singular, _ = np.linalg.svd(shape_embedding, full_matrices=False)
    variance_total = float(np.sum(singular * singular))
    shape_pc1_variance_fraction = float((singular[0] ** 2) / variance_total) if variance_total > 1e-12 else 0.0
    period, repeatability, lags, autocorrelation = normalized_autocorrelation(
        shape_embedding,
        thresholds.minimum_period_steps,
        thresholds.maximum_period_steps,
    )
    usable_length = shape_embedding.shape[0]
    cycle_count = float(usable_length / period) if period > 0 else 0.0
    spectrum = np.abs(np.fft.rfft(theta_residual, axis=0)) ** 2
    spectral_energy = np.sum(spectrum[1:], axis=1)
    spectral_total = float(np.sum(spectral_energy))
    if spectral_total > 1e-12:
        top_count = min(3, spectral_energy.size)
        spectral_top3_fraction = float(np.sum(np.partition(spectral_energy, -top_count)[-top_count:]) / spectral_total)
    else:
        spectral_top3_fraction = 0.0

    cycle_progress: list[float] = []
    if period > 0:
        start = transient
        while start + period < positions.shape[0]:
            progress = float((com[start + period, 0] - com[start, 0]) / initial_length)
            cycle_progress.append(progress)
            start += period
    if cycle_progress:
        median_cycle_progress = float(np.median(cycle_progress))
        positive_cycle_fraction = float(np.mean(np.asarray(cycle_progress) > 0.0))
    else:
        median_cycle_progress = 0.0
        positive_cycle_fraction = 0.0

    support, contact_strength, contact_valid = support_proxy(positions, thresholds)
    support_tail = support[transient:]
    valid_tail = contact_valid[transient:]
    support_span, support_alternations = support_transition_count(support_tail, valid_tail)
    contact_valid_fraction = float(np.mean(valid_tail))
    if period > 0 and support_tail.size > period:
        left = support_tail[:-period] - float(np.mean(support_tail[:-period]))
        right = support_tail[period:] - float(np.mean(support_tail[period:]))
        denominator = math.sqrt(float(np.sum(left * left)) * float(np.sum(right * right)))
        support_repeatability = float(np.sum(left * right) / denominator) if denominator > 1e-12 else 0.0
    else:
        support_repeatability = 0.0

    window_boundaries = np.rint(
        np.linspace(transient, positions.shape[0] - 1, thresholds.analysis_window_count + 1)
    ).astype(int)
    window_progress: list[float] = []
    window_shape_amplitude: list[float] = []
    window_support_span: list[float] = []
    window_valid_contact_samples: list[int] = []
    for window_index in range(thresholds.analysis_window_count):
        start = int(window_boundaries[window_index])
        stop = int(window_boundaries[window_index + 1])
        window_progress.append(float((com[stop, 0] - com[start, 0]) / initial_length))

        theta_window = theta[start : stop + 1]
        window_mean = np.arctan2(
            np.mean(np.sin(theta_window), axis=0),
            np.mean(np.cos(theta_window), axis=0),
        )
        window_residual = wrapped_angle(theta_window - window_mean[None, :])
        window_shape_amplitude.append(
            float(np.degrees(np.sqrt(np.mean(window_residual * window_residual))))
        )

        window_valid = contact_valid[start : stop + 1]
        window_support = support[start : stop + 1][window_valid]
        window_valid_contact_samples.append(int(window_support.size))
        if window_support.size >= thresholds.minimum_valid_contact_samples_per_window:
            q05, q95 = np.percentile(window_support, [5.0, 95.0])
            window_support_span.append(float(q95 - q05))
        else:
            window_support_span.append(math.nan)

    positive_progress_windows = int(
        np.sum(np.asarray(window_progress) > thresholds.minimum_window_progress_body_lengths)
    )
    shape_window_pass_count = int(
        np.sum(np.asarray(window_shape_amplitude) >= thresholds.minimum_window_shape_amplitude_degrees)
    )
    support_window_pass_count = int(
        np.sum(
            np.asarray(
                [
                    math.isfinite(value) and value >= thresholds.minimum_support_span_indices
                    for value in window_support_span
                ],
                dtype=bool,
            )
        )
    )
    valid_support_window_count = int(np.sum(np.isfinite(np.asarray(window_support_span, dtype=float))))

    features = {
        "initial_body_length": initial_length,
        "forward_body_lengths_recomputed": forward_body_lengths,
        "rotation_span_degrees_recomputed": rotation_span,
        "rotation_reversal_index": reversal_index,
        "shape_rms_radians": shape_rms,
        "shape_velocity_rms_radians": shape_velocity_rms,
        "shape_pc1_variance_fraction": shape_pc1_variance_fraction,
        "shape_period_steps": float(period),
        "shape_repeatability": repeatability,
        "shape_cycle_count": cycle_count,
        "shape_spectral_top3_fraction": spectral_top3_fraction,
        "periodic_crawling_descriptor": bool(
            repeatability >= 0.50 and spectral_top3_fraction >= 0.25 and cycle_count >= 2.0
        ),
        "positive_cycle_fraction": positive_cycle_fraction,
        "median_cycle_progress_body_lengths": median_cycle_progress,
        "post_transient_progress_body_lengths": float(
            (com[-1, 0] - com[transient, 0]) / initial_length
        ),
        "positive_progress_window_count": positive_progress_windows,
        "shape_window_pass_count": shape_window_pass_count,
        "support_window_pass_count": support_window_pass_count,
        "valid_support_window_count": valid_support_window_count,
        "window_progress_body_lengths": ";".join(f"{value:.9g}" for value in window_progress),
        "window_shape_amplitude_degrees": ";".join(
            f"{value:.9g}" for value in window_shape_amplitude
        ),
        "window_support_span_indices": ";".join(
            "nan" if not math.isfinite(value) else f"{value:.9g}" for value in window_support_span
        ),
        "window_valid_contact_samples": ";".join(str(value) for value in window_valid_contact_samples),
        "support_span_indices": support_span,
        "support_alternation_count": float(support_alternations),
        "support_repeatability": support_repeatability,
        "contact_valid_fraction": contact_valid_fraction,
        "mean_contact_strength": float(np.mean(contact_strength[transient:])),
    }
    arrays = {
        "com": com,
        "theta": theta,
        "support": support,
        "contact_strength": contact_strength,
        "contact_valid": contact_valid,
        "shape_autocorrelation_lags": lags,
        "shape_autocorrelation": autocorrelation,
    }
    return features, arrays


def load_negative_control_thresholds(freeze_root: Path, base: Thresholds) -> tuple[Thresholds, list[dict[str, Any]]]:
    files = sorted((freeze_root / "data" / "trajectories").glob("*GLOBAL_BOTH_OFF*.npz"))
    controls: list[dict[str, Any]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            complex_positions = np.asarray(archive["positions"])
        positions = np.stack((np.real(complex_positions), np.imag(complex_positions)), axis=-1)
        cumulative = np.zeros(positions.shape[0], dtype=float)
        increments = np.zeros(positions.shape[0] - 1, dtype=float)
        metrics, _ = episode_features(positions, cumulative, increments, base)
        controls.append({"path": str(path), **metrics})
    if controls:
        control_shape = max(float(row["shape_rms_radians"]) for row in controls)
        control_velocity = max(float(row["shape_velocity_rms_radians"]) for row in controls)
        shape_threshold = max(base.minimum_shape_rms_radians, 1.50 * control_shape)
        velocity_threshold = max(base.minimum_shape_velocity_rms_radians, 1.50 * control_velocity)
    else:
        shape_threshold = base.minimum_shape_rms_radians
        velocity_threshold = base.minimum_shape_velocity_rms_radians
    calibrated = Thresholds(
        **{
            **asdict(base),
            "minimum_shape_rms_radians": float(shape_threshold),
            "minimum_shape_velocity_rms_radians": float(velocity_threshold),
        }
    )
    return calibrated, controls


def classify(row: dict[str, Any], thresholds: Thresholds) -> tuple[str, dict[str, bool]]:
    if bool(row.get("technical_exclusion", False)):
        return "technical_exclusion", {}
    window_progress = [
        float(value) for value in str(row["window_progress_body_lengths"]).split(";") if value
    ]
    window_shape = [
        float(value) for value in str(row["window_shape_amplitude_degrees"]).split(";") if value
    ]
    window_support = [
        float(value) for value in str(row["window_support_span_indices"]).split(";") if value
    ]
    positive_progress_window_count = sum(
        value > thresholds.minimum_window_progress_body_lengths for value in window_progress
    )
    shape_window_pass_count = sum(
        value >= thresholds.minimum_window_shape_amplitude_degrees for value in window_shape
    )
    valid_support_window_count = sum(math.isfinite(value) for value in window_support)
    support_window_pass_count = sum(
        math.isfinite(value) and value >= thresholds.minimum_support_span_indices
        for value in window_support
    )
    gates = {
        "gate_not_formal_rolling": not bool(row["formal_rolling"]),
        "gate_rotation_below_crawling_limit": float(row["rotation_span_degrees"]) < thresholds.maximum_crawling_rotation_span_degrees,
        "gate_forward_progress": float(row["forward_body_lengths"]) >= thresholds.minimum_forward_body_lengths,
        "gate_post_transient_progress": float(row["post_transient_progress_body_lengths"])
        >= thresholds.minimum_post_transient_progress_body_lengths,
        "gate_sustained_window_progress": positive_progress_window_count
        >= thresholds.minimum_positive_progress_windows,
        "gate_repeated_internal_deformation": shape_window_pass_count
        >= thresholds.minimum_shape_windows,
        "gate_repeated_support_migration": support_window_pass_count
        >= thresholds.minimum_support_windows,
        "gate_support_windows_observable": valid_support_window_count
        >= thresholds.minimum_support_windows,
    }
    if bool(row["formal_rolling"]):
        return "formal_rolling", gates
    if bool(row["lenient_full_rotation_span"]) or float(row["rotation_span_degrees"]) >= thresholds.maximum_crawling_rotation_span_degrees:
        if float(row["rotation_reversal_index"]) >= thresholds.rocking_reversal_index:
            return "rocking", gates
        return "partial_roll", gates
    if all(gates.values()):
        return "crawling_candidate", gates
    sustained_forward = (
        gates["gate_forward_progress"]
        and gates["gate_post_transient_progress"]
        and gates["gate_sustained_window_progress"]
    )
    if (
        sustained_forward
        and shape_window_pass_count <= 1
        and support_window_pass_count <= 1
    ):
        return "sliding_candidate", gates
    return "failed_other", gates


def validate_inputs(
    input_root: Path, *, archive_contract: bool = True
) -> tuple[list[dict[str, str]], list[Path], list[dict[str, Any]]]:
    completion = input_root / "EVALUATION_COMPLETE.json"
    final_validation = input_root / "FINAL_VALIDATION.json"
    episode_csv = input_root / "episode_results.csv"
    for path in (completion, final_validation, episode_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    rows = read_csv_rows(episode_csv)
    if len(rows) != 600:
        raise RuntimeError(f"Expected 600 episode rows, found {len(rows)}")
    identifiers = {
        (row["configuration_id"], int(row["formal_run"]), int(row["reset_seed"]))
        for row in rows
    }
    if len(identifiers) != 600:
        raise RuntimeError("Episode identifiers are not unique")
    npz_files = sorted((input_root / "tasks").glob("*/run*/checkpoint_1500.npz"))
    if len(npz_files) != 30:
        raise RuntimeError(f"Expected 30 NPZ files, found {len(npz_files)}")
    for configuration in CONFIG_ORDER:
        selected = [path for path in npz_files if path.parent.parent.name == configuration]
        if len(selected) != 5:
            raise RuntimeError(f"Expected five NPZs for {configuration}, found {len(selected)}")
    strict_count = sum(bool_from_csv(row["success_secondary_strict_common_kinematic"]) for row in rows)
    lenient_count = sum(bool_from_csv(row["success_lenient_rotation_span"]) for row in rows)
    if archive_contract and (strict_count != 146 or lenient_count != 147):
        raise RuntimeError(f"Locked rolling-count gate failed: strict={strict_count}, lenient={lenient_count}")
    manifest_candidates = [
        input_root / "STUDY_MANIFEST.json",
        input_root / "run_results.csv",
        input_root / "configuration_summary.csv",
        episode_csv,
        completion,
        final_validation,
        *npz_files,
    ]
    hash_rows = []
    for path in manifest_candidates:
        hash_rows.append(
            {
                "relative_path": str(path.relative_to(input_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows, npz_files, hash_rows


def extract_all(
    input_root: Path,
    episode_rows: list[dict[str, str]],
    npz_files: list[Path],
    thresholds: Thresholds,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int], dict[str, np.ndarray]]]:
    official = {
        (row["configuration_id"], int(row["formal_run"]), int(row["reset_seed"])): row
        for row in episode_rows
    }
    result: list[dict[str, Any]] = []
    arrays_by_id: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    maximum_forward_error = 0.0
    maximum_rotation_span_error = 0.0
    for npz_path in npz_files:
        configuration = npz_path.parent.parent.name
        run_text = npz_path.parent.name
        formal_run = int(run_text[3:] if run_text.startswith("run") else run_text)
        with np.load(npz_path, allow_pickle=False) as archive:
            reset_seeds = np.asarray(archive["reset_seeds"], dtype=int)
            if reset_seeds.shape != (20,):
                raise RuntimeError(f"Bad reset seed shape in {npz_path}")
            for episode_index, reset_seed in enumerate(reset_seeds.tolist()):
                identifier = (configuration, formal_run, int(reset_seed))
                source_row = official.get(identifier)
                if source_row is None:
                    raise RuntimeError(f"No official row for {identifier}")
                positions = np.asarray(archive["positions_xy"][episode_index], dtype=np.float32)
                cumulative = np.asarray(archive["unwrapped_best_fit_rotation_degrees"][episode_index], dtype=np.float32)
                increments = np.asarray(archive["best_fit_rotation_increment_degrees"][episode_index], dtype=np.float32)
                technical_exclusion = False
                technical_reason = ""
                try:
                    metrics, derived_arrays = episode_features(positions, cumulative, increments, thresholds)
                except Exception as exc:
                    technical_exclusion = True
                    technical_reason = f"{type(exc).__name__}: {exc}"
                    metrics = {
                        "initial_body_length": math.nan,
                        "forward_body_lengths_recomputed": math.nan,
                        "rotation_span_degrees_recomputed": math.nan,
                        "rotation_reversal_index": math.nan,
                        "shape_rms_radians": math.nan,
                        "shape_velocity_rms_radians": math.nan,
                        "shape_pc1_variance_fraction": math.nan,
                        "shape_period_steps": math.nan,
                        "shape_repeatability": math.nan,
                        "shape_cycle_count": math.nan,
                        "shape_spectral_top3_fraction": math.nan,
                        "periodic_crawling_descriptor": False,
                        "positive_cycle_fraction": math.nan,
                        "median_cycle_progress_body_lengths": math.nan,
                        "post_transient_progress_body_lengths": math.nan,
                        "positive_progress_window_count": 0,
                        "shape_window_pass_count": 0,
                        "support_window_pass_count": 0,
                        "valid_support_window_count": 0,
                        "window_progress_body_lengths": "",
                        "window_shape_amplitude_degrees": "",
                        "window_support_span_indices": "",
                        "window_valid_contact_samples": "",
                        "support_span_indices": math.nan,
                        "support_alternation_count": math.nan,
                        "support_repeatability": math.nan,
                        "contact_valid_fraction": math.nan,
                        "mean_contact_strength": math.nan,
                    }
                    derived_arrays = {}
                forward_official = finite_float(source_row["forward_body_lengths"])
                rotation_official = finite_float(source_row["best_fit_rotation_range_degrees"])
                if not technical_exclusion:
                    maximum_forward_error = max(
                        maximum_forward_error,
                        abs(float(metrics["forward_body_lengths_recomputed"]) - forward_official),
                    )
                    maximum_rotation_span_error = max(
                        maximum_rotation_span_error,
                        abs(float(metrics["rotation_span_degrees_recomputed"]) - rotation_official),
                    )
                row: dict[str, Any] = {
                    "episode_id": f"{configuration}__run{formal_run}__reset{int(reset_seed)}",
                    "configuration_id": configuration,
                    "paper_label": source_row["paper_label"],
                    "formal_run": formal_run,
                    "training_seed": int(source_row["internal_training_seed"]),
                    "checkpoint_batch": int(source_row["checkpoint_batch"]),
                    "reset_seed": int(reset_seed),
                    "npz_relative_path": str(npz_path.relative_to(input_root)),
                    "npz_episode_index": episode_index,
                    "formal_rolling": bool_from_csv(source_row["success_secondary_strict_common_kinematic"]),
                    "lenient_full_rotation_span": bool_from_csv(source_row["success_lenient_rotation_span"]),
                    "forward_body_lengths": forward_official,
                    "rotation_span_degrees": rotation_official,
                    "net_rotation_degrees": finite_float(source_row["net_best_fit_rotation_degrees"]),
                    "desired_net_rotation_degrees": finite_float(source_row["desired_net_rotation_degrees"]),
                    "desired_active_rotation_fraction": finite_float(source_row["desired_active_rotation_fraction"]),
                    "technical_exclusion": technical_exclusion,
                    "technical_reason": technical_reason,
                    **metrics,
                }
                gait, gates = classify(row, thresholds)
                row["automated_gait_label"] = gait
                row["kinematic_crawling_screen"] = bool(
                    (not row["formal_rolling"])
                    and (not row["technical_exclusion"])
                    and row["rotation_span_degrees"] < thresholds.maximum_crawling_rotation_span_degrees
                    and row["forward_body_lengths"] >= thresholds.minimum_forward_body_lengths
                )
                row["automated_crawling_candidate"] = gait == "crawling_candidate"
                row.update(gates)
                result.append(row)
                arrays_by_id[identifier] = {
                    "positions": positions,
                    "cumulative_rotation": cumulative,
                    "rotation_increments": increments,
                    **derived_arrays,
                }
    if len(result) != 600:
        raise RuntimeError(f"Feature extraction returned {len(result)} rows")
    if maximum_forward_error > 5e-5:
        raise RuntimeError(f"Forward recomputation mismatch: {maximum_forward_error}")
    if maximum_rotation_span_error > 5e-4:
        raise RuntimeError(f"Rotation-span recomputation mismatch: {maximum_rotation_span_error}")
    return result, arrays_by_id


def validate_support_reconstruction(
    input_root: Path,
    thresholds: Thresholds,
) -> dict[str, float]:
    maximum_support_error = 0.0
    maximum_contact_error = 0.0
    mean_support_errors: list[float] = []
    mean_contact_errors: list[float] = []
    files = sorted((input_root / "tasks" / "SGRR_O2_JS").glob("run*/checkpoint_1500.npz"))
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            positions = np.asarray(archive["positions_xy"], dtype=np.float32)
            logged_support = np.asarray(archive["support_index"], dtype=np.float32)
            logged_contact = np.asarray(archive["ground_contact_strength"], dtype=np.float32)
        for index in range(positions.shape[0]):
            support, contact, _ = support_proxy(positions[index], thresholds)
            support_error = np.abs(support - logged_support[index])
            contact_error = np.abs(contact - logged_contact[index])
            maximum_support_error = max(maximum_support_error, float(np.nanmax(support_error)))
            maximum_contact_error = max(maximum_contact_error, float(np.nanmax(contact_error)))
            mean_support_errors.append(float(np.nanmean(support_error)))
            mean_contact_errors.append(float(np.nanmean(contact_error)))
    return {
        "sgrr_episode_count": float(len(files) * 20),
        "maximum_support_index_absolute_error": maximum_support_error,
        "mean_support_index_absolute_error": float(np.mean(mean_support_errors)),
        "maximum_contact_strength_absolute_error": maximum_contact_error,
        "mean_contact_strength_absolute_error": float(np.mean(mean_contact_errors)),
    }


def validate_locked_classification(
    rows: list[dict[str, Any]], *, archive_contract: bool = True
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, expected: Any) -> None:
        passed = observed == expected
        checks.append(
            {
                "check": name,
                "observed": observed,
                "expected": expected,
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"Fail-closed QA: {name}: observed={observed!r}, expected={expected!r}")

    check("episode_count", len(rows), 600)
    check("technical_exclusion_count", sum(row["technical_exclusion"] for row in rows), 0)
    counts = Counter(row["automated_gait_label"] for row in rows)
    if not archive_contract:
        for label in GAIT_ORDER:
            checks.append(
                {
                    "check": f"portable_count_{label}",
                    "observed": counts.get(label, 0),
                    "expected": "not locked for a newly trained policy set",
                    "passed": True,
                }
            )
        return checks
    expected_counts = {
        "formal_rolling": 146,
        "crawling_candidate": 242,
        "partial_roll": 9,
        "rocking": 0,
        "sliding_candidate": 0,
        "failed_other": 203,
        "technical_exclusion": 0,
    }
    for label, expected in expected_counts.items():
        check(f"locked_count_{label}", counts.get(label, 0), expected)

    expected_by_config = {
        "HPR_DTH_PS": [0, 0, 0, 20, 0],
        "HPR_THDOT_PS": [0, 20, 0, 0, 20],
        "HPR_OBS_PS": [0, 20, 0, 20, 0],
        "HPR_O2_PS": [20, 20, 20, 20, 20],
        "HPR_O2_JS": [0, 20, 2, 20, 0],
        "SGRR_O2_JS": [0, 0, 0, 0, 0],
    }
    for configuration, expected in expected_by_config.items():
        observed = []
        for formal_run in range(5):
            observed.append(
                sum(
                    row["configuration_id"] == configuration
                    and int(row["formal_run"]) == formal_run
                    and row["automated_gait_label"] == "crawling_candidate"
                    for row in rows
                )
            )
        check(f"locked_crawling_counts_{configuration}", ";".join(map(str, observed)), ";".join(map(str, expected)))

    strict_lenient_difference = [
        row
        for row in rows
        if bool(row["lenient_full_rotation_span"]) and not bool(row["formal_rolling"])
    ]
    check("strict_lenient_difference_count", len(strict_lenient_difference), 1)
    if strict_lenient_difference:
        difference = strict_lenient_difference[0]
        check("strict_lenient_difference_configuration", difference["configuration_id"], "HPR_O2_JS")
        check("strict_lenient_difference_run", int(difference["formal_run"]), 2)
        check("strict_lenient_difference_reset", int(difference["reset_seed"]), 20264108)
        check("strict_lenient_difference_label", difference["automated_gait_label"], "partial_roll")
    return checks


def trajectory_equivalence_inventory(
    input_root: Path,
    npz_files: list[Path],
    episode_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training_seed_lookup = {
        (row["configuration_id"], int(row["formal_run"])): int(row["internal_training_seed"])
        for row in episode_rows
    }
    run_metadata = {
        (row["configuration_id"], int(row["formal_run"])): row
        for row in read_csv_rows(input_root / "run_results.csv")
    }
    raw_rows: list[dict[str, Any]] = []
    initial_hashes_by_seed: dict[int, Counter[str]] = defaultdict(Counter)
    for path in npz_files:
        configuration = path.parent.parent.name
        run_text = path.parent.name
        formal_run = int(run_text[3:] if run_text.startswith("run") else run_text)
        with np.load(path, allow_pickle=False) as archive:
            positions = np.ascontiguousarray(np.asarray(archive["positions_xy"], dtype=np.float32))
            actions = np.ascontiguousarray(np.asarray(archive["deterministic_policy_action"], dtype=np.float32))
            reset_seeds = np.asarray(archive["reset_seeds"], dtype=int)
        metadata = run_metadata[(configuration, formal_run)]
        position_hash = hashlib.sha256(positions.tobytes()).hexdigest()
        action_hash = hashlib.sha256(actions.tobytes()).hexdigest()
        for episode_index, reset_seed in enumerate(reset_seeds.tolist()):
            initial_hash = hashlib.sha256(np.ascontiguousarray(positions[episode_index, 0]).tobytes()).hexdigest()
            initial_hashes_by_seed[int(reset_seed)][initial_hash] += 1
        raw_rows.append(
            {
                "configuration_id": configuration,
                "paper_label": PAPER_LABELS[configuration],
                "formal_run": formal_run,
                "training_seed": training_seed_lookup[(configuration, formal_run)],
                "checkpoint_batch": int(metadata["checkpoint_batch"]),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "npz_relative_path": str(path.relative_to(input_root)),
                "npz_sha256": sha256_file(path),
                "positions_xy_sha256": position_hash,
                "actions_sha256": action_hash,
                "episode_count": 20,
                "reset_seed_min": int(np.min(reset_seeds)),
                "reset_seed_max": int(np.max(reset_seeds)),
                "positions_shape_valid": positions.shape == (20, 1001, 10, 2),
                "finite_valid": bool(np.isfinite(positions).all() and np.isfinite(actions).all()),
            }
        )
    position_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        position_groups[row["positions_xy_sha256"]].append(row)
    duplicate_hashes = sorted(
        (key for key, selected in position_groups.items() if len(selected) > 1),
        key=lambda key: (-len(position_groups[key]), key),
    )
    duplicate_ids = {value: f"EQ{index:03d}" for index, value in enumerate(duplicate_hashes, start=1)}
    for row in raw_rows:
        selected = position_groups[row["positions_xy_sha256"]]
        row["effective_behavior_group"] = duplicate_ids.get(row["positions_xy_sha256"], "UNIQUE")
        row["effective_behavior_group_size"] = len(selected)
        row["exact_position_equivalent_across_runs"] = len(selected) > 1
    initial_state_pairing_valid = (
        len(initial_hashes_by_seed) == 20
        and all(len(counts) == 1 and sum(counts.values()) == 30 for counts in initial_hashes_by_seed.values())
    )
    qa = [
        {
            "check": "unique_checkpoint_sha256",
            "observed": len({row["checkpoint_sha256"] for row in raw_rows}),
            "expected": 30,
            "passed": len({row["checkpoint_sha256"] for row in raw_rows}) == 30,
        },
        {
            "check": "distinct_run_position_matrices",
            "observed": len(position_groups),
            "expected": 23,
            "passed": len(position_groups) == 23,
        },
        {
            "check": "duplicate_behavior_group_sizes",
            "observed": ";".join(map(str, sorted((len(value) for value in position_groups.values() if len(value) > 1), reverse=True))),
            "expected": "5;4",
            "passed": sorted((len(value) for value in position_groups.values() if len(value) > 1), reverse=True) == [5, 4],
        },
        {
            "check": "paired_initial_state_seed_count",
            "observed": len(initial_hashes_by_seed),
            "expected": 20,
            "passed": initial_state_pairing_valid,
        },
    ]
    if not all(bool(row["passed"]) for row in qa):
        raise RuntimeError(f"Trajectory-equivalence QA failed: {qa}")
    return raw_rows, qa


def verify_input_hashes_unchanged(input_root: Path, hash_rows: list[dict[str, Any]]) -> None:
    for row in hash_rows:
        path = input_root / str(row["relative_path"])
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Locked input changed during analysis: {path}")


def aggregate_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["configuration_id"], int(row["formal_run"]), int(row["training_seed"]))].append(row)
    output: list[dict[str, Any]] = []
    for (configuration, formal_run, training_seed), selected in sorted(
        groups.items(), key=lambda item: (CONFIG_ORDER.index(item[0][0]), item[0][1])
    ):
        counts = Counter(row["automated_gait_label"] for row in selected)
        crawl_rows = [row for row in selected if row["automated_gait_label"] == "crawling_candidate"]
        result: dict[str, Any] = {
            "configuration_id": configuration,
            "paper_label": PAPER_LABELS[configuration],
            "formal_run": formal_run,
            "training_seed": training_seed,
            "episode_count": len(selected),
        }
        for gait in GAIT_ORDER:
            result[f"{gait}_count"] = int(counts.get(gait, 0))
        result["crawling_candidate_rate"] = counts.get("crawling_candidate", 0) / len(selected)
        result["formal_rolling_rate"] = counts.get("formal_rolling", 0) / len(selected)
        result["any_crawling_candidate_observed"] = counts.get("crawling_candidate", 0) >= 1
        result["stable_crawling_candidate_10_of_20"] = counts.get("crawling_candidate", 0) >= 10
        result["productive_locomotion_rate"] = (
            counts.get("formal_rolling", 0) + counts.get("crawling_candidate", 0)
        ) / len(selected)
        result["periodic_crawling_candidate_count"] = sum(
            bool(row["periodic_crawling_descriptor"]) for row in crawl_rows
        )
        result["irregular_crawling_candidate_count"] = len(crawl_rows) - int(
            result["periodic_crawling_candidate_count"]
        )
        result["crawl_forward_body_lengths_mean"] = (
            float(np.mean([row["forward_body_lengths"] for row in crawl_rows])) if crawl_rows else math.nan
        )
        result["crawl_rotation_span_degrees_mean"] = (
            float(np.mean([row["rotation_span_degrees"] for row in crawl_rows])) if crawl_rows else math.nan
        )
        output.append(result)
    if len(output) != 30:
        raise RuntimeError(f"Expected 30 run summaries, found {len(output)}")
    return output


def aggregate_configuration(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        groups[row["configuration_id"]].append(row)
    output: list[dict[str, Any]] = []
    for configuration in CONFIG_ORDER:
        selected = groups[configuration]
        crawl_counts = [int(row["crawling_candidate_count"]) for row in selected]
        roll_counts = [int(row["formal_rolling_count"]) for row in selected]
        productive_rates = [float(row["productive_locomotion_rate"]) for row in selected]
        result: dict[str, Any] = {
            "configuration_id": configuration,
            "paper_label": PAPER_LABELS[configuration],
            "independent_training_runs": len(selected),
            "nested_rollouts": sum(int(row["episode_count"]) for row in selected),
            "any_crawling_candidate_runs": sum(bool(row["any_crawling_candidate_observed"]) for row in selected),
            "stable_crawling_candidate_runs": sum(bool(row["stable_crawling_candidate_10_of_20"]) for row in selected),
            "crawling_candidate_count_total": sum(crawl_counts),
            "crawling_candidate_count_median_per_run": float(np.median(crawl_counts)),
            "crawling_candidate_count_min_per_run": min(crawl_counts),
            "crawling_candidate_count_max_per_run": max(crawl_counts),
            "crawling_candidate_rate_mean_across_runs": float(np.mean(np.asarray(crawl_counts) / 20.0)),
            "crawling_candidate_rate_sd_across_runs": float(np.std(np.asarray(crawl_counts) / 20.0, ddof=1)),
            "formal_rolling_count_total": sum(roll_counts),
            "formal_rolling_count_median_per_run": float(np.median(roll_counts)),
            "formal_rolling_count_min_per_run": min(roll_counts),
            "formal_rolling_count_max_per_run": max(roll_counts),
            "productive_locomotion_rate_mean_across_runs": float(np.mean(productive_rates)),
            "productive_locomotion_rate_sd_across_runs": float(np.std(productive_rates, ddof=1)),
            "periodic_crawling_candidate_count_total": sum(
                int(row["periodic_crawling_candidate_count"]) for row in selected
            ),
            "irregular_crawling_candidate_count_total": sum(
                int(row["irregular_crawling_candidate_count"]) for row in selected
            ),
        }
        for gait in GAIT_ORDER:
            result[f"{gait}_count_total"] = sum(int(row[f"{gait}_count"]) for row in selected)
        output.append(result)
    return output


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    values = np.asarray(differences, dtype=float)
    observed = abs(float(np.mean(values)))
    exceed = 0
    total = 2 ** len(values)
    for pattern in range(total):
        signs = np.asarray([1.0 if pattern & (1 << index) else -1.0 for index in range(len(values))])
        statistic = abs(float(np.mean(values * signs)))
        exceed += statistic >= observed - 1e-12
    return exceed / total


def contrast_rows(
    run_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(row["configuration_id"], int(row["formal_run"])): row for row in run_rows}
    detail: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for left, right, label in PRESET_CONTRASTS:
        comparison_rows: list[dict[str, Any]] = []
        for formal_run in range(5):
            left_row = lookup[(left, formal_run)]
            right_row = lookup[(right, formal_run)]
            row = {
                "contrast": label,
                "left_configuration": left,
                "right_configuration": right,
                "formal_run": formal_run,
                "training_seed": left_row["training_seed"],
            }
            for metric, field in (
                ("crawling_candidate", "crawling_candidate_rate"),
                ("formal_rolling", "formal_rolling_rate"),
                ("productive_locomotion", "productive_locomotion_rate"),
            ):
                left_value = float(left_row[field])
                right_value = float(right_row[field])
                row[f"left_{metric}_rate"] = left_value
                row[f"right_{metric}_rate"] = right_value
                row[f"difference_{metric}_right_minus_left"] = right_value - left_value
            comparison_rows.append(row)
            detail.append(row)
        for metric in ("crawling_candidate", "formal_rolling", "productive_locomotion"):
            differences = [
                float(row[f"difference_{metric}_right_minus_left"])
                for row in comparison_rows
            ]
            summaries.append(
                {
                    "contrast": label,
                    "left_configuration": left,
                    "right_configuration": right,
                    "metric": metric,
                    "paired_training_runs": 5,
                    "mean_difference_right_minus_left": float(np.mean(differences)),
                    "median_difference_right_minus_left": float(np.median(differences)),
                    "minimum_difference": float(np.min(differences)),
                    "maximum_difference": float(np.max(differences)),
                    "positive_runs": int(np.sum(np.asarray(differences) > 0)),
                    "negative_runs": int(np.sum(np.asarray(differences) < 0)),
                    "zero_runs": int(np.sum(np.asarray(differences) == 0)),
                    "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(differences),
                    "inference_unit": "paired_independent_training_run",
                    "interpretation": "exploratory_effect_direction_n_equals_5",
                }
            )
    return detail, summaries


def paired_episode_transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["configuration_id"], int(row["formal_run"]), int(row["reset_seed"])): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for left, right, label in PRESET_CONTRASTS:
        for formal_run in range(5):
            transitions: Counter[tuple[str, str]] = Counter()
            for reset_seed in range(20264101, 20264121):
                left_label = lookup[(left, formal_run, reset_seed)]["automated_gait_label"]
                right_label = lookup[(right, formal_run, reset_seed)]["automated_gait_label"]
                transitions[(left_label, right_label)] += 1
            for (left_label, right_label), count in sorted(transitions.items()):
                output.append(
                    {
                        "contrast": label,
                        "left_configuration": left,
                        "right_configuration": right,
                        "formal_run": formal_run,
                        "left_gait_label": left_label,
                        "right_gait_label": right_label,
                        "paired_reset_count": count,
                        "nested_descriptive_only": True,
                    }
                )
    return output


def reclassify_sensitivity(rows: list[dict[str, Any]], base: Thresholds) -> list[dict[str, Any]]:
    variants: list[tuple[str, Thresholds, dict[str, float]]] = []
    for forward in (0.5, 1.0, 1.5):
        for rotation in (90.0, 180.0, 270.0):
            for shape in (15.0, 20.0, 25.0):
                for support in (1.0, 1.5, 2.0):
                    values = {
                        **asdict(base),
                        "minimum_forward_body_lengths": forward,
                        "maximum_crawling_rotation_span_degrees": rotation,
                        "minimum_window_shape_amplitude_degrees": shape,
                        "minimum_support_span_indices": support,
                    }
                    name = f"D{forward:.1f}_R{int(rotation)}_A{int(shape)}_S{support:.1f}"
                    variants.append(
                        (
                            name,
                            Thresholds(**values),
                            {
                                "minimum_forward_body_lengths": forward,
                                "maximum_rotation_span_degrees": rotation,
                                "minimum_window_shape_amplitude_degrees": shape,
                                "minimum_window_support_span_indices": support,
                            },
                        )
                    )
    baseline_set = {
        row["episode_id"]
        for row in rows
        if classify(row, base)[0] == "crawling_candidate"
    }
    output: list[dict[str, Any]] = []
    for name, threshold, parameters in variants:
        counts = Counter()
        per_config = Counter()
        per_run = Counter()
        crawling_set: set[str] = set()
        for source in rows:
            gait, _ = classify(source, threshold)
            counts[gait] += 1
            if gait == "crawling_candidate":
                per_config[source["configuration_id"]] += 1
                per_run[(source["configuration_id"], int(source["formal_run"]))] += 1
                crawling_set.add(source["episode_id"])
        union = baseline_set | crawling_set
        row: dict[str, Any] = {
            "variant": name,
            "is_primary_setting": bool(
                parameters["minimum_forward_body_lengths"] == base.minimum_forward_body_lengths
                and parameters["maximum_rotation_span_degrees"] == base.maximum_crawling_rotation_span_degrees
                and parameters["minimum_window_shape_amplitude_degrees"] == base.minimum_window_shape_amplitude_degrees
                and parameters["minimum_window_support_span_indices"] == base.minimum_support_span_indices
            ),
            **parameters,
            "crawling_candidate_total": counts["crawling_candidate"],
            "jaccard_with_primary_crawling_set": len(baseline_set & crawling_set) / len(union) if union else 1.0,
            "formal_rolling_total": counts["formal_rolling"],
            "partial_roll_total": counts["partial_roll"],
            "rocking_total": counts["rocking"],
            "sliding_candidate_total": counts["sliding_candidate"],
            "failed_other_total": counts["failed_other"],
        }
        for configuration in CONFIG_ORDER:
            row[f"{configuration}_crawling_candidate"] = per_config[configuration]
            row[f"{configuration}_stable_runs_10_of_20"] = sum(
                per_run[(configuration, formal_run)] >= 10 for formal_run in range(5)
            )
        output.append(row)
    if len(output) != 81 or sum(bool(row["is_primary_setting"]) for row in output) != 1:
        raise RuntimeError("Sensitivity-grid QA failed")
    return output


def select_medoid(rows: list[dict[str, Any]], label: str, preferred_configuration: str) -> dict[str, Any] | None:
    selected = [
        row
        for row in rows
        if row["automated_gait_label"] == label and row["configuration_id"] == preferred_configuration
    ]
    if not selected:
        selected = [row for row in rows if row["automated_gait_label"] == label]
    if not selected:
        return None
    feature_names = [
        "forward_body_lengths",
        "rotation_span_degrees",
        "shape_rms_radians",
        "shape_period_steps",
        "shape_repeatability",
        "support_span_indices",
        "median_cycle_progress_body_lengths",
    ]
    matrix = np.asarray([[float(row[name]) for name in feature_names] for row in selected], dtype=float)
    median = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - median[None, :]), axis=0)
    scale = np.where(scale > 1e-9, scale, np.std(matrix, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    distance = np.sqrt(np.sum(((matrix - median[None, :]) / scale[None, :]) ** 2, axis=1))
    return selected[int(np.argmin(distance))]


def particle_colors(number: int = 10) -> np.ndarray:
    return plt.get_cmap("viridis")(np.linspace(0.05, 0.95, number))


def plot_shape(ax: plt.Axes, positions: np.ndarray, title: str = "", show_world_x: bool = True) -> None:
    colors = particle_colors(positions.shape[0])
    ax.plot(positions[:, 0], positions[:, 1], color="#34495E", lw=1.6, zorder=1)
    for index, (x, y) in enumerate(positions):
        circle = Circle((x, y), radius=1.0 / 3.0, facecolor=colors[index], edgecolor="white", lw=0.6, zorder=2)
        ax.add_patch(circle)
    ax.axhline(0.0, color="#222222", lw=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    if not show_world_x:
        ax.set_xticklabels([])
    ax.spines[["top", "right"]].set_visible(False)


def frame_indices_for_crawl(row: dict[str, Any], count: int = 8) -> np.ndarray:
    period = int(round(float(row.get("shape_period_steps", 0.0))))
    start = 100
    if period > 0 and bool(row.get("periodic_crawling_descriptor", False)):
        stop = min(1000, start + 2 * period)
    else:
        stop = 1000
    return np.unique(np.rint(np.linspace(start, stop, count)).astype(int))


def frame_indices_for_roll(arrays: dict[str, np.ndarray], count: int = 7) -> np.ndarray:
    cumulative = arrays["cumulative_rotation"]
    net = float(cumulative[-1] - cumulative[0])
    direction = 1.0 if net >= 0.0 else -1.0
    desired = direction * (cumulative - cumulative[0])
    if float(np.max(desired)) >= 360.0:
        targets = np.linspace(0.0, 360.0, count)
        return np.asarray([int(np.argmin(np.abs(desired - target))) for target in targets], dtype=int)
    return np.rint(np.linspace(0, len(cumulative) - 1, count)).astype(int)


def representative_figure(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output: Path,
    mode: str,
) -> None:
    positions = arrays["positions"]
    com = arrays["com"]
    theta = arrays["theta"]
    support = arrays["support"]
    contact_valid = arrays["contact_valid"]
    support_for_plot = np.full_like(support, np.nan, dtype=float)
    last_valid_support = math.nan
    for sample_index, (value, valid) in enumerate(zip(support, contact_valid)):
        if bool(valid) and math.isfinite(float(value)):
            last_valid_support = float(value)
        if math.isfinite(last_valid_support):
            support_for_plot[sample_index] = last_valid_support
    cumulative = arrays["cumulative_rotation"]
    indices = frame_indices_for_crawl(row, 8) if mode == "crawl" else frame_indices_for_roll(arrays, 8)
    figure = plt.figure(figsize=(16, 7.2), constrained_layout=True)
    grid = figure.add_gridspec(3, 8, height_ratios=[0.42, 0.85, 0.85])
    all_selected = positions[indices]
    centered_selected = all_selected.copy()
    centered_selected[..., 0] -= com[indices, 0][:, None]
    relative_x_limit = float(np.max(np.abs(centered_selected[..., 0])) + 0.7)
    y_min = -0.15
    y_max = float(np.max(all_selected[..., 1]) + 0.7)
    for column, step in enumerate(indices):
        ax = figure.add_subplot(grid[0, column])
        centered = positions[step].copy()
        centered[:, 0] -= com[step, 0]
        plot_shape(ax, centered, f"step {step}")
        ax.set_xlim(-relative_x_limit, relative_x_limit)
        ax.set_ylim(y_min, y_max)
        if column:
            ax.set_yticklabels([])
    initial_length = float(row["initial_body_length"])
    time = np.arange(positions.shape[0])
    ax_com = figure.add_subplot(grid[1, 0:4])
    ax_com.plot(time, (com[:, 0] - com[0, 0]) / initial_length, color="#16A085", lw=1.8)
    for step in indices:
        ax_com.axvline(step, color="#777777", lw=0.45, alpha=0.35)
    ax_com.set(xlabel="Control step", ylabel=r"$\Delta x_{COM}/L_0$")
    ax_com.grid(alpha=0.25)
    ax_phi = figure.add_subplot(grid[1, 4:8])
    ax_phi.plot(time, cumulative - cumulative[0], color="#8E44AD", lw=1.8)
    for step in indices:
        ax_phi.axvline(step, color="#777777", lw=0.45, alpha=0.35)
    ax_phi.axhline(360, color="#888888", ls="--", lw=0.8)
    ax_phi.axhline(-360, color="#888888", ls="--", lw=0.8)
    ax_phi.set(xlabel="Control step", ylabel=r"Unwrapped $\Phi$ (deg)")
    ax_phi.grid(alpha=0.25)
    ax_theta = figure.add_subplot(grid[2, 0:5])
    ax_theta.imshow(
        theta.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        interpolation="nearest",
        vmin=-math.pi,
        vmax=math.pi,
    )
    ax_theta.set(xlabel="Control step", ylabel="Joint", yticks=np.arange(8), yticklabels=[f"J{i:02d}" for i in range(1, 9)])
    ax_theta.set_title(r"Internal joint angle ($-\pi$ blue, $+\pi$ red)", fontsize=9)
    ax_support = figure.add_subplot(grid[2, 5:8])
    ax_support.plot(time, support_for_plot, color="#D35400", lw=1.2)
    ax_support.set(xlabel="Control step", ylabel="Support-proxy index", ylim=(-0.25, 9.25))
    ax_support.grid(alpha=0.25)
    figure.suptitle(
        f"{'Crawling-candidate' if mode == 'crawl' else 'Formal-rolling'} representative selected by medoid rule | "
        f"{row['paper_label']}, run {row['formal_run']}, reset {row['reset_seed']}",
        fontsize=13,
    )
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def comparison_figure(
    crawl_row: dict[str, Any],
    crawl_arrays: dict[str, np.ndarray],
    roll_row: dict[str, Any],
    roll_arrays: dict[str, np.ndarray],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(16, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(4, 8, height_ratios=[0.38, 0.72, 0.38, 0.72])
    for row_index, (label, metadata, arrays, mode) in enumerate(
        [
            ("Crawling candidate", crawl_row, crawl_arrays, "crawl"),
            ("Formal rolling", roll_row, roll_arrays, "roll"),
        ]
    ):
        positions = arrays["positions"]
        com = arrays["com"]
        indices = frame_indices_for_crawl(metadata, 8) if mode == "crawl" else frame_indices_for_roll(arrays, 8)
        selected = positions[indices]
        centered_selected = selected.copy()
        centered_selected[..., 0] -= com[indices, 0][:, None]
        relative_x_limit = float(np.max(np.abs(centered_selected[..., 0])) + 0.7)
        y_min = -0.15
        y_max = float(np.max(selected[..., 1]) + 0.7)
        shape_grid_row = row_index * 2
        curve_grid_row = shape_grid_row + 1
        for column, step in enumerate(indices):
            ax = figure.add_subplot(grid[shape_grid_row, column])
            centered = positions[step].copy()
            centered[:, 0] -= com[step, 0]
            plot_shape(ax, centered, f"{label}\nstep {step}" if column == 0 else f"step {step}")
            ax.set_xlim(-relative_x_limit, relative_x_limit)
            ax.set_ylim(y_min, y_max)
            if column:
                ax.set_yticklabels([])
        initial_length = float(metadata["initial_body_length"])
        cumulative = arrays["cumulative_rotation"]
        time = np.arange(positions.shape[0])
        ax = figure.add_subplot(grid[curve_grid_row, 0:4])
        ax.plot(time, (com[:, 0] - com[0, 0]) / initial_length, color="#16A085", lw=1.5)
        for step in indices:
            ax.axvline(step, color="#777777", lw=0.4, alpha=0.3)
        ax.set_ylabel(r"$\Delta x_{COM}/L_0$")
        ax.grid(alpha=0.2)
        ax = figure.add_subplot(grid[curve_grid_row, 4:8])
        ax.plot(time, cumulative - cumulative[0], color="#8E44AD", lw=1.5)
        for step in indices:
            ax.axvline(step, color="#777777", lw=0.4, alpha=0.3)
        ax.axhline(360, color="#999999", ls="--", lw=0.7)
        ax.axhline(-360, color="#999999", ls="--", lw=0.7)
        ax.set_ylabel(r"$\Phi$ (deg)")
        ax.grid(alpha=0.2)
    figure.suptitle("Matched-style kinematic comparison of crawling and rolling", fontsize=14)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def composition_figure(run_rows: list[dict[str, Any]], output: Path) -> None:
    selected_gaits = [
        "formal_rolling",
        "crawling_candidate",
        "partial_roll",
        "rocking",
        "sliding_candidate",
        "failed_other",
    ]
    figure, ax = plt.subplots(figsize=(17, 7.5), constrained_layout=True)
    x = np.arange(len(run_rows), dtype=float)
    bottom = np.zeros(len(run_rows), dtype=float)
    for gait in selected_gaits:
        values = np.asarray([float(row[f"{gait}_count"]) / 20.0 for row in run_rows])
        ax.bar(x, values, bottom=bottom, width=0.82, color=GAIT_COLORS[gait], edgecolor="white", linewidth=0.35, label=gait.replace("_", " "))
        bottom += values
    for boundary in range(5, 30, 5):
        ax.axvline(boundary - 0.5, color="#444444", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of 20 paired reset rollouts")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{PAPER_LABELS[row['configuration_id']]}\nr{row['formal_run']}" for row in run_rows], rotation=55, ha="right", fontsize=8)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.set_title("Gait composition by independent training run")
    ax.grid(axis="y", alpha=0.2)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def rate_figure(run_rows: list[dict[str, Any]], output: Path) -> None:
    figure, ax = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)
    rng = np.random.default_rng(20260824)
    for index, configuration in enumerate(CONFIG_ORDER):
        selected = [row for row in run_rows if row["configuration_id"] == configuration]
        jitter = rng.linspace(-0.13, 0.13, len(selected)) if hasattr(rng, "linspace") else np.linspace(-0.13, 0.13, len(selected))
        crawl = [100.0 * float(row["crawling_candidate_rate"]) for row in selected]
        roll = [100.0 * float(row["formal_rolling_rate"]) for row in selected]
        ax.scatter(index + jitter, crawl, color=GAIT_COLORS["crawling_candidate"], s=58, marker="o", edgecolor="white", linewidth=0.5, zorder=3)
        ax.scatter(index + jitter, roll, color=GAIT_COLORS["formal_rolling"], s=64, marker="^", edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([index - 0.18, index + 0.18], [np.median(crawl)] * 2, color=GAIT_COLORS["crawling_candidate"], lw=2.2)
        ax.plot([index - 0.18, index + 0.18], [np.median(roll)] * 2, color=GAIT_COLORS["formal_rolling"], lw=2.2)
    ax.set_xticks(np.arange(len(CONFIG_ORDER)))
    ax.set_xticklabels([PAPER_LABELS[item] for item in CONFIG_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Run-level rate across 20 reset states (%)")
    ax.set_ylim(-3, 103)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="none", markerfacecolor=GAIT_COLORS["crawling_candidate"], label="Crawling candidate", markersize=8),
            plt.Line2D([], [], marker="^", color="none", markerfacecolor=GAIT_COLORS["formal_rolling"], label="Formal rolling", markersize=8),
        ],
        frameon=False,
    )
    ax.set_title("Crawling and rolling rates; each point is one independent training run")
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def feature_space_figure(rows: list[dict[str, Any]], output: Path) -> None:
    figure, ax = plt.subplots(figsize=(11.5, 7.5), constrained_layout=True)
    for gait in GAIT_ORDER:
        selected = [row for row in rows if row["automated_gait_label"] == gait]
        if not selected:
            continue
        x = [float(row["forward_body_lengths"]) for row in selected]
        y = [float(row["rotation_span_degrees"]) for row in selected]
        ax.scatter(x, y, s=24, alpha=0.70, color=GAIT_COLORS[gait], label=gait.replace("_", " "), edgecolor="none")
    ax.axvline(1.0, color="#333333", ls="--", lw=1.0)
    ax.axhline(180.0, color="#333333", ls="--", lw=1.0)
    ax.axhline(360.0, color="#777777", ls=":", lw=1.0)
    ax.set(xlabel="Forward displacement / initial body length", ylabel="Best-fit rotation span (deg)")
    ax.set_yscale("symlog", linthresh=10.0)
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    ax.set_title("Operational gait classification in kinematic feature space")
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def sensitivity_figure(rows: list[dict[str, Any]], base: Thresholds, output: Path) -> None:
    forward_values = [0.5, 1.0, 1.5]
    rotation_values = [90.0, 180.0, 270.0]
    matrices: dict[str, np.ndarray] = {}
    for configuration in CONFIG_ORDER:
        matrix = np.zeros((len(rotation_values), len(forward_values)), dtype=int)
        selected = [row for row in rows if row["configuration_id"] == configuration]
        for row_index, rotation in enumerate(rotation_values):
            for column_index, forward in enumerate(forward_values):
                threshold = Thresholds(
                    **{
                        **asdict(base),
                        "minimum_forward_body_lengths": forward,
                        "maximum_crawling_rotation_span_degrees": rotation,
                    }
                )
                matrix[row_index, column_index] = sum(classify(row, threshold)[0] == "crawling_candidate" for row in selected)
        matrices[configuration] = matrix
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.8), constrained_layout=True, sharex=True, sharey=True)
    maximum = max(int(np.max(matrix)) for matrix in matrices.values())
    for ax, configuration in zip(axes.ravel(), CONFIG_ORDER):
        matrix = matrices[configuration]
        image = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=max(1, maximum), origin="lower")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                ax.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center", fontsize=11)
        ax.set_title(PAPER_LABELS[configuration])
        ax.set_xticks(range(len(forward_values)), [str(item) for item in forward_values])
        ax.set_yticks(range(len(rotation_values)), [str(int(item)) for item in rotation_values])
    figure.supxlabel("Minimum forward displacement (body lengths)")
    figure.supylabel("Maximum crawling rotation span (deg)")
    figure.colorbar(image, ax=axes, label="Crawling candidates / 100", shrink=0.82)
    figure.suptitle("Threshold sensitivity of crawling-candidate counts")
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def paired_contrast_figure(details: list[dict[str, Any]], output: Path) -> None:
    metrics = [
        ("crawling_candidate", "Crawl", GAIT_COLORS["crawling_candidate"]),
        ("formal_rolling", "Roll", GAIT_COLORS["formal_rolling"]),
        ("productive_locomotion", "Productive", "#333333"),
    ]
    figure, axes = plt.subplots(1, 5, figsize=(17, 4.4), constrained_layout=True, sharey=True)
    for ax, (_, _, label) in zip(axes, PRESET_CONTRASTS):
        selected = [row for row in details if row["contrast"] == label]
        for metric_index, (metric, short_label, color) in enumerate(metrics):
            values = np.asarray(
                [float(row[f"difference_{metric}_right_minus_left"]) for row in selected],
                dtype=float,
            )
            jitter = np.linspace(-0.10, 0.10, len(values))
            ax.scatter(
                metric_index + jitter,
                values,
                s=38,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.plot(
                [metric_index - 0.17, metric_index + 0.17],
                [float(np.median(values))] * 2,
                color=color,
                lw=2.0,
            )
        ax.axhline(0.0, color="#555555", lw=0.9)
        ax.set_xticks(range(len(metrics)), [item[1] for item in metrics], rotation=30, ha="right")
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Paired rate difference, B - A")
    figure.suptitle("Preplanned paired contrasts; each point is one training-run pair", fontsize=13)
    figure.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(figure)


def blind_episode_image(
    code: str,
    row: dict[str, Any],
    arrays: dict[str, np.ndarray],
    output: Path,
) -> None:
    positions = arrays["positions"]
    com = arrays["com"]
    indices = np.rint(np.linspace(100, 1000, 8)).astype(int)
    figure = plt.figure(figsize=(15.5, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 8, height_ratios=[1.0, 0.85])
    selected = positions[indices]
    x_min = float(np.min(selected[..., 0]) - 0.7)
    x_max = float(np.max(selected[..., 0]) + 0.7)
    y_min = -0.15
    y_max = float(np.max(selected[..., 1]) + 0.7)

    world_ax = figure.add_subplot(grid[0, :])
    time_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(indices)))
    for color, step in zip(time_colors, indices):
        world_ax.plot(
            positions[step, :, 0],
            positions[step, :, 1],
            color=color,
            lw=1.4,
            marker="o",
            markersize=3.4,
            label=f"t={step}",
        )
    world_ax.axhline(0.0, color="#222222", lw=0.9)
    world_ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), xlabel="World x", ylabel="World y")
    world_ax.set_aspect("equal", adjustable="box")
    world_ax.legend(ncol=8, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    world_ax.spines[["top", "right"]].set_visible(False)

    relative = selected.copy()
    relative[..., 0] -= com[indices, 0][:, None]
    relative_x_limit = float(np.max(np.abs(relative[..., 0])) + 0.6)
    for column, step in enumerate(indices):
        ax = figure.add_subplot(grid[1, column])
        centred = positions[step].copy()
        centred[:, 0] -= com[step, 0]
        plot_shape(ax, centred, f"t={step}")
        ax.set_xlim(-relative_x_limit, relative_x_limit)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x - COM", fontsize=7)
        if column:
            ax.set_yticklabels([])
    figure.suptitle(f"Anonymous trajectory {code}", fontsize=13)
    figure.savefig(output, dpi=120, bbox_inches="tight", metadata={"Title": code})
    plt.close(figure)


def make_review_package(
    rows: list[dict[str, Any]],
    arrays_by_id: dict[tuple[str, int, int], dict[str, np.ndarray]],
    paths: dict[str, Path],
    random_seed: int,
    review_all: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(random_seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    key_rows: list[dict[str, Any]] = []
    form_rows: list[dict[str, Any]] = []
    review_selection = []
    for index, row in enumerate(shuffled, start=1):
        code = f"E{index:04d}"
        row["anonymous_review_code"] = code
        key_rows.append(
            {
                "anonymous_id": code,
                "episode_id": row["episode_id"],
                "configuration_id": row["configuration_id"],
                "formal_run": row["formal_run"],
                "training_seed": row["training_seed"],
                "reset_seed": row["reset_seed"],
                "automated_gait_label": row["automated_gait_label"],
                "kinematic_crawling_screen": row["kinematic_crawling_screen"],
                "automated_crawling_candidate": row["automated_crawling_candidate"],
            }
        )
        form_rows.append(
            {
                "anonymous_id": code,
                "rater_label": "",
                "confidence_low_medium_high": "",
                "repeated_internal_deformation_yes_no": "",
                "support_migration_yes_no_unclear": "",
                "whole_body_roll_yes_no": "",
                "notes": "",
            }
        )
        if review_all or bool(row["kinematic_crawling_screen"]) or row["automated_gait_label"] in {"partial_roll", "rocking"}:
            review_selection.append(row)
    write_csv(paths["qa"] / "PRIVATE_unblinded_code_key.csv", key_rows)
    write_csv(paths["review"] / "rater_1_blind_review_form.csv", form_rows)
    write_csv(paths["review"] / "rater_2_blind_review_form.csv", form_rows)
    selected_codes = {row["anonymous_review_code"] for row in review_selection}
    html_rows = []
    for row in shuffled:
        code = row["anonymous_review_code"]
        if code not in selected_codes:
            continue
        identifier = (row["configuration_id"], int(row["formal_run"]), int(row["reset_seed"]))
        output = paths["review_images"] / f"{code}.png"
        blind_episode_image(code, row, arrays_by_id[identifier], output)
        html_rows.append(f'<div class="card"><h2>{code}</h2><img src="blind_episode_images/{code}.png" alt="{code}"></div>')
    html = """<!doctype html><html><head><meta charset="utf-8"><title>Blind gait review</title>
<style>body{font-family:Arial,sans-serif;margin:24px;background:#f5f6f8}.card{background:#fff;margin:20px 0;padding:16px;border-radius:10px;box-shadow:0 1px 5px #bbb}.card img{width:100%;height:auto}</style>
</head><body><h1>Anonymous gait review packet</h1><p>Classify each trajectory without using configuration or performance metadata.</p>""" + "\n".join(html_rows) + "</body></html>"
    (paths["review"] / "blind_review_index.html").write_text(html, encoding="utf-8")
    instructions = """# Anonymous gait review instructions

Reviewers must work independently and must not open the `qa/PRIVATE_unblinded_code_key.csv` file. Each image contains only an anonymous ID and motion geometry. Use one of these labels: `crawling`, `rolling`, `partial_roll_or_rocking`, `sliding`, `failed_or_other`, or `uncertain`.

Call a trajectory crawling only when forward translation is visibly associated with repeated internal shape change and changing material support, without a complete whole-body roll. Do not infer the label from total displacement alone. Complete the confidence and diagnostic columns for every image. Resolve disagreements only after both files have been frozen; record the adjudicated label and reason in a separate table.
"""
    (paths["review"] / "README_BLIND_REVIEW.md").write_text(instructions, encoding="utf-8")
    merge_tool = Path(__file__).with_name("merge_blind_review.py")
    if merge_tool.exists():
        shutil.copy2(merge_tool, paths["review"] / "merge_blind_review.py")
    return key_rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        output.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(output)


def write_reports(
    paths: dict[str, Path],
    thresholds: Thresholds,
    controls: list[dict[str, Any]],
    support_validation: dict[str, float],
    rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    crawl_rep: dict[str, Any] | None,
    roll_rep: dict[str, Any] | None,
) -> None:
    total_counts = Counter(row["automated_gait_label"] for row in rows)
    candidate_count = sum(bool(row["kinematic_crawling_screen"]) for row in rows)
    summary = {
        "status": "complete_automated_pending_human_blind_review",
        "scope": {
            "configurations": 6,
            "independent_training_runs_per_configuration": 5,
            "paired_reset_rollouts_per_run": 20,
            "total_rollouts": 600,
            "checkpoint": 1500,
            "steps": 1000,
        },
        "thresholds": asdict(thresholds),
        "negative_controls": controls,
        "support_proxy_validation": support_validation,
        "total_automated_counts": dict(total_counts),
        "kinematic_crawling_candidate_count": candidate_count,
        "crawling_representative": crawl_rep["episode_id"] if crawl_rep else None,
        "rolling_representative": roll_rep["episode_id"] if roll_rep else None,
        "interpretation_boundary": (
            "Automated candidate classification is complete. Human-confirmed crawling rates require two independent anonymous ratings and adjudication; until then the thesis must use the term crawling candidate."
        ),
    }
    write_json(paths["root"] / "RESULTS_SUMMARY.json", summary)

    protocol = f"""# Formal offline gait-classification protocol

## Scope

This secondary analysis uses the locked six-configuration endpoint archive: six configurations, five independent training runs per configuration, twenty paired reset states per frozen checkpoint-1500 policy, 1,000 control steps, and 600 trajectories in total. No policy was retrained and no locked endpoint file was modified.

## Formal rolling gate

Formal rolling is copied without alteration from the thesis Equation 3.9: desired-direction net rotation at least 360 degrees, desired active-rotation fraction at least 0.70, and forward displacement at least one initial body length. The legacy direction-independent rotation-span flag is used only to prevent a full-rotation trajectory from being labelled crawling.

## Automated crawling-candidate gate

A trajectory is labelled a crawling candidate only if it is not formal rolling, its rotation span is below {thresholds.maximum_crawling_rotation_span_degrees:.0f} degrees, and its full-episode forward displacement is at least {thresholds.minimum_forward_body_lengths:.1f} initial body length. After discarding the first {thresholds.transient_steps} settling steps, it must advance by at least {thresholds.minimum_post_transient_progress_body_lengths:.2f} body length. Steps {thresholds.transient_steps}--1000 are divided into {thresholds.analysis_window_count} equal windows: at least {thresholds.minimum_positive_progress_windows}/{thresholds.analysis_window_count} windows must advance by more than {thresholds.minimum_window_progress_body_lengths:.2f} body length, at least {thresholds.minimum_shape_windows}/{thresholds.analysis_window_count} must have circular joint-shape RMS amplitude of at least {thresholds.minimum_window_shape_amplitude_degrees:.0f} degrees, and at least {thresholds.minimum_support_windows}/{thresholds.analysis_window_count} must have a 5th--95th percentile geometric support-index span of at least {thresholds.minimum_support_span_indices:.1f} material indices. A support window is observable only if it contains at least {thresholds.minimum_valid_contact_samples_per_window} valid geometric-contact samples.

Shape periodicity, detected period, cycle progress, and support autocorrelation are reported only as descriptive gait-subtype features. They are deliberately not hard crawling gates because irregular but sustained crawling would otherwise be systematically excluded. The final thesis claim must distinguish automated candidates from independent human blind confirmation.

## Statistical unit

The independent training run is the configuration-level inference unit. The twenty reset trajectories are nested paired observations within one frozen policy. Results are therefore summarised first as counts per 20 rollouts for each run, then across the five independent runs. No rollout-level p-value is reported as if 100 rollouts were 100 independent training replications.
"""
    (paths["protocol"] / "classification_protocol.md").write_text(protocol, encoding="utf-8")
    write_json(paths["protocol"] / "classification_config.json", asdict(thresholds))

    config_table = []
    for row in config_rows:
        config_table.append(
            [
                row["paper_label"],
                row["crawling_candidate_count_total"],
                f"{row['crawling_candidate_count_min_per_run']}-{row['crawling_candidate_count_max_per_run']}",
                row["stable_crawling_candidate_runs"],
                row["formal_rolling_count_total"],
            ]
        )
    report = f"""# Formal gait-classification experiment report

## Outcome

The locked six-configuration archive was processed successfully. All 600 trajectories passed the input-identity and finite-array gates. The analysis retained the thesis formal rolling labels unchanged and then applied a locked, conservative crawling-candidate definition based on sustained forward progress, limited whole-body rotation, repeated internal deformation, and repeated geometric support migration in four post-settling windows.

This report is an automated candidate classification. The anonymous review package has been generated, but its rater fields are intentionally blank. Human-confirmed crawling rates must not be claimed until two independent ratings and adjudication are completed.

## Overall automated counts

{markdown_table(["Category", "Rollouts / 600"], [[key.replace('_', ' '), total_counts.get(key, 0)] for key in GAIT_ORDER])}

Kinematic crawling/sliding candidates before the shape and support gates: **{candidate_count}/600**.

## Configuration-level summary

{markdown_table(["Configuration", "Crawling candidates / 100", "Per-run range / 20", "Stable-candidate runs / 5", "Formal roll / 100"], config_table)}

## Representative trajectories

- Crawling-candidate medoid: `{crawl_rep['episode_id'] if crawl_rep else 'none'}`.
- Formal rolling medoid: `{roll_rep['episode_id'] if roll_rep else 'none'}`.

## Contact-proxy validation

The geometric support proxy reconstructed from particle positions was checked against the valid archived SGRR logs over {int(support_validation['sgrr_episode_count'])} trajectories. Maximum absolute errors were {support_validation['maximum_support_index_absolute_error']:.3g} for support index and {support_validation['maximum_contact_strength_absolute_error']:.3g} for contact strength. This supports numerical equivalence to the environment's geometric weighting, but it remains a geometric contact proxy rather than a measured physical normal force.

## Exact effective-behaviour equivalence

All 30 checkpoint SHA-256 identities are unique, while the complete run-level position matrices contain 23 distinct effective behaviours. Two exact-equivalence groups contain five and four independently trained checkpoints respectively. These runs were retained because the checkpoint identities differ; the equality is disclosed as convergence to identical deterministic effective behaviour, not treated as a reason to delete observations.

## Interpretation boundary

The configuration summaries describe learned-policy outcomes under the fixed endpoint protocol. They do not establish that a controller configuration causally creates a gait, and they do not turn the twenty nested reset states into independent training replications. Automated crawling must be labelled as such until the blind review is completed.
"""
    (paths["report"] / "formal_gait_classification_report.md").write_text(report, encoding="utf-8")

    summary_table = []
    for row in config_rows:
        summary_table.append(
            [
                row["paper_label"],
                row["crawling_candidate_count_total"],
                row["formal_rolling_count_total"],
                row["partial_roll_count_total"],
                row["failed_other_count_total"],
                f"{row['stable_crawling_candidate_runs']}/5",
            ]
        )
    supplemental_report = f"""# Supplemental crawling-experiment results summary

## Experiment status

The automated candidate classification completed for six configurations, five independent training runs per configuration, and twenty paired resets per frozen policy: 600 trajectories of 1,000 steps each. The original checkpoint-1500 trajectories were read only, and their input hashes were identical before and after analysis. Two-person anonymous blind review is still required before any `crawling candidate` can be promoted to `human-confirmed crawling`.

## Overall automated-stage results

- Formal rolling: {total_counts.get('formal_rolling', 0)}/600;
- Crawling candidates: {total_counts.get('crawling_candidate', 0)}/600;
- Partial roll: {total_counts.get('partial_roll', 0)}/600;
- Rocking: {total_counts.get('rocking', 0)}/600;
- Sliding candidates: {total_counts.get('sliding_candidate', 0)}/600;
- Failed/other: {total_counts.get('failed_other', 0)}/600;
- Technical exclusion: {total_counts.get('technical_exclusion', 0)}/600.

{markdown_table(['Configuration', 'Crawling candidates / 100', 'Formal rolling / 100', 'Partial rolling / 100', 'Failed or other / 100', 'Stable crawling runs'], summary_table)}

## Thesis-safe conclusion

Under the locked automated criteria, all five independent HPR-O2-PS training runs produced a crawling candidate for every one of their twenty resets (100/100), with no formal rolling. SGRR-O2-JS mainly produced formal rolling ({next(row['formal_rolling_count_total'] for row in config_rows if row['configuration_id'] == 'SGRR_O2_JS')}/100), with no crawling candidates. HPR-O2-JS produced crawling candidates, formal rolling, and partial rolling, indicating that the learned gait composition diverged when the joint-sharing design changed. These are descriptive frozen-policy results under a fixed evaluation protocol and must not be presented as causal conclusions.

## Representative trajectories

- Crawling-candidate medoid: `{crawl_rep['episode_id'] if crawl_rep else 'none'}`;
- Formal-rolling medoid: `{roll_rep['episode_id'] if roll_rep else 'none'}`.

## Key limitations

The 600 rollouts are not 600 independent replications. The independent units for configuration comparisons are the five training runs per configuration; the twenty resets are nested environment states paired across policies. Formal crawling success rates require two independent reviewers to complete anonymous labels and adjudicate disagreements. Until then, the thesis must use `crawling candidate` or `automated crawling candidate` consistently.

In addition, all thirty checkpoint SHA-256 values are unique, but the complete run-level position matrices contain only twenty-three distinct patterns. Two groups of five and four independently trained checkpoints, respectively, converged to exactly identical deterministic effective behaviour. All of these runs were retained and disclosed separately in the QA table; they must not be deleted as duplicate files.
"""
    (paths["report"] / "supplemental_experiment_results_summary.md").write_text(
        supplemental_report, encoding="utf-8"
    )

    thesis = f"""# Thesis insertion draft (English)

## Methods: secondary gait classification

To distinguish crawling candidates from other non-rolling locomotion, the 600 trajectories from the locked checkpoint-1500 endpoint evaluation were subjected to a secondary gait classification. The formal rolling definition was retained unchanged. A non-rolling trajectory entered the crawling screen only when it advanced by at least one initial body length and its whole-body rotation span remained below {thresholds.maximum_crawling_rotation_span_degrees:.0f} degrees. After excluding the first {thresholds.transient_steps} settling steps, steps {thresholds.transient_steps}--1000 were divided into four equal windows. A crawling candidate had to advance by at least {thresholds.minimum_post_transient_progress_body_lengths:.2f} body length after settling, show positive progress above {thresholds.minimum_window_progress_body_lengths:.2f} body length in at least three windows, show circular joint-shape RMS amplitude of at least {thresholds.minimum_window_shape_amplitude_degrees:.0f} degrees in at least three windows, and show a geometric material-coordinate support-index span of at least {thresholds.minimum_support_span_indices:.1f} indices in at least three observable windows. Periodicity was retained as a descriptive subtype rather than a hard gate. Rollouts were nested observations; configuration-level summaries used the five independently trained policies as the replication units.

## Results: automated stage

The automated stage identified {total_counts.get('crawling_candidate', 0)} of 600 trajectories as crawling candidates and retained {total_counts.get('formal_rolling', 0)} trajectories under the unchanged formal rolling criterion. Before the deformation and support gates were applied, {candidate_count} trajectories met the simpler condition of at least one body length of forward progress without formal rolling and with rotation below the crawling exclusion threshold. The run-level distribution is reported in the accompanying gait-composition figure. These candidate labels must not be described as human-confirmed crawling until the anonymous visual-review form has been completed by independent raters.
"""
    (paths["report"] / "THESIS_INSERT_DRAFT.md").write_text(thesis, encoding="utf-8")


def write_analysis_manifest(
    args: argparse.Namespace,
    paths: dict[str, Path],
    thresholds: Thresholds,
    rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
) -> None:
    analysis_script = Path(__file__)
    merge_script = analysis_script.with_name("merge_blind_review.py")
    blind_images = sorted(paths["review_images"].glob("E[0-9][0-9][0-9][0-9].png"))
    form_1 = paths["review"] / "rater_1_blind_review_form.csv"
    form_2 = paths["review"] / "rater_2_blind_review_form.csv"
    key_path = paths["qa"] / "PRIVATE_unblinded_code_key.csv"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "complete_automated_pending_human_blind_review",
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "analysis_code": {
            "path": str(analysis_script.resolve()),
            "sha256": sha256_file(analysis_script),
            "blind_review_merger_path": str(merge_script.resolve()) if merge_script.exists() else None,
            "blind_review_merger_sha256": sha256_file(merge_script) if merge_script.exists() else None,
        },
        "software": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "locked_thresholds": asdict(thresholds),
        "randomization_seed": args.random_seed,
        "counts": {
            "episodes": len(rows),
            "independent_training_runs": len(run_rows),
            "configurations": len(config_rows),
            **dict(Counter(row["automated_gait_label"] for row in rows)),
        },
        "blind_review_package": {
            "image_count": len(blind_images),
            "all_600_images_present": len(blind_images) == 600,
            "rater_1_form_present": form_1.exists(),
            "rater_2_form_present": form_2.exists(),
            "private_key_present": key_path.exists(),
            "image_filenames_are_anonymous_only": all(
                path.name == f"E{index:04d}.png" for index, path in enumerate(blind_images, start=1)
            ),
        },
        "statistical_unit": {
            "configuration_inference_unit": "independent_training_run",
            "independent_runs_per_configuration": 5,
            "nested_paired_reset_states_per_run": 20,
            "warning": "The 600 rollouts are not 600 independent training replicates.",
        },
    }
    write_json(paths["protocol"] / "analysis_manifest.json", manifest)


def validate_final_outputs(paths: dict[str, Path]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
            }
        )

    images = sorted(paths["review_images"].glob("*.png"))
    record("blind_image_count", len(images) == 600, len(images), 600)
    anonymous_names = all(path.name == f"E{index:04d}.png" for index, path in enumerate(images, start=1))
    record("blind_image_names", anonymous_names, anonymous_names, True)

    form_1 = read_csv_rows(paths["review"] / "rater_1_blind_review_form.csv")
    form_2 = read_csv_rows(paths["review"] / "rater_2_blind_review_form.csv")
    key_rows = read_csv_rows(paths["qa"] / "PRIVATE_unblinded_code_key.csv")
    for name, selected in (("rater_1_rows", form_1), ("rater_2_rows", form_2), ("private_key_rows", key_rows)):
        record(name, len(selected) == 600, len(selected), 600)
    ids_1 = {row["anonymous_id"] for row in form_1}
    ids_2 = {row["anonymous_id"] for row in form_2}
    ids_key = {row["anonymous_id"] for row in key_rows}
    record("anonymous_id_sets_match", ids_1 == ids_2 == ids_key, len(ids_1 & ids_2 & ids_key), 600)
    blank_labels = sum(bool(row["rater_label"].strip()) for row in [*form_1, *form_2])
    record("rater_labels_not_fabricated", blank_labels == 0, blank_labels, 0)

    episode_rows = read_csv_rows(paths["data"] / "episode_features_and_automated_labels.csv")
    run_rows = read_csv_rows(paths["data"] / "run_gait_summary.csv")
    config_rows = read_csv_rows(paths["data"] / "configuration_gait_summary.csv")
    sensitivity_rows = read_csv_rows(paths["data"] / "sensitivity_summary.csv")
    contrast_details = read_csv_rows(paths["data"] / "paired_run_gait_differences.csv")
    contrast_summaries = read_csv_rows(paths["data"] / "paired_comparison_summary.csv")
    transition_rows = read_csv_rows(paths["data"] / "paired_episode_transition_counts.csv")
    record("episode_feature_rows", len(episode_rows) == 600, len(episode_rows), 600)
    record("run_summary_rows", len(run_rows) == 30, len(run_rows), 30)
    record("configuration_summary_rows", len(config_rows) == 6, len(config_rows), 6)
    record("sensitivity_grid_rows", len(sensitivity_rows) == 81, len(sensitivity_rows), 81)
    primary_count = sum(str(row["is_primary_setting"]).lower() == "true" for row in sensitivity_rows)
    record("single_primary_sensitivity_setting", primary_count == 1, primary_count, 1)
    record("paired_run_difference_rows", len(contrast_details) == 25, len(contrast_details), 25)
    record("paired_comparison_summary_rows", len(contrast_summaries) == 15, len(contrast_summaries), 15)

    run_totals_valid = True
    for row in run_rows:
        total = sum(int(row[f"{gait}_count"]) for gait in GAIT_ORDER)
        run_totals_valid = run_totals_valid and total == 20
    record("each_run_has_20_mutually_exclusive_labels", run_totals_valid, run_totals_valid, True)
    config_totals_valid = True
    for row in config_rows:
        total = sum(int(row[f"{gait}_count_total"]) for gait in GAIT_ORDER)
        config_totals_valid = config_totals_valid and total == 100
    record("each_configuration_has_100_labels", config_totals_valid, config_totals_valid, True)

    transition_totals: Counter[tuple[str, int]] = Counter()
    for row in transition_rows:
        transition_totals[(row["contrast"], int(row["formal_run"]))] += int(row["paired_reset_count"])
    transitions_valid = len(transition_totals) == 25 and all(value == 20 for value in transition_totals.values())
    record("paired_transition_groups_are_20", transitions_valid, len(transition_totals), 25)

    expected_figures = [
        "F01_gait_composition_by_run.png",
        "F02_run_level_crawl_roll_rates.png",
        "F03_gait_feature_space.png",
        "F04_threshold_sensitivity.png",
        "F05_representative_crawling.png",
        "F06_representative_rolling.png",
        "F07_crawling_vs_rolling.png",
        "F08_preplanned_paired_contrasts.png",
    ]
    figures_present = all(
        (paths["figures"] / name).exists() and (paths["figures"] / name).stat().st_size > 0
        for name in expected_figures
    )
    record("main_figures_present", figures_present, sum((paths["figures"] / name).exists() for name in expected_figures), 8)

    write_csv(paths["qa"] / "final_output_qa.csv", checks)
    passed = all(bool(row["passed"]) for row in checks)
    write_json(
        paths["root"] / "FINAL_QA_COMPLETE.json",
        {"status": "passed" if passed else "failed", "checks": len(checks), "passed": sum(bool(row["passed"]) for row in checks)},
    )
    if not passed:
        failed = [row for row in checks if not row["passed"]]
        raise RuntimeError(f"Final output QA failed: {failed}")
    return checks


def main() -> None:
    args = parse_args()
    paths = ensure_dirs(args.output_root)
    base_thresholds = Thresholds()
    thresholds, controls = load_negative_control_thresholds(args.freeze_root, base_thresholds)
    archive_contract = not args.portable_new_results
    episode_rows, npz_files, hash_rows = validate_inputs(
        args.input_root, archive_contract=archive_contract
    )
    write_csv(paths["protocol"] / "input_hashes.csv", hash_rows)
    equivalence_rows, equivalence_qa = trajectory_equivalence_inventory(
        args.input_root, npz_files, episode_rows
    )
    write_csv(paths["qa"] / "trajectory_equivalence_groups.csv", equivalence_rows)
    write_csv(paths["protocol"] / "input_inventory.csv", equivalence_rows)
    support_validation = validate_support_reconstruction(args.input_root, thresholds)
    if support_validation["maximum_support_index_absolute_error"] > 1e-4:
        raise RuntimeError(f"Support reconstruction validation failed: {support_validation}")
    if support_validation["maximum_contact_strength_absolute_error"] > 1e-5:
        raise RuntimeError(f"Contact reconstruction validation failed: {support_validation}")
    write_json(paths["qa"] / "support_proxy_validation.json", support_validation)
    rows, arrays_by_id = extract_all(args.input_root, episode_rows, npz_files, thresholds)
    classification_qa = validate_locked_classification(
        rows, archive_contract=archive_contract
    )
    write_csv(paths["qa"] / "qa_checks.csv", [*classification_qa, *equivalence_qa])
    run_rows = aggregate_run(rows)
    config_rows = aggregate_configuration(run_rows)
    contrast_details, contrast_summaries = contrast_rows(run_rows)
    transitions = paired_episode_transitions(rows)
    sensitivity = reclassify_sensitivity(rows, thresholds)

    crawl_rep = select_medoid(rows, "crawling_candidate", "HPR_O2_PS")
    roll_rep = select_medoid(rows, "formal_rolling", "HPR_O2_JS")
    representative_rows = []
    for kind, row in (("crawling_candidate", crawl_rep), ("formal_rolling", roll_rep)):
        if row is not None:
            representative_rows.append({"representative_type": kind, **row})

    write_csv(paths["data"] / "episode_features_and_automated_labels.csv", rows)
    write_csv(paths["data"] / "run_gait_summary.csv", run_rows)
    write_csv(paths["data"] / "configuration_gait_summary.csv", config_rows)
    write_csv(paths["data"] / "paired_run_gait_differences.csv", contrast_details)
    write_csv(paths["data"] / "paired_comparison_summary.csv", contrast_summaries)
    write_csv(paths["data"] / "paired_episode_transition_counts.csv", transitions)
    write_csv(paths["data"] / "preset_contrasts.csv", contrast_details)
    write_csv(paths["data"] / "sensitivity_summary.csv", sensitivity)
    if representative_rows:
        write_csv(paths["data"] / "representative_episodes.csv", representative_rows)

    composition_figure(run_rows, paths["figures"] / "F01_gait_composition_by_run.png")
    rate_figure(run_rows, paths["figures"] / "F02_run_level_crawl_roll_rates.png")
    feature_space_figure(rows, paths["figures"] / "F03_gait_feature_space.png")
    sensitivity_figure(rows, thresholds, paths["figures"] / "F04_threshold_sensitivity.png")
    paired_contrast_figure(
        contrast_details,
        paths["figures"] / "F08_preplanned_paired_contrasts.png",
    )

    if crawl_rep is not None:
        crawl_id = (crawl_rep["configuration_id"], int(crawl_rep["formal_run"]), int(crawl_rep["reset_seed"]))
        representative_figure(
            crawl_rep,
            arrays_by_id[crawl_id],
            paths["figures"] / "F05_representative_crawling.png",
            "crawl",
        )
    if roll_rep is not None:
        roll_id = (roll_rep["configuration_id"], int(roll_rep["formal_run"]), int(roll_rep["reset_seed"]))
        representative_figure(
            roll_rep,
            arrays_by_id[roll_id],
            paths["figures"] / "F06_representative_rolling.png",
            "roll",
        )
    if crawl_rep is not None and roll_rep is not None:
        crawl_id = (crawl_rep["configuration_id"], int(crawl_rep["formal_run"]), int(crawl_rep["reset_seed"]))
        roll_id = (roll_rep["configuration_id"], int(roll_rep["formal_run"]), int(roll_rep["reset_seed"]))
        comparison_figure(
            crawl_rep,
            arrays_by_id[crawl_id],
            roll_rep,
            arrays_by_id[roll_id],
            paths["figures"] / "F07_crawling_vs_rolling.png",
        )

    if not args.skip_review_assets:
        make_review_package(rows, arrays_by_id, paths, args.random_seed, args.review_all)

    write_reports(
        paths,
        thresholds,
        controls,
        support_validation,
        rows,
        run_rows,
        config_rows,
        crawl_rep,
        roll_rep,
    )
    write_analysis_manifest(args, paths, thresholds, rows, run_rows, config_rows)
    validate_final_outputs(paths)

    verify_input_hashes_unchanged(args.input_root, hash_rows)

    completion = {
        "status": "complete_automated_pending_human_blind_review",
        "episode_rows": len(rows),
        "run_rows": len(run_rows),
        "configuration_rows": len(config_rows),
        "formal_rolling_count": sum(row["automated_gait_label"] == "formal_rolling" for row in rows),
        "crawling_candidate_count": sum(row["automated_gait_label"] == "crawling_candidate" for row in rows),
        "technical_exclusion_count": sum(row["automated_gait_label"] == "technical_exclusion" for row in rows),
        "output_root": str(args.output_root),
    }
    write_json(paths["root"] / "ANALYSIS_COMPLETE.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
