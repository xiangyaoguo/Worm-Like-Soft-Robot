from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


SOURCE_STUDY_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = Path(
    os.environ.get("THESIS_HPR_OUTPUT", str(SOURCE_STUDY_ROOT))
).resolve()
DATA_ROOT = STUDY_ROOT / "data"
FIGURE_ROOT = STUDY_ROOT / "figures"
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_FIGURE_ROOT = FIGURE_ROOT / "report_jpg"
REPORT_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = (9201, 9205)
JOINTS = tuple(range(1, 9))
BLUE = "#1F5A91"
ORANGE = "#D97706"
TEAL = "#0F766E"
RED = "#B42318"
PURPLE = "#7E57C2"
GRAY = "#667085"
LIGHT = "#F2F4F7"
INK = "#172B4D"
SEED_COLOURS = {9201: BLUE, 9205: ORANGE}
CONDITION_COLOURS = {
    "BASELINE": TEAL,
    "GLOBAL_K1_OFF": RED,
    "GLOBAL_K2_OFF": PURPLE,
    "GLOBAL_BOTH_OFF": GRAY,
}
CONDITION_LABELS = {
    "BASELINE": "Baseline",
    "GLOBAL_K1_OFF": "All K₁ disabled",
    "GLOBAL_K2_OFF": "All K₂ disabled",
    "GLOBAL_BOTH_OFF": "K₁ and K₂ disabled",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D0D5DD", linewidth=0.7, alpha=0.55)
    axis.set_axisbelow(True)


def bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_effects(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = (
        "desired_revolutions",
        "forward_body_lengths",
        "desired_active_rotation_fraction",
    )
    for training_seed in SEEDS:
        policy = episodes[episodes.training_seed == training_seed]
        baseline = (
            policy[policy.condition_id == "BASELINE"]
            .set_index("reset_seed")
            .sort_index()
        )
        for condition_id, group in policy.groupby("condition_id", sort=False):
            current = group.set_index("reset_seed").sort_index()
            if list(current.index) != list(baseline.index):
                raise RuntimeError(f"Paired reset mismatch for seed {training_seed}, {condition_id}")
            row: dict[str, Any] = {
                "training_seed": int(training_seed),
                "condition_id": condition_id,
                "condition_family": str(current.condition_family.iloc[0]),
                "joint_one_based": (
                    None
                    if pd.isna(current.joint_one_based.iloc[0])
                    else int(current.joint_one_based.iloc[0])
                ),
                "success_count": int(current.success_kinematic.astype(bool).sum()),
                "success_rate": float(current.success_kinematic.astype(bool).mean()),
                "success_rate_delta_vs_baseline": float(
                    current.success_kinematic.astype(float).mean()
                    - baseline.success_kinematic.astype(float).mean()
                ),
            }
            for metric_index, metric in enumerate(metrics):
                differences = current[metric].to_numpy(float) - baseline[metric].to_numpy(float)
                low, high = bootstrap_mean_ci(
                    differences,
                    seed=20260810 + training_seed + metric_index * 1000 + sum(ord(c) for c in condition_id),
                )
                row[f"{metric}_mean"] = float(current[metric].mean())
                row[f"{metric}_delta_mean"] = float(np.mean(differences))
                row[f"{metric}_delta_ci_low"] = low
                row[f"{metric}_delta_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def figure_design() -> Path:
    fig, ax = plt.subplots(figsize=(11.4, 4.7))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    boxes = [
        (0.20, 2.45, 2.15, 1.48, "Independent HPR\ntraining runs", "random seeds 9201, 9205\ntraining checkpoint 1500"),
        (2.80, 2.45, 2.15, 1.48, "Paired reset\nstates", "20264101–20264120\n20 × 1,000 steps"),
        (5.40, 2.45, 2.30, 1.48, "Frozen\ninterventions", "global channels\njoint ablation / retention"),
        (8.15, 2.45, 3.00, 1.48, "Common kinematic\nendpoint", "≥1 revolution; ≥0.70 direction\n≥1 body length forward"),
    ]
    for x, y, w, h, title, subtitle in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#EFF6FF", edgecolor=BLUE, linewidth=1.5
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 1.03, title, ha="center", va="center", fontsize=10.2, fontweight="bold", color=INK, linespacing=1.05)
        ax.text(x + w / 2, y + 0.39, subtitle, ha="center", va="center", fontsize=8.6, color=GRAY, linespacing=1.15)
    for x0, x1 in ((2.35, 2.80), (4.95, 5.40), (7.70, 8.15)):
        ax.add_patch(FancyArrowPatch((x0, 3.19), (x1, 3.19), arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=BLUE))
    lower = FancyBboxPatch(
        (0.95, 0.42), 9.5, 1.25, boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor="#F8FAFC", edgecolor="#98A2B3", linewidth=1.2
    )
    ax.add_patch(lower)
    ax.text(5.7, 1.31, "Evidence hierarchy", ha="center", va="center", fontsize=10.6, fontweight="bold", color=INK)
    ax.text(
        5.7, 0.82,
        "20 paired rollouts describe within-policy robustness.\nThe two independent training runs are the cross-training replication units.",
        ha="center", va="center", fontsize=8.7, color=GRAY, linespacing=1.18
    )
    ax.set_title("Formal HPR frozen-policy intervention design", fontsize=16, fontweight="bold", color=INK, pad=10)
    path = FIGURE_ROOT / "fig_01_intervention_design.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_global(summary: pd.DataFrame) -> Path:
    order = ["BASELINE", "GLOBAL_K1_OFF", "GLOBAL_K2_OFF", "GLOBAL_BOTH_OFF"]
    data = summary[summary.condition_id.isin(order)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 8.2), constrained_layout=True)
    panels = [
        ("kinematic_success_rate", "Kinematic rolling success (%)", 100.0, (0, 108), None),
        ("desired_revolutions_mean", "Desired-direction revolutions", 1.0, None, 1.0),
        ("forward_body_lengths_mean", "Forward displacement (body lengths)", 1.0, None, 1.0),
        ("direction_fraction_mean", "Desired active-rotation fraction", 1.0, (0, 1.03), 0.70),
    ]
    x = np.arange(len(order))
    width = 0.34
    for axis, (metric, ylabel, multiplier, ylim, threshold) in zip(axes.flat, panels):
        for seed_index, seed in enumerate(SEEDS):
            values = []
            errors = []
            for condition_id in order:
                row = data[(data.training_seed == seed) & (data.condition_id == condition_id)].iloc[0]
                values.append(float(row[metric]) * multiplier)
                if metric == "desired_revolutions_mean":
                    errors.append(float(row["desired_revolutions_sample_sd"]))
                elif metric == "forward_body_lengths_mean":
                    errors.append(float(row["forward_body_lengths_sample_sd"]))
                elif metric == "direction_fraction_mean":
                    errors.append(float(row["direction_fraction_sample_sd"]))
                else:
                    errors.append(0.0)
            positions = x + (seed_index - 0.5) * width
            axis.bar(
                positions, values, width=width, color=SEED_COLOURS[seed], alpha=0.88,
                label=f"HPR policy {seed}", yerr=errors if any(errors) else None,
                capsize=3, edgecolor="white", linewidth=0.7
            )
            if metric == "kinematic_success_rate":
                for pos, value in zip(positions, values):
                    axis.text(pos, value + 2.3, f"{int(round(value / 5))}/20", ha="center", va="bottom", fontsize=8.3)
        if threshold is not None:
            axis.axhline(threshold, color=RED, linestyle="--", linewidth=1.1, label="Endpoint threshold")
        axis.set_xticks(x, [CONDITION_LABELS[value] for value in order], rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        if ylim:
            axis.set_ylim(*ylim)
        style_axes(axis)
    axes[0, 0].legend(frameon=False, loc="upper right")
    axes[0, 1].legend(frameon=False, loc="upper right")
    fig.suptitle("Global learned-feedback channel interventions", fontsize=16, fontweight="bold", color=INK)
    path = FIGURE_ROOT / "fig_02_global_channel_outcomes.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def heatmap(axis: plt.Axes, values: np.ndarray, row_labels: list[str], title: str, fmt: str, vmin: float, vmax: float, cmap: Any) -> None:
    image = axis.imshow(values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    axis.set_xticks(np.arange(8), [f"J{joint:02d}" for joint in JOINTS])
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            contrast = "white" if abs(value - vmin) > 0.62 * abs(vmax - vmin) else INK
            axis.text(column, row, format(value, fmt), ha="center", va="center", fontsize=8.2, color=contrast)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    plt.colorbar(image, ax=axis, fraction=0.025, pad=0.02)


def matrix_values(effects: pd.DataFrame, suffix: str, metric: str) -> np.ndarray:
    rows = []
    for seed in SEEDS:
        values = []
        for joint in JOINTS:
            condition_id = f"J{joint:02d}_{suffix}"
            row = effects[(effects.training_seed == seed) & (effects.condition_id == condition_id)].iloc[0]
            values.append(float(row[metric]))
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def figure_joint_necessity_sufficiency(effects: pd.DataFrame) -> Path:
    success = np.vstack(
        [
            matrix_values(effects, "BOTH_OFF", "success_rate") * 100.0,
            matrix_values(effects, "ONLY", "success_rate") * 100.0,
        ]
    )
    delta_turns = np.vstack(
        [
            matrix_values(effects, "BOTH_OFF", "desired_revolutions_delta_mean"),
            matrix_values(effects, "ONLY", "desired_revolutions_delta_mean"),
        ]
    )
    labels = [
        "Policy 9201 | joint off", "Policy 9205 | joint off",
        "Policy 9201 | joint only", "Policy 9205 | joint only",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11.3, 6.8), constrained_layout=True)
    heatmap(axes[0], success, labels, "Kinematic success rate (%)", ".0f", 0, 100, "YlGnBu")
    span = max(1.0, float(np.max(np.abs(delta_turns))))
    heatmap(
        axes[1], delta_turns, labels,
        "Paired change in desired revolutions relative to baseline",
        "+.2f", -span, span,
        LinearSegmentedColormap.from_list("diverging", [RED, "#FFFFFF", BLUE]),
    )
    fig.suptitle("Whole-joint ablation and single-joint retention", fontsize=16, fontweight="bold", color=INK)
    path = FIGURE_ROOT / "fig_03_joint_ablation_retention_heatmap.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_channel_heatmap(effects: pd.DataFrame) -> Path:
    success = np.vstack(
        [
            matrix_values(effects, "K1_OFF", "success_rate") * 100.0,
            matrix_values(effects, "K2_OFF", "success_rate") * 100.0,
        ]
    )
    delta_turns = np.vstack(
        [
            matrix_values(effects, "K1_OFF", "desired_revolutions_delta_mean"),
            matrix_values(effects, "K2_OFF", "desired_revolutions_delta_mean"),
        ]
    )
    labels = [
        "Policy 9201 · K₁ off", "Policy 9205 · K₁ off",
        "Policy 9201 · K₂ off", "Policy 9205 · K₂ off",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11.3, 6.8), constrained_layout=True)
    heatmap(axes[0], success, labels, "Kinematic success rate (%)", ".0f", 0, 100, "YlGnBu")
    span = max(0.5, float(np.max(np.abs(delta_turns))))
    heatmap(
        axes[1], delta_turns, labels,
        "Paired change in desired revolutions relative to baseline",
        "+.2f", -span, span,
        LinearSegmentedColormap.from_list("diverging2", [RED, "#FFFFFF", BLUE]),
    )
    fig.suptitle("Joint-specific K₁ and K₂ channel ablations", fontsize=16, fontweight="bold", color=INK)
    path = FIGURE_ROOT / "fig_04_joint_channel_effect_heatmap.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.30, facecolor="white")
    plt.close(fig)
    return path


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank(method="average").to_numpy(float)
    ry = pd.Series(y).rank(method="average").to_numpy(float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def figure_cross_seed(effects: pd.DataFrame) -> tuple[Path, dict[str, Any]]:
    families = [
        ("whole_joint_ablation", "Whole joint off", "o", TEAL),
        ("joint_k1_ablation", "K₁ off", "s", BLUE),
        ("joint_k2_ablation", "K₂ off", "^", ORANGE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 4.25), constrained_layout=True)
    agreement: dict[str, Any] = {}
    all_values: list[float] = []
    point_offsets = [(5, 5), (5, -10), (-17, 5), (5, 5), (5, -10), (-17, 5), (5, 5), (5, -10)]
    for axis, (family, label, marker, colour) in zip(axes, families):
        subset = effects[effects.condition_family == family]
        x = []
        y = []
        joints = []
        for joint in JOINTS:
            a = subset[(subset.training_seed == 9201) & (subset.joint_one_based == joint)].iloc[0]
            b = subset[(subset.training_seed == 9205) & (subset.joint_one_based == joint)].iloc[0]
            x.append(float(a.desired_revolutions_delta_mean))
            y.append(float(b.desired_revolutions_delta_mean))
            joints.append(joint)
        x_a = np.asarray(x)
        y_a = np.asarray(y)
        all_values.extend(x)
        all_values.extend(y)
        rho = rank_correlation(x_a, y_a)
        sign_agreement = float(np.mean(np.sign(x_a) == np.sign(y_a)))
        agreement[family] = {
            "spearman_rank_correlation": rho,
            "effect_direction_agreement_fraction": sign_agreement,
        }
        axis.scatter(x_a, y_a, s=58, marker=marker, color=colour, edgecolor="white", linewidth=0.7)
        for xv, yv, joint, offset in zip(x_a, y_a, joints, point_offsets):
            axis.annotate(f"J{joint}", (xv, yv), xytext=offset, textcoords="offset points", fontsize=7.2, color=INK)
        axis.set_title(f"{label}\nρ={rho:.2f}; direction agreement={sign_agreement:.0%}", fontsize=10.2, fontweight="bold", color=INK)
    span = max(0.5, max(abs(value) for value in all_values) * 1.1)
    for index, axis in enumerate(axes):
        axis.plot([-span, span], [-span, span], linestyle="--", color="#98A2B3", linewidth=0.9)
        axis.axhline(0, color="#D0D5DD", linewidth=0.8)
        axis.axvline(0, color="#D0D5DD", linewidth=0.8)
        axis.set_xlim(-span, span)
        axis.set_ylim(-span, span)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("HPR policy 9201 Δ revolutions", fontsize=9)
        if index == 0:
            axis.set_ylabel("HPR policy 9205 Δ revolutions", fontsize=9)
        style_axes(axis)
    fig.suptitle("Cross-policy agreement in joint-intervention effects", fontsize=15, fontweight="bold", color=INK)
    path = FIGURE_ROOT / "fig_05_cross_seed_agreement.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path, agreement


def load_trajectory(training_seed: int, condition_id: str, reset_seed: int = 20264101) -> np.ndarray:
    path = DATA_ROOT / "trajectories" / f"seed{training_seed}__{condition_id}__reset{reset_seed}.npz"
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["positions"], dtype=np.complex128)


def body_length(position: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(position))))


def figure_morphology(training_seed: int) -> Path:
    order = ["BASELINE", "GLOBAL_K1_OFF", "GLOBAL_K2_OFF", "GLOBAL_BOTH_OFF"]
    frames = [0, 250, 500, 750, 1000]
    fig, axes = plt.subplots(len(order), len(frames), figsize=(13.2, 8.3), constrained_layout=True)
    for row_index, condition_id in enumerate(order):
        trajectory = load_trajectory(training_seed, condition_id)
        initial_com_x = float(np.real(np.mean(trajectory[0])))
        length = body_length(trajectory[0])
        for column_index, frame in enumerate(frames):
            axis = axes[row_index, column_index]
            pos = trajectory[frame]
            com_x = float(np.real(np.mean(pos)))
            centred_x = np.real(pos) - com_x
            y = np.imag(pos)
            axis.plot(centred_x, y, color="#344054", linewidth=1.4, zorder=1)
            axis.scatter(centred_x, y, c=np.arange(len(pos)), cmap="viridis", s=28, edgecolor="white", linewidth=0.4, zorder=2)
            axis.axhline(0, color="#98A2B3", linewidth=1.0)
            axis.set_xlim(-5.3, 5.3)
            axis.set_ylim(-0.15, 5.1)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            displacement_bl = (com_x - initial_com_x) / max(length, 1e-12)
            axis.text(0.5, 0.02, f"x={displacement_bl:.2f} BL", transform=axis.transAxes, ha="center", va="bottom", fontsize=7.5, color=GRAY)
            if row_index == 0:
                axis.set_title(f"t = {frame}", fontsize=10, fontweight="bold", color=INK)
            if column_index == 0:
                axis.text(-0.08, 0.5, CONDITION_LABELS[condition_id], transform=axis.transAxes, ha="right", va="center", fontsize=9.2, fontweight="bold", color=INK)
    fig.suptitle(f"Matched morphology sequence: formal HPR policy {training_seed}, reset 20264101", fontsize=15, fontweight="bold", color=INK)
    figure_number = "06" if training_seed == 9201 else "07"
    path = FIGURE_ROOT / f"fig_{figure_number}_morphology_seed{training_seed}.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def trajectory_series(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    com_x = np.real(np.mean(trajectory, axis=1))
    initial_length = body_length(trajectory[0])
    forward_bl = (com_x - com_x[0]) / max(initial_length, 1e-12)
    rotation = []
    for previous, current in zip(trajectory[:-1], trajectory[1:]):
        previous_centered = previous - np.mean(previous)
        current_centered = current - np.mean(current)
        cross = np.sum(np.conj(previous_centered) * current_centered)
        rotation.append(0.0 if abs(cross) <= 1e-12 else -float(np.angle(cross)))
    desired_revolutions = np.r_[0.0, np.cumsum(rotation)] / (2.0 * np.pi)
    return desired_revolutions, forward_bl


def figure_time_series() -> Path:
    order = ["BASELINE", "GLOBAL_K1_OFF", "GLOBAL_K2_OFF", "GLOBAL_BOTH_OFF"]
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.8), sharex=True, constrained_layout=True)
    for column, seed in enumerate(SEEDS):
        for condition_id in order:
            trajectory = load_trajectory(seed, condition_id)
            revolutions, forward_bl = trajectory_series(trajectory)
            colour = CONDITION_COLOURS[condition_id]
            label = CONDITION_LABELS[condition_id]
            axes[0, column].plot(revolutions, color=colour, linewidth=1.6, label=label)
            axes[1, column].plot(forward_bl, color=colour, linewidth=1.6, label=label)
        axes[0, column].axhline(1.0, color=RED, linestyle="--", linewidth=1.0)
        axes[1, column].axhline(1.0, color=RED, linestyle="--", linewidth=1.0)
        axes[0, column].set_title(f"Formal HPR policy {seed}", fontsize=12, fontweight="bold", color=INK)
        axes[1, column].set_xlabel("Control step")
        style_axes(axes[0, column])
        style_axes(axes[1, column])
    axes[0, 0].set_ylabel("Cumulative desired revolutions")
    axes[1, 0].set_ylabel("Forward displacement (body lengths)")
    axes[0, 1].legend(frameon=False, loc="upper left", fontsize=8.4)
    fig.suptitle("Matched representative trajectories under global interventions", fontsize=15, fontweight="bold", color=INK)
    path = FIGURE_ROOT / "fig_08_representative_time_series.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def summary_payload(summary: pd.DataFrame, effects: pd.DataFrame, agreement: dict[str, Any], figure_paths: Iterable[Path]) -> dict[str, Any]:
    global_ids = ["BASELINE", "GLOBAL_K1_OFF", "GLOBAL_K2_OFF", "GLOBAL_BOTH_OFF"]
    global_rows = []
    for seed in SEEDS:
        for condition_id in global_ids:
            row = summary[(summary.training_seed == seed) & (summary.condition_id == condition_id)].iloc[0]
            global_rows.append(
                {
                    "training_seed": seed,
                    "condition_id": condition_id,
                    "success_count": int(row.kinematic_success_count),
                    "desired_revolutions_mean": float(row.desired_revolutions_mean),
                    "forward_body_lengths_mean": float(row.forward_body_lengths_mean),
                    "direction_fraction_mean": float(row.direction_fraction_mean),
                }
            )
    family_summaries = {}
    for family in (
        "whole_joint_ablation",
        "single_joint_retention",
        "joint_k1_ablation",
        "joint_k2_ablation",
    ):
        subset = effects[effects.condition_family == family]
        family_summaries[family] = {
            "minimum_success_count_across_policy_conditions": int(subset.success_count.min()),
            "maximum_success_count_across_policy_conditions": int(subset.success_count.max()),
            "mean_paired_revolution_change": float(subset.desired_revolutions_delta_mean.mean()),
            "conditions_with_same_effect_direction_across_two_policies": int(
                sum(
                    np.sign(
                        subset[(subset.training_seed == 9201) & (subset.joint_one_based == joint)]
                        .desired_revolutions_delta_mean.iloc[0]
                    )
                    == np.sign(
                        subset[(subset.training_seed == 9205) & (subset.joint_one_based == joint)]
                        .desired_revolutions_delta_mean.iloc[0]
                    )
                    for joint in JOINTS
                )
            ),
        }
    return {
        "schema": "formal_hpr_freeze_validation/analysis_summary/v1",
        "formal_training_units": 2,
        "paired_rollouts_per_policy_condition": 20,
        "total_rollouts": int(len(pd.read_csv(DATA_ROOT / "episode_results.csv"))),
        "global_conditions": global_rows,
        "family_summaries": family_summaries,
        "cross_seed_agreement": agreement,
        "figure_paths": [str(path) for path in figure_paths],
        "evidence_boundary": (
            "Agreement across the two consistently rolling formal HPR policies supports "
            "cross-policy replication within this formal study, not a universal mechanism "
            "for every possible HPR-trained policy. The 20 reset rollouts are nested paired "
            "conditions, not independent training runs."
        ),
    }


def main() -> None:
    episodes = pd.read_csv(DATA_ROOT / "episode_results.csv")
    summary = pd.read_csv(DATA_ROOT / "condition_policy_summary.csv").rename(
        columns={"id": "condition_id", "family": "condition_family"}
    )
    if len(episodes) != 1440 or len(summary) != 72:
        raise RuntimeError(f"Incomplete full matrix: episodes={len(episodes)}, summaries={len(summary)}")
    effects = paired_effects(episodes)
    effects.to_csv(DATA_ROOT / "paired_effects.csv", index=False, encoding="utf-8-sig")

    figures = [figure_design(), figure_global(summary)]
    figures.append(figure_joint_necessity_sufficiency(effects))
    figures.append(figure_channel_heatmap(effects))
    cross_path, agreement = figure_cross_seed(effects)
    figures.append(cross_path)
    figures.append(figure_morphology(9201))
    figures.append(figure_morphology(9205))
    figures.append(figure_time_series())

    # Word's PDF exporter can stall on high-resolution RGBA PNGs even when
    # every pixel is opaque.  Re-encode the report assets as ordinary RGB PNGs
    # without changing their visible content.
    for figure_path in figures:
        with Image.open(figure_path) as image:
            rgb = image.convert("RGB")
            rgb.save(figure_path, format="PNG", dpi=(240, 240), optimize=True)
            report_rgb = rgb
            if rgb.width > 1800:
                report_height = round(rgb.height * 1800 / rgb.width)
                report_rgb = rgb.resize((1800, report_height), Image.Resampling.LANCZOS)
            report_rgb.save(
                REPORT_FIGURE_ROOT / f"{figure_path.stem}.jpg",
                format="JPEG",
                quality=92,
                subsampling=0,
                dpi=(240, 240),
                optimize=False,
            )

    # Combine the two policy-specific morphology plates for the Word report.
    # Keeping the two full-resolution PNGs separate preserves reusable assets,
    # while one composite inline image avoids a Microsoft Word PDF-export bug.
    morphology_paths = [
        REPORT_FIGURE_ROOT / "fig_06_morphology_seed9201.jpg",
        REPORT_FIGURE_ROOT / "fig_07_morphology_seed9205.jpg",
    ]
    morphology_images = [Image.open(path).convert("RGB") for path in morphology_paths]
    try:
        gap = 24
        canvas = Image.new(
            "RGB",
            (max(image.width for image in morphology_images), sum(image.height for image in morphology_images) + gap),
            "white",
        )
        y = 0
        for image in morphology_images:
            canvas.paste(image, ((canvas.width - image.width) // 2, y))
            y += image.height + gap
        canvas.save(
            REPORT_FIGURE_ROOT / "fig_06_morphology_both_policies.jpg",
            format="JPEG",
            quality=92,
            subsampling=0,
            dpi=(240, 240),
            optimize=True,
        )
    finally:
        for image in morphology_images:
            image.close()

    evidence_paths = [
        REPORT_FIGURE_ROOT / "fig_05_cross_seed_agreement.jpg",
        REPORT_FIGURE_ROOT / "fig_08_representative_time_series.jpg",
    ]
    evidence_images = [Image.open(path).convert("RGB") for path in evidence_paths]
    try:
        gap = 24
        canvas = Image.new(
            "RGB",
            (max(image.width for image in evidence_images), sum(image.height for image in evidence_images) + gap),
            "white",
        )
        y = 0
        for image in evidence_images:
            canvas.paste(image, ((canvas.width - image.width) // 2, y))
            y += image.height + gap
        canvas.save(
            REPORT_FIGURE_ROOT / "fig_05_cross_policy_and_time_series.jpg",
            format="JPEG",
            quality=92,
            subsampling=0,
            dpi=(240, 240),
            optimize=True,
        )
    finally:
        for image in evidence_images:
            image.close()

    payload = summary_payload(summary, effects, agreement, figures)
    atomic_json(DATA_ROOT / "ANALYSIS_SUMMARY.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
