from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
FIGURE_ROOT = ROOT / "figures"
FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
FORMAL_CONFIG = Path(
    r"C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\roll_learning"
    r"\obs2_roll_repro_v2_1_formal_20260803_r2\_control\experiment_config.json"
)
REFERENCE_EFFECTS = Path(
    r"C:\Users\PUBLIC_USER\Documents\GraduateThesisProject"
    r"\formal_hpr_freeze_validation_20260810\data\paired_effects.csv"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


formal_config = load_json(FORMAL_CONFIG)
site_packages = str(formal_config["runtime"]["site_packages"])
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


INK = "#172B4D"
BLUE = "#1F5A91"
TEAL = "#0F766E"
ORANGE = "#D97706"
RED = "#B42318"
PURPLE = "#7E57C2"
GRAY = "#667085"
JOINTS = tuple(range(1, 9))
GLOBAL_ORDER = ("BASELINE", "GLOBAL_K1_OFF", "GLOBAL_K2_OFF", "GLOBAL_BOTH_OFF")
GLOBAL_LABELS = {
    "BASELINE": "Baseline",
    "GLOBAL_K1_OFF": "All K1 disabled",
    "GLOBAL_K2_OFF": "All K2 disabled",
    "GLOBAL_BOTH_OFF": "K1 and K2 disabled",
}
GLOBAL_COLOURS = (TEAL, BLUE, ORANGE, GRAY)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean_ci(values: np.ndarray, seed: int, samples: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if np.all(values == values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def paired_effects(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline = episodes[episodes.condition_id == "BASELINE"].set_index("reset_seed").sort_index()
    metric_names = (
        "desired_revolutions",
        "forward_body_lengths",
        "desired_active_rotation_fraction",
    )
    for index, (condition_id, group) in enumerate(episodes.groupby("condition_id", sort=False)):
        current = group.set_index("reset_seed").sort_index()
        if not current.index.equals(baseline.index):
            raise RuntimeError(f"Paired reset mismatch: {condition_id}")
        first = current.iloc[0]
        row: dict[str, Any] = {
            "paper_run_id": 2,
            "internal_training_seed": 9203,
            "condition_id": condition_id,
            "condition_family": first["condition_family"],
            "joint_one_based": first["joint_one_based"],
            "success_count": int(current.success_kinematic.astype(bool).sum()),
            "success_rate": float(current.success_kinematic.astype(bool).mean()),
            "success_rate_delta_vs_baseline": float(
                current.success_kinematic.astype(bool).mean()
                - baseline.success_kinematic.astype(bool).mean()
            ),
        }
        for metric_index, metric in enumerate(metric_names):
            current_values = current[metric].to_numpy(float)
            delta = current_values - baseline[metric].to_numpy(float)
            low, high = bootstrap_mean_ci(delta, seed=20260811 + 100 * index + metric_index)
            row[f"{metric}_mean"] = float(current_values.mean())
            row[f"{metric}_delta_mean"] = float(delta.mean())
            row[f"{metric}_delta_ci_low"] = low
            row[f"{metric}_delta_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D0D5DD", linewidth=0.7, alpha=0.55)
    axis.set_axisbelow(True)


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = FIGURE_ROOT / filename
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def figure_global(summary: pd.DataFrame) -> Path:
    subset = summary.set_index("id").loc[list(GLOBAL_ORDER)]
    x = np.arange(len(GLOBAL_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.4), constrained_layout=True)
    panels = (
        # The success-count panel has no horizontal threshold: seven is the
        # observed run-2 baseline count, not a decision threshold.  Its bar
        # already provides the appropriate reference.
        ("kinematic_success_count", "Common-criterion successes (/20)", None),
        ("desired_revolutions_mean", "Desired revolutions (mean)", None),
        ("forward_body_lengths_mean", "Forward displacement (body lengths, mean)", 1.0),
        ("direction_fraction_mean", "Desired active-rotation fraction (mean)", 0.70),
    )
    for axis, (field, ylabel, threshold) in zip(axes.flat, panels):
        values = subset[field].to_numpy(float)
        bars = axis.bar(x, values, color=GLOBAL_COLOURS, width=0.68)
        for bar, value in zip(bars, values):
            label = f"{value:.0f}" if field == "kinematic_success_count" else f"{value:.2f}"
            axis.text(bar.get_x() + bar.get_width() / 2, value, label, ha="center", va="bottom", fontsize=8.5)
        if threshold is not None:
            axis.axhline(threshold, color=RED, linestyle="--", linewidth=1.0)
        axis.set_xticks(x, [GLOBAL_LABELS[item] for item in GLOBAL_ORDER], rotation=17, ha="right")
        axis.set_ylabel(ylabel)
        style_axis(axis)
    fig.suptitle("HPR run 2: global learned-feedback interventions", fontsize=15, fontweight="bold", color=INK)
    return save_figure(fig, "fig_run2_global_channel_outcomes.png")


def heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    rows: list[str],
    title: str,
    fmt: str,
    vmin: float,
    vmax: float,
    cmap: Any,
) -> None:
    image = axis.imshow(values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    axis.set_xticks(np.arange(8), [f"J{joint:02d}" for joint in JOINTS])
    axis.set_yticks(np.arange(len(rows)), rows)
    axis.set_title(title, fontsize=11.5, fontweight="bold", color=INK)
    span = max(abs(vmin), abs(vmax), 1e-12)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            colour = "white" if abs(value) > 0.58 * span else INK
            axis.text(column, row, format(value, fmt), ha="center", va="center", fontsize=8.2, color=colour)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    plt.colorbar(image, ax=axis, fraction=0.026, pad=0.02)


def joint_matrix(effects: pd.DataFrame, suffixes: tuple[str, ...], field: str) -> np.ndarray:
    rows = []
    for suffix in suffixes:
        rows.append(
            [
                float(effects[effects.condition_id == f"J{joint:02d}_{suffix}"].iloc[0][field])
                for joint in JOINTS
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def figure_ablation_retention(effects: pd.DataFrame) -> Path:
    suffixes = ("BOTH_OFF", "ONLY")
    success = 20.0 * joint_matrix(effects, suffixes, "success_rate")
    delta = joint_matrix(effects, suffixes, "desired_revolutions_delta_mean")
    span = max(0.5, float(np.max(np.abs(delta))))
    fig, axes = plt.subplots(2, 1, figsize=(11.3, 5.8), constrained_layout=True)
    heatmap(axes[0], success, ["Joint disabled", "Joint retained alone"], "Success count (/20)", ".0f", 0, 20, "YlGnBu")
    heatmap(
        axes[1], delta, ["Joint disabled", "Joint retained alone"],
        "Paired change in desired revolutions vs baseline", "+.2f", -span, span,
        LinearSegmentedColormap.from_list("effect", [RED, "white", BLUE]),
    )
    fig.suptitle("HPR run 2: whole-joint necessity and single-joint sufficiency tests", fontsize=15, fontweight="bold", color=INK)
    return save_figure(fig, "fig_run2_joint_ablation_retention.png")


def figure_channel_effects(effects: pd.DataFrame) -> Path:
    suffixes = ("K1_OFF", "K2_OFF")
    success = 20.0 * joint_matrix(effects, suffixes, "success_rate")
    delta = joint_matrix(effects, suffixes, "desired_revolutions_delta_mean")
    span = max(0.5, float(np.max(np.abs(delta))))
    fig, axes = plt.subplots(2, 1, figsize=(11.3, 5.8), constrained_layout=True)
    heatmap(axes[0], success, ["K1 disabled", "K2 disabled"], "Success count (/20)", ".0f", 0, 20, "YlGnBu")
    heatmap(
        axes[1], delta, ["K1 disabled", "K2 disabled"],
        "Paired change in desired revolutions vs baseline", "+.2f", -span, span,
        LinearSegmentedColormap.from_list("effect2", [RED, "white", BLUE]),
    )
    fig.suptitle("HPR run 2: joint-specific channel ablations", fontsize=15, fontweight="bold", color=INK)
    return save_figure(fig, "fig_run2_joint_channel_effects.png")


def figure_cross_run(effects: pd.DataFrame) -> Path:
    reference = pd.read_csv(REFERENCE_EFFECTS)
    reference["paper_run_id"] = reference.training_seed.map({9201: 0, 9205: 4})
    current = effects.copy()
    combined = pd.concat([reference, current], ignore_index=True, sort=False)
    families = (
        ("whole_joint_ablation", "Whole joint disabled"),
        ("joint_k1_ablation", "K1 disabled"),
        ("joint_k2_ablation", "K2 disabled"),
    )
    matrices = []
    for family, _ in families:
        rows = []
        for run_id in (0, 2, 4):
            run = combined[(combined.paper_run_id == run_id) & (combined.condition_family == family)]
            rows.append(
                [
                    float(run[run.joint_one_based == joint].iloc[0].desired_revolutions_delta_mean)
                    for joint in JOINTS
                ]
            )
        matrices.append(np.asarray(rows, dtype=np.float64))
    span = max(0.5, max(float(np.max(np.abs(matrix))) for matrix in matrices))
    fig, axes = plt.subplots(3, 1, figsize=(11.3, 8.0), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("cross", [RED, "white", BLUE])
    for axis, matrix, (_, title) in zip(axes, matrices, families):
        heatmap(
            axis, matrix, ["HPR run 0", "HPR run 2", "HPR run 4"],
            f"{title}: paired change in desired revolutions", "+.2f", -span, span, cmap,
        )
    fig.suptitle("Cross-policy comparison of frozen-policy intervention effects", fontsize=15, fontweight="bold", color=INK)
    return save_figure(fig, "fig_run2_effect_agreement_runs0_2_4.png")


def main() -> None:
    episodes_path = DATA_ROOT / "episode_results.csv"
    summary_path = DATA_ROOT / "condition_policy_summary.csv"
    if not episodes_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Run the full run 2 matrix before plotting")
    episodes = pd.read_csv(episodes_path)
    summary = pd.read_csv(summary_path)
    if len(episodes) != 720 or len(summary) != 36:
        raise RuntimeError(f"Incomplete run 2 matrix: episodes={len(episodes)}, summary={len(summary)}")
    effects = paired_effects(episodes)
    effects.to_csv(DATA_ROOT / "paired_effects.csv", index=False, encoding="utf-8-sig")
    figures = (
        figure_global(summary),
        figure_ablation_retention(effects),
        figure_channel_effects(effects),
        figure_cross_run(effects),
    )
    figure_receipts = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in figures
    ]
    baseline = effects[effects.condition_id == "BASELINE"].iloc[0]
    payload = {
        "schema": "formal_hpr_run2_freeze/analysis_summary/v1",
        "paper_run_id": 2,
        "internal_training_seed": 9203,
        "paired_resets_per_condition": 20,
        "condition_count": 36,
        "total_rollouts": 720,
        "baseline_common_success_count": int(baseline.success_count),
        "figure_receipts": figure_receipts,
        "reference_effects_runs0_4": {
            "path": str(REFERENCE_EFFECTS),
            "sha256": sha256_file(REFERENCE_EFFECTS),
        },
        "evidence_boundary": (
            "HPR run 2 is an independently trained policy evaluated under nested paired resets. "
            "The cross-policy panel compares effect patterns across paper-facing HPR runs 0, 2, "
            "and 4; the 20 reset rollouts are not independent training runs."
        ),
    }
    (DATA_ROOT / "ANALYSIS_SUMMARY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
