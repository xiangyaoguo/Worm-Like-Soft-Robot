from __future__ import annotations

"""Create report-ready figures from the frozen local-actor analysis tables.

This script never opens a checkpoint or trajectory.  It only reads outputs made by
``analyze_local_actor_jacobian_physics.py`` and writes PNG/PDF figures plus a
figure manifest to the same new analysis bundle.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "analysis_local_actor"
EVENT_ORDER = [
    "official_prelaunch",
    "official_rolling_outside_pulse",
    "official_pulse_q1",
    "official_pulse_q2",
    "official_pulse_q3",
    "official_pulse_q4",
    "official_pulse_q5",
]
EVENT_SHORT = {
    # No sampled step falls in this bin for the frozen C11 trace set.  Keep the
    # preregistered column visible, but label it explicitly so an empty cell is
    # not mistaken for a numerical zero.
    "official_prelaunch": "prelaunch (NA)",
    "official_rolling_outside_pulse": "rolling/outside",
    "official_pulse_q1": "pulse Q1",
    "official_pulse_q2": "pulse Q2",
    "official_pulse_q3": "pulse Q3",
    "official_pulse_q4": "pulse Q4",
    "official_pulse_q5": "pulse Q5",
}
BLUE = "#1f5f8b"
ORANGE = "#d97829"
INK = "#17212b"


def configure_plotting() -> None:
    matplotlib.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#647484",
            "axes.labelcolor": INK,
            "xtick.color": "#405060",
            "ytick.color": "#405060",
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cmap: str = "RdBu_r",
    center_zero: bool = True,
    annotate: bool = False,
) -> Any:
    values = np.asarray(values, dtype=np.float64)
    if center_zero:
        limit = float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 1.0
        limit = max(limit, 1e-12)
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(xlabels)), labels=xlabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)
    ax.set_title(title, loc="left", fontweight="bold")
    if annotate and values.size <= 256:
        threshold = 0.55 * max(float(np.nanmax(np.abs(values))), 1e-12)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if abs(value) >= threshold else INK,
                )
    return image


def structural_mask_figure(input_dir: Path, figure_dir: Path) -> list[dict[str, Any]]:
    data = pd.read_csv(input_dir / "jacobian_structural_mask_16x16.csv")
    matrix = data.pivot(
        index="output_label", columns="input_label", values="local_block_entry"
    )
    output_order = (
        data.sort_values("output_index")["output_label"].drop_duplicates().tolist()
    )
    input_order = data.sort_values("input_index")["input_label"].drop_duplicates().tolist()
    matrix = matrix.loc[output_order, input_order].astype(float)
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(16), labels=input_order, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(16), labels=output_order, fontsize=7)
    ax.set_title(
        "Direct policy Jacobian structure: 8 local 2×2 blocks, 224 cross-joint structural zeros",
        loc="left",
        fontweight="bold",
    )
    ax.set_xlabel("actor input")
    ax.set_ylabel("physical-K output")
    for boundary in range(2, 16, 2):
        ax.axhline(boundary - 0.5, color="white", lw=0.8)
        ax.axvline(boundary - 0.5, color="white", lw=0.8)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="local block entry")
    paths = save_figure(fig, figure_dir / "J01_direct_jacobian_structure")
    return [
        {
            "figure_id": "J01",
            "title": "Direct 16x16 Jacobian structural mask",
            "path": str(path),
            "evidence_type": "architecture_identity",
        }
        for path in paths
    ]


def representative_full_jacobian_figure(
    input_dir: Path, figure_dir: Path
) -> list[dict[str, Any]]:
    data = pd.read_csv(input_dir / "jacobian_full_16x16_by_seed_event.csv")
    phase = "official_pulse_q3" if "official_pulse_q3" in set(data.event_bin) else data.event_bin.iloc[0]
    selected = data[data.event_bin == phase]
    grouped = (
        selected.groupby(
            ["output_index", "output_label", "input_index", "input_label"], as_index=False
        )["median_derivative"]
        .median()
        .sort_values(["output_index", "input_index"])
    )
    output_order = grouped.sort_values("output_index")["output_label"].drop_duplicates().tolist()
    input_order = grouped.sort_values("input_index")["input_label"].drop_duplicates().tolist()
    matrix = grouped.pivot(
        index="output_label", columns="input_label", values="median_derivative"
    ).loc[output_order, input_order]
    fig, ax = plt.subplots(figsize=(12.2, 10.0))
    image = annotated_heatmap(
        ax,
        matrix.to_numpy(),
        input_order,
        output_order,
        f"Median direct 16×16 physical-K Jacobian across training seeds — {EVENT_SHORT.get(phase, phase)}",
        annotate=True,
    )
    ax.set_xlabel("observation input")
    ax.set_ylabel("K output")
    for boundary in range(2, 16, 2):
        ax.axhline(boundary - 0.5, color="#617080", lw=0.45)
        ax.axvline(boundary - 0.5, color="#617080", lw=0.45)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="d physical K / d observation")
    paths = save_figure(fig, figure_dir / "J02_full_jacobian_pulse_q3")
    return [
        {
            "figure_id": "J02",
            "title": "Representative event-aligned full 16x16 Jacobian",
            "path": str(path),
            "evidence_type": "analytic_policy_derivative",
        }
        for path in paths
    ]


def local_derivative_phase_figures(
    input_dir: Path, figure_dir: Path
) -> list[dict[str, Any]]:
    data = pd.read_csv(input_dir / "jacobian_local_summary_by_seed_event.csv")
    data["signal"] = (
        data["joint"]
        + " "
        + data["K_channel"]
        + " ← "
        + data["observation_channel"]
    )
    signals = []
    for joint in [f"J{index:02d}" for index in range(1, 9)]:
        for k_channel in ("K1", "K2"):
            for observation_channel in ("delta_theta", "theta_dot"):
                signals.append(f"{joint} {k_channel} ← {observation_channel}")
    grouped = (
        data.groupby(["signal", "event_bin"], as_index=False)["derivative_median"]
        .median()
        .pivot(index="signal", columns="event_bin", values="derivative_median")
        .reindex(index=signals, columns=EVENT_ORDER)
    )
    fig, ax = plt.subplots(figsize=(12.0, 14.5))
    image = annotated_heatmap(
        ax,
        grouped.to_numpy(),
        [EVENT_SHORT[name] for name in EVENT_ORDER],
        signals,
        "Local actor sensitivity on visited states (median across five training seeds)",
    )
    ax.set_xlabel("official-event-aligned analysis bin")
    ax.set_ylabel("local derivative")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="median derivative")
    paths = save_figure(fig, figure_dir / "J03_local_derivative_by_event")

    consistency = (
        data.groupby(["signal", "event_bin"], as_index=False)["sign_consistency"]
        .median()
        .pivot(index="signal", columns="event_bin", values="sign_consistency")
        .reindex(index=signals, columns=EVENT_ORDER)
    )
    fig, ax = plt.subplots(figsize=(12.0, 14.5))
    image = annotated_heatmap(
        ax,
        consistency.to_numpy(),
        [EVENT_SHORT[name] for name in EVENT_ORDER],
        signals,
        "Within-seed, within-bin derivative sign consistency on visited states",
        cmap="viridis",
        center_zero=False,
    )
    image.set_clim(0.5, 1.0)
    ax.set_xlabel("official-event-aligned analysis bin")
    ax.set_ylabel("local derivative")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="max(P[positive], P[negative])")
    paths.extend(save_figure(fig, figure_dir / "J04_local_derivative_sign_consistency"))
    rows: list[dict[str, Any]] = []
    for path in paths[:2]:
        rows.append(
            {
                "figure_id": "J03",
                "title": "Local derivatives by event-aligned bin",
                "path": str(path),
                "evidence_type": "analytic_policy_derivative",
            }
        )
    for path in paths[2:]:
        rows.append(
            {
                "figure_id": "J04",
                "title": "Within-seed, within-bin local derivative sign consistency",
                "path": str(path),
                "evidence_type": "analytic_policy_derivative",
            }
        )
    return rows


def physics_phase_figures(input_dir: Path, figure_dir: Path) -> list[dict[str, Any]]:
    data = pd.read_csv(input_dir / "shapley_torque_power_by_seed_event.csv")
    data["signal"] = data["joint"] + " " + data["K_channel"]
    signal_order = [
        f"J{joint:02d} {channel}"
        for joint in range(1, 9)
        for channel in ("K1", "K2")
    ]
    rows: list[dict[str, Any]] = []
    specifications = (
        (
            "shapley_clipped_torque_abs_mean",
            "Clip-aware Shapley active-torque magnitude by event bin",
            "P01_shapley_torque_by_event",
            "mean |Shapley torque contribution|",
            "magma",
        ),
        (
            "active_power_proxy_mean",
            "Signed active-control power proxy by event bin",
            "P02_power_proxy_by_event",
            "mean phi × theta_dot (proxy)",
            "RdBu_r",
        ),
        (
            "saturation_rate",
            "Joint-total torque saturation rate by event bin (same value on K1/K2 rows)",
            "P03_saturation_by_event",
            "fraction |u1+u2| ≥ 9",
            "viridis",
        ),
    )
    for index, (column, title, filename, colorbar_label, cmap) in enumerate(
        specifications, start=1
    ):
        matrix = (
            data.groupby(["signal", "event_bin"], as_index=False)[column]
            .median()
            .pivot(index="signal", columns="event_bin", values=column)
            .reindex(index=signal_order, columns=EVENT_ORDER)
        )
        fig, ax = plt.subplots(figsize=(11.5, 8.5))
        image = annotated_heatmap(
            ax,
            matrix.to_numpy(),
            [EVENT_SHORT[name] for name in EVENT_ORDER],
            signal_order,
            title,
            cmap=cmap,
            center_zero=(column == "active_power_proxy_mean"),
        )
        ax.set_xlabel("official-event-aligned analysis bin")
        ax.set_ylabel("joint/control channel")
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label=colorbar_label)
        paths = save_figure(fig, figure_dir / filename)
        for path in paths:
            rows.append(
                {
                    "figure_id": f"P{index:02d}",
                    "title": title,
                    "path": str(path),
                    "evidence_type": "control_boundary_decomposition",
                }
            )
    return rows


def validation_figure(input_dir: Path, figure_dir: Path) -> list[dict[str, Any]]:
    data = pd.read_csv(input_dir / "jacobian_validation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].hist(
        np.maximum(data["analytic_vs_autograd_max_abs"].to_numpy(), 1e-18),
        bins=24,
        color=BLUE,
        alpha=0.85,
    )
    axes[0].set_xscale("log")
    axes[0].set_title("Analytic vs autograd", loc="left", fontweight="bold")
    axes[0].set_xlabel("max absolute error per sample/joint")
    axes[0].set_ylabel("count")
    axes[1].hist(
        np.maximum(data["analytic_vs_finite_difference_max_abs"].to_numpy(), 1e-18),
        bins=24,
        color=ORANGE,
        alpha=0.85,
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Analytic vs central finite difference", loc="left", fontweight="bold")
    axes[1].set_xlabel("max absolute error per sample/joint")
    axes[1].set_ylabel("count")
    fig.suptitle("Independent Jacobian validation on randomly sampled visited states", fontweight="bold")
    paths = save_figure(fig, figure_dir / "V01_jacobian_validation")
    return [
        {
            "figure_id": "V01",
            "title": "Jacobian validation error distributions",
            "path": str(path),
            "evidence_type": "numerical_validation",
        }
        for path in paths
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    figure_dir = (
        args.output.resolve() if args.output is not None else input_dir / "figures"
    )
    required = (
        "ANALYSIS_MANIFEST.json",
        "JACOBIAN_VALIDATION_PASS.json",
        "jacobian_structural_mask_16x16.csv",
        "jacobian_full_16x16_by_seed_event.csv",
        "jacobian_local_summary_by_seed_event.csv",
        "shapley_torque_power_by_seed_event.csv",
        "jacobian_validation.csv",
    )
    missing = [name for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Analysis outputs missing: {missing}")
    configure_plotting()
    rows: list[dict[str, Any]] = []
    rows.extend(structural_mask_figure(input_dir, figure_dir))
    rows.extend(representative_full_jacobian_figure(input_dir, figure_dir))
    rows.extend(local_derivative_phase_figures(input_dir, figure_dir))
    rows.extend(physics_phase_figures(input_dir, figure_dir))
    rows.extend(validation_figure(input_dir, figure_dir))
    figure_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        figure_dir / "figure_manifest.csv",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
    )
    payload = {
        "schema": "obs2_v2_1_local_actor_figures/v1",
        "input": str(input_dir),
        "figure_dir": str(figure_dir),
        "figure_file_count": len(rows),
        "notes": [
            "Cross-joint zeros in J01/J02 are structural zeros of the local actor, not evidence of absent physical coupling.",
            "Power panels show phi times observed theta_dot, a control-boundary proxy rather than exact ten-substep energy.",
        ],
    }
    temporary = figure_dir / "FIGURE_MANIFEST.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(figure_dir / "FIGURE_MANIFEST.json")


if __name__ == "__main__":
    main()
