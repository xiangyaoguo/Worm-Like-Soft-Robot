from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(r"C:\Users\PUBLIC_USER\Documents\Graduate_Thesis_Project")
ATLAS = (
    ROOT
    / "05_Experimental_Data_and_Code"
    / "formal10_initial3_k1k2_atlas_20260804"
    / "data"
    / "all_policy_surfaces_float32.npy"
)
TRACE_ROOT = ROOT / "formal_r2_k_analysis_20260804" / "traces"
HPR_TRACE = TRACE_ROOT / "formal__seed9201__R0__eval_seed20264101.npz"
SGRR_TRACE = TRACE_ROOT / "formal__seed9201__Rroll__eval_seed20264101.npz"
OUT_DIR = (
    ROOT
    / "05_Experimental_Data_and_Code"
    / "hpr_sgrr_run0_shared_scale_response_surfaces_20260826"
)

OUTPUT_STEM = "Figure_HPR_SGRR_run0_K1_K2_shared_scale"
PNG_PATH = OUT_DIR / f"{OUTPUT_STEM}.png"
PDF_PATH = OUT_DIR / f"{OUTPUT_STEM}.pdf"
SVG_PATH = OUT_DIR / f"{OUTPUT_STEM}.svg"
DATA_PATH = OUT_DIR / "Figure_HPR_SGRR_run0_K1_K2_selected_data.npz"
MANIFEST_PATH = OUT_DIR / "Figure_HPR_SGRR_run0_K1_K2_manifest.json"

RUN_INDEX = {"HPR-O2-JS run 0": 0, "SGRR-O2-JS run 0": 1}
CHECKPOINT_INDEX = 14  # checkpoint 1500 in 100:100:1500
CHANNEL_NAMES = (r"$K_1$", r"$K_2$")
JOINTS = tuple(f"J{i:02d}" for i in range(1, 9))
GRID_SIZE = 101
DELTA_THETA = np.linspace(-2.0 * math.pi, 2.0 * math.pi, GRID_SIZE)
THETA_DOT = np.linspace(-10.0, 10.0, GRID_SIZE)
TRACE_STRIDE = 5  # Preserve the sampling convention used by the current Figure 4.4.

CHECKPOINT_SHA256 = {
    "HPR-O2-JS run 0": "B697425FFA994CCC4CE32DB29573E6F87576C3C16CE39EA35A50A38A183E7EA6",
    "SGRR-O2-JS run 0": "B2AE292C1E55A317EA29911CFCF67CF28DB1CC82D5F5E36611C5BF4CA031BF34",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest().upper()


def configure_plotting() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 9.0,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "axes.unicode_minus": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not ATLAS.is_file():
        raise FileNotFoundError(ATLAS)
    for trace_path in (HPR_TRACE, SGRR_TRACE):
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)

    atlas = np.load(ATLAS, mmap_mode="r")
    expected_shape = (13, 15, 8, 2, GRID_SIZE, GRID_SIZE)
    if atlas.shape != expected_shape or atlas.dtype != np.float32:
        raise RuntimeError(
            f"Unexpected atlas contract: shape={atlas.shape}, dtype={atlas.dtype}; "
            f"expected {expected_shape}, float32"
        )

    # [policy row, joint, channel, delta_theta, theta_dot]
    surfaces = np.asarray(
        atlas[[RUN_INDEX["HPR-O2-JS run 0"], RUN_INDEX["SGRR-O2-JS run 0"]], CHECKPOINT_INDEX],
        dtype=np.float32,
    )
    with np.load(HPR_TRACE) as payload:
        hpr_observation = np.asarray(payload["observation"], dtype=np.float32)
    with np.load(SGRR_TRACE) as payload:
        sgrr_observation = np.asarray(payload["observation"], dtype=np.float32)
    observations = np.stack((hpr_observation, sgrr_observation), axis=0)

    if surfaces.shape != (2, 8, 2, GRID_SIZE, GRID_SIZE):
        raise RuntimeError(f"Unexpected selected surface shape: {surfaces.shape}")
    if observations.shape != (2, 1000, 8, 2):
        raise RuntimeError(f"Unexpected observation shape: {observations.shape}")
    if not np.isfinite(surfaces).all() or not np.isfinite(observations).all():
        raise RuntimeError("Selected inputs contain non-finite values")
    return surfaces, observations, atlas


def draw_figure(surfaces: np.ndarray, observations: np.ndarray) -> tuple[plt.Figure, list[float], list[int]]:
    configure_plotting()
    fig = plt.figure(figsize=(16.2, 8.8), constrained_layout=False)
    outer = fig.add_gridspec(
        2,
        1,
        left=0.025,
        right=0.975,
        bottom=0.070,
        top=0.920,
        hspace=0.48,
    )

    limits: list[float] = []
    visible_counts: list[int] = []
    row_names = ("HPR-O2-JS\nrun 0", "SGRR-O2-JS\nrun 0")

    for channel_index, channel_name in enumerate(CHANNEL_NAMES):
        panel_values = surfaces[:, :, channel_index]
        limit = float(np.max(np.abs(panel_values.astype(np.float64))))
        if not np.isfinite(limit) or limit <= 0.0:
            raise RuntimeError(f"Invalid shared colour limit for channel {channel_index}: {limit}")
        limits.append(limit)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

        panel_grid = outer[channel_index].subgridspec(
            2,
            10,
            width_ratios=[1.15, 1, 1, 1, 1, 1, 1, 1, 1, 0.10],
            wspace=0.16,
            hspace=0.10,
        )
        heat_axes: list[list[plt.Axes]] = [[], []]
        image = None

        for policy_index, row_name in enumerate(row_names):
            label_ax = fig.add_subplot(panel_grid[policy_index, 0])
            label_ax.axis("off")
            label_ax.text(
                0.98,
                0.50,
                row_name,
                ha="right",
                va="center",
                fontsize=9.2,
                linespacing=1.12,
            )

            for joint_index, joint_name in enumerate(JOINTS):
                ax = fig.add_subplot(panel_grid[policy_index, joint_index + 1])
                heat_axes[policy_index].append(ax)
                image = ax.imshow(
                    panel_values[policy_index, joint_index],
                    origin="lower",
                    aspect="auto",
                    interpolation="bilinear",
                    extent=[THETA_DOT[0], THETA_DOT[-1], DELTA_THETA[0], DELTA_THETA[-1]],
                    cmap="RdBu_r",
                    norm=norm,
                    rasterized=True,
                )

                points = observations[policy_index, ::TRACE_STRIDE, joint_index]
                inside = (
                    (points[:, 0] >= DELTA_THETA[0])
                    & (points[:, 0] <= DELTA_THETA[-1])
                    & (points[:, 1] >= THETA_DOT[0])
                    & (points[:, 1] <= THETA_DOT[-1])
                )
                selected = points[inside]
                visible_counts.append(int(selected.shape[0]))
                ax.scatter(
                    selected[:, 1],
                    selected[:, 0],
                    s=3.3,
                    c="white",
                    edgecolors="#4a4a4a",
                    alpha=0.66,
                    linewidths=0.12,
                    rasterized=True,
                    zorder=3,
                )

                ax.set_xlim(float(THETA_DOT[0]), float(THETA_DOT[-1]))
                ax.set_ylim(float(DELTA_THETA[0]), float(DELTA_THETA[-1]))
                ax.set_xticks((-10.0, 0.0, 10.0))
                ax.set_yticks((-5.0, 0.0, 5.0))
                ax.tick_params(axis="both", labelsize=7.4, width=0.75, length=3.0, pad=1.5)
                if policy_index == 0:
                    ax.tick_params(labelbottom=False)
                if joint_index != 0:
                    ax.tick_params(labelleft=False)
                if policy_index == 0:
                    ax.set_title(joint_name, fontsize=9.6, fontweight="bold", pad=3.0)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_linewidth(0.75)
                    ax.spines[side].set_color("#202020")

        if image is None:
            raise RuntimeError("No heat map was drawn")
        cax = fig.add_subplot(panel_grid[:, 9])
        colorbar = fig.colorbar(image, cax=cax)
        ticks = np.linspace(-limit, limit, 5)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{value:.0f}" for value in ticks])
        colorbar.ax.tick_params(labelsize=7.8, width=0.7, length=3.0)
        colorbar.outline.set_linewidth(0.7)
        colorbar.set_label(
            rf"Unclipped {channel_name} (simulator units)",
            fontsize=8.7,
            labelpad=5.5,
        )

        panel_bbox = outer[channel_index].get_position(fig)
        first_heat_bbox = heat_axes[0][0].get_position(fig)
        last_heat_bbox = heat_axes[0][-1].get_position(fig)
        fig.text(
            panel_bbox.x0,
            panel_bbox.y1 + 0.023,
            f"({'a' if channel_index == 0 else 'b'})  {channel_name} response surfaces",
            ha="left",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
        )
        fig.text(
            first_heat_bbox.x0 - 0.020,
            (panel_bbox.y0 + panel_bbox.y1) / 2.0,
            r"$s_i$ (rad)",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9.1,
        )
        fig.text(
            (first_heat_bbox.x0 + last_heat_bbox.x1) / 2.0,
            panel_bbox.y0 - 0.037,
            r"$\dot{\theta}_i$ (rad s$^{-1}$)",
            ha="center",
            va="top",
            fontsize=9.1,
        )

    return fig, limits, visible_counts


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    surfaces, observations, _ = load_inputs()

    # Preserve the exact selected inputs so the plotted figure can be audited independently.
    np.savez_compressed(
        DATA_PATH,
        delta_theta=DELTA_THETA.astype(np.float32),
        theta_dot=THETA_DOT.astype(np.float32),
        hpr_surfaces=surfaces[0],
        sgrr_surfaces=surfaces[1],
        hpr_observation=observations[0],
        sgrr_observation=observations[1],
    )

    fig, limits, visible_counts = draw_figure(surfaces, observations)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(PDF_PATH, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(SVG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    per_channel_ranges = {}
    for channel_index, channel_plain in enumerate(("K1", "K2")):
        values = surfaces[:, :, channel_index].astype(np.float64)
        per_channel_ranges[channel_plain] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "shared_symmetric_colour_limit": limits[channel_index],
            "display_clipping_fraction": float(np.mean(np.abs(values) > limits[channel_index])),
        }

    manifest = {
        "schema": "hpr_sgrr_run0_shared_scale_response_surfaces/v1",
        "figure": {
            "description": "Two stacked panels; each panel contains HPR and SGRR rows across J01-J08.",
            "checkpoint": 1500,
            "training_seed": 9201,
            "paper_run_index": 0,
            "evaluation_reset_seed": 20264101,
            "deterministic_output": "IndependentNormal loc",
            "gain_mapping": "K1=100*loc[0], K2=100*loc[1]",
            "gain_clipping": "none",
            "physical_torque_note": "The later active-torque computation is clipped to [-9, 9]; that clipping is not applied to this figure.",
            "grid": {
                "delta_theta": [float(DELTA_THETA[0]), float(DELTA_THETA[-1])],
                "theta_dot": [float(THETA_DOT[0]), float(THETA_DOT[-1])],
                "points_per_axis": GRID_SIZE,
            },
            "colour_scale": "One full-range, zero-centred symmetric scale shared by all 16 mini-plots within each channel panel.",
            "ranges": per_channel_ranges,
            "trace_stride": TRACE_STRIDE,
            "visible_trace_point_counts_in_plot_order": visible_counts,
        },
        "inputs": {
            "surface_atlas": {"path": str(ATLAS), "sha256": sha256_file(ATLAS)},
            "hpr_trace": {"path": str(HPR_TRACE), "sha256": sha256_file(HPR_TRACE)},
            "sgrr_trace": {"path": str(SGRR_TRACE), "sha256": sha256_file(SGRR_TRACE)},
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "atlas_indices": {
                "HPR-O2-JS run 0": [0, CHECKPOINT_INDEX],
                "SGRR-O2-JS run 0": [1, CHECKPOINT_INDEX],
            },
        },
        "outputs": {
            "png": {"path": str(PNG_PATH), "sha256": sha256_file(PNG_PATH)},
            "pdf": {"path": str(PDF_PATH), "sha256": sha256_file(PDF_PATH)},
            "svg": {"path": str(SVG_PATH), "sha256": sha256_file(SVG_PATH)},
            "selected_data": {"path": str(DATA_PATH), "sha256": sha256_file(DATA_PATH)},
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest["figure"], indent=2, ensure_ascii=False))
    print(f"PNG: {PNG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"SVG: {SVG_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
