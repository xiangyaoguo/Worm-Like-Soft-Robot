from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import sha256_file, write_json


BLUE = "#2468A2"
ORANGE = "#D2762C"
GREEN = "#3B7D5B"
GREY = "#5B6573"
LIGHT = "#E9EEF3"
RED = "#B64A4A"


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def box(ax, xy, width, height, text, edge, face="white", fontsize=9, lw=1.4):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, colour=GREY, style="-|>", lw=1.4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, color=colour, lw=lw))


def save(fig, out: Path, stem: str) -> list[Path]:
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def study_design(out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.0, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Formal capacity-matched observation ablation: design and data provenance", fontweight="bold", pad=14)

    box(ax, (0.04, 0.76), 0.27, 0.14, "Five archived formal O2-HPR runs\n(run 0–4)", BLUE, "#EAF3FA")
    box(ax, (0.365, 0.76), 0.27, 0.14, "Frozen extension protocol\nbefore O1-sham training", GREY, LIGHT)
    box(ax, (0.69, 0.76), 0.27, 0.14, "New formal O1-sham-HPR\nmatched run 0–4", ORANGE, "#FFF1E5")
    arrow(ax, (0.31, 0.83), (0.365, 0.83))
    arrow(ax, (0.635, 0.83), (0.69, 0.83))

    box(ax, (0.04, 0.48), 0.27, 0.17, "Full actor observation\n[s, angular velocity]\n15 M steps; checkpoint 100–1500", BLUE, "#EAF3FA")
    box(ax, (0.365, 0.48), 0.27, 0.17, "Matched initialization gate\nactor + critic + optimizer + RNG\nall hashes must match", RED, "#FBECEC")
    box(ax, (0.69, 0.48), 0.27, 0.17, "Actor-only sham observation\n[s, 0]\n15 M steps; checkpoint 100–1500", ORANGE, "#FFF1E5")
    arrow(ax, (0.175, 0.76), (0.175, 0.65))
    arrow(ax, (0.825, 0.76), (0.825, 0.65))
    arrow(ax, (0.31, 0.565), (0.365, 0.565), RED)
    arrow(ax, (0.69, 0.565), (0.635, 0.565), RED)

    box(ax, (0.17, 0.20), 0.66, 0.16, "One frozen deterministic evaluator\n20 paired resets × 15 checkpoints × 2 arms × 5 runs\nCommon criterion: rotation ≥360°, direction fraction ≥0.70, forward ≥1 body length", GREEN, "#EAF5EF")
    arrow(ax, (0.175, 0.48), (0.36, 0.36), BLUE)
    arrow(ax, (0.825, 0.48), (0.64, 0.36), ORANGE)
    box(ax, (0.28, 0.035), 0.44, 0.09, "Primary unit: matched independent training run (n = 5 pairs)", GREY, "white", fontsize=9)
    arrow(ax, (0.50, 0.20), (0.50, 0.125), GREEN)
    fig.text(0.02, 0.01, "Paper-facing labels are run 0–4; internal seed values remain in reproducibility manifests only.", fontsize=8, color=GREY)
    return save(fig, out, "P00_study_design_and_provenance")


def architecture(out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Capacity-matched actor-information ablation with an invariant physical controller", fontweight="bold", pad=14)

    box(ax, (0.04, 0.74), 0.20, 0.13, "Environment state\nspatial difference s\nangular velocity", GREY, LIGHT)
    box(ax, (0.32, 0.77), 0.22, 0.10, "O2 actor input\n[s, angular velocity]", BLUE, "#EAF3FA")
    box(ax, (0.32, 0.57), 0.22, 0.10, "O1-sham actor input\n[s, 0]", ORANGE, "#FFF1E5")
    arrow(ax, (0.24, 0.805), (0.32, 0.82), BLUE)
    arrow(ax, (0.24, 0.785), (0.32, 0.62), ORANGE)
    ax.text(0.275, 0.675, "actor-only mask", ha="center", va="center", fontsize=8, color=ORANGE)

    box(ax, (0.59, 0.67), 0.25, 0.15, "Eight independently parameterised\njoint-specific actor networks\n(same 2-D input shape and parameters)", GREY, "white", fontsize=8.3)
    arrow(ax, (0.54, 0.82), (0.59, 0.77), BLUE)
    arrow(ax, (0.54, 0.62), (0.59, 0.71), ORANGE)
    box(ax, (0.87, 0.69), 0.10, 0.11, "K1, K2\nper joint", GREEN, "#EAF5EF")
    arrow(ax, (0.84, 0.745), (0.87, 0.745), GREEN)

    box(ax, (0.32, 0.34), 0.22, 0.11, "Shared centralised critic\nfull O2 in both arms", BLUE, "#EAF3FA")
    arrow(ax, (0.14, 0.74), (0.32, 0.395), BLUE)
    box(ax, (0.66, 0.30), 0.30, 0.15, "Physical torque (unchanged)\nclip(K1*s + K2*angular velocity, -9, 9)", GREEN, "#EAF5EF")
    arrow(ax, (0.92, 0.69), (0.86, 0.45), GREEN)
    arrow(ax, (0.24, 0.78), (0.66, 0.37), GREY)

    box(ax, (0.15, 0.08), 0.70, 0.11, "Only the actor's access to current angular velocity changes.\nCritic information, actor capacity, K2 output and K2 physical feedback remain active.", RED, "#FBECEC", fontsize=10)
    arrow(ax, (0.50, 0.30), (0.50, 0.19), RED)
    return save(fig, out, "P01_capacity_matched_observation_ablation")


def evidence_hierarchy(out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Predeclared evidence hierarchy and fail-closed gates", fontweight="bold", pad=14)
    levels = [
        (0.08, 0.82, 0.84, 0.10, "1  Protocol freeze", "study contract, seed map, code snapshot, evaluator thresholds", GREY, LIGHT),
        (0.11, 0.67, 0.78, 0.10, "2  Initialization gate", "paired actor, critic, optimizer and RNG hashes; capacity equality", RED, "#FBECEC"),
        (0.14, 0.52, 0.72, 0.10, "3  Execution gate", "five complete O1-sham runs; 15 M steps; checkpoints 100–1500", ORANGE, "#FFF1E5"),
        (0.17, 0.37, 0.66, 0.10, "4  Frozen evaluation", "20 paired resets per checkpoint; deterministic policy; common criterion", BLUE, "#EAF3FA"),
        (0.20, 0.22, 0.60, 0.10, "5  Primary analysis", "five run-level endpoint effects; no episode-level pseudoreplication", GREEN, "#EAF5EF"),
        (0.23, 0.07, 0.54, 0.10, "6  Secondary evidence", "checkpoint, sensitivity, gains, torque, trajectory and actor-probe analyses", GREY, "white"),
    ]
    for idx, (x, y, w, h, title, detail, edge, face) in enumerate(levels):
        box(ax, (x, y), w, h, f"{title}\n{detail}", edge, face, fontsize=9.5)
        if idx < len(levels) - 1:
            next_x, next_y, next_w, next_h, *_ = levels[idx + 1]
            arrow(ax, (0.5, y), (0.5, next_y + next_h), GREY)
    ax.text(0.50, 0.015, "Any failed gate → stop → no result figures", ha="center", va="center", color=RED, fontweight="bold")
    return save(fig, out, "P02_evidence_hierarchy_and_gates")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"Protocol output directory must be empty: {out}")
    setup()
    files = []
    files.extend(study_design(out))
    files.extend(architecture(out))
    files.extend(evidence_hierarchy(out))
    manifest = {
        "schema": "o1_o2_protocol_figure_manifest/v1",
        "status": "protocol_only_no_outcomes",
        "files": [{"path": str(path), "sha256": sha256_file(path)} for path in files],
    }
    write_json(out / "PROTOCOL_FIGURE_MANIFEST.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
