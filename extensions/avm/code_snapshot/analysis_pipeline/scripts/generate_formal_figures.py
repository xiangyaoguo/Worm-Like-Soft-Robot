from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import ARMS, CHECKPOINTS, RESET_SEEDS, RUNS, load_json, parse_bool, sha256_file, write_json


COLOUR = {"O1_sham": "#D2762C", "O2": "#2468A2"}
MARKER = {"O1_sham": "s", "O2": "o"}
LABEL = {"O1_sham": "O1-sham", "O2": "O2"}
GREY = "#5B6573"
LIGHT = "#D9E0E6"
RED = "#B64A4A"
GREEN = "#3B7D5B"


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def table(data_root: Path, name: str) -> pd.DataFrame:
    path = data_root / name
    if not path.exists():
        path = data_root / f"{name}.gz"
    return pd.read_csv(path)


def verify_receipt(data_root: Path, receipt_path: Path) -> dict:
    receipt = load_json(receipt_path)
    if receipt.get("status") != "pass":
        raise RuntimeError("Validation receipt is not a pass")
    if Path(receipt["data_root"]).resolve() != data_root.resolve():
        raise RuntimeError("Validation receipt belongs to another data root")
    for raw_path, expected in receipt["input_sha256"].items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Input changed after validation: {path}")
    return receipt


def save(fig, out: Path, stem: str, caption: str, index: list[dict]) -> None:
    paths = []
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    plt.close(fig)
    index.append({"id": stem, "caption": caption, "files": [str(path) for path in paths]})


def legend_handles():
    return [
        Line2D([0], [0], color=COLOUR[arm], marker=MARKER[arm], label=LABEL[arm], lw=1.8)
        for arm in ARMS
    ]


def as_success(frame: pd.DataFrame) -> pd.DataFrame:
    copy = frame.copy()
    copy["success_common"] = [int(parse_bool(value)) for value in copy["success_common"]]
    return copy


def r01_completeness(episodes, out, index):
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
    for ax, arm in zip(axes, ARMS):
        matrix = np.zeros((5, 15))
        for run in RUNS:
            for ci, checkpoint in enumerate(CHECKPOINTS):
                matrix[run, ci] = len(episodes[(episodes.arm == arm) & (episodes.paper_run == run) & (episodes.checkpoint == checkpoint)])
        image = ax.imshow(matrix, vmin=0, vmax=20, cmap="Blues", aspect="auto")
        for run in RUNS:
            for ci in range(15):
                ax.text(ci, run, f"{int(matrix[run, ci])}", ha="center", va="center", fontsize=7,
                        color="white" if matrix[run, ci] > 12 else "black")
        ax.set_yticks(RUNS, [f"run {run}" for run in RUNS])
        ax.set_title(LABEL[arm], loc="left", color=COLOUR[arm], fontweight="bold")
    axes[-1].set_xticks(range(15), CHECKPOINTS, rotation=45)
    axes[-1].set_xlabel("Training checkpoint")
    fig.colorbar(image, ax=axes, label="Evaluated resets", fraction=0.02)
    fig.suptitle("R01  Formal evaluation completeness (required: 20 per cell)", fontweight="bold")
    save(fig, out, "R01_evaluation_completeness", "Twenty frozen resets must be present for every arm, run and checkpoint.", index)


def r02_hash_audit(manifest, out, index):
    fields = ["actor_init_sha256", "critic_init_sha256", "optimizer_init_sha256", "torch_cpu_rng_sha256",
              "torch_cuda_rng_sha256", "numpy_rng_sha256", "python_rng_sha256"]
    values = np.ones((len(fields), 5))
    for fi, field in enumerate(fields):
        for run in RUNS:
            pair = manifest[manifest.paper_run == run].set_index("arm")
            values[fi, run] = float(str(pair.loc["O1_sham", field]) == str(pair.loc["O2", field]))
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.imshow(values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    for y in range(len(fields)):
        for x in RUNS:
            ax.text(x, y, "MATCH" if values[y, x] else "FAIL", ha="center", va="center", fontsize=8)
    ax.set_xticks(RUNS, [f"run {r}" for r in RUNS])
    ax.set_yticks(range(len(fields)), [f.replace("_sha256", "").replace("_", " ") for f in fields])
    ax.set_title("R02  Batch-0 pairing audit", fontweight="bold")
    save(fig, out, "R02_initialisation_pairing_audit", "Matched batch-0 actor, critic, optimizer and random-state hashes.", index)


def training_facets(training, value, ylabel, stem, title, out, index):
    fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
    for run, ax in zip(RUNS, axes):
        for arm in ARMS:
            sub = training[(training.arm == arm) & (training.paper_run == run)].sort_values("batch")
            raw = sub[value].astype(float)
            smooth = raw.rolling(50, min_periods=1).mean()
            ax.plot(sub.batch, raw, color=COLOUR[arm], alpha=.13, lw=.45)
            ax.plot(sub.batch, smooth, color=COLOUR[arm], lw=1.5, marker=None)
        ax.set_ylabel(f"run {run}")
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Training batch")
    fig.text(0.02, .5, ylabel, rotation=90, va="center")
    fig.legend(handles=legend_handles(), loc="upper right", frameon=False)
    fig.suptitle(title + " (thin: raw; thick: preregistered 50-batch mean)", fontweight="bold")
    save(fig, out, stem, title, index)


def r05_ppo(training, out, index):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.3), sharex=True)
    for arm in ARMS:
        grouped = training[training.arm == arm].groupby("batch")
        x = np.asarray(sorted(grouped.groups))
        kl = grouped.ppo_approx_kl.mean().reindex(x).to_numpy()
        updates = grouped.ppo_updates_completed.mean().reindex(x).to_numpy()
        axes[0].plot(x, pd.Series(kl).rolling(50, min_periods=1).mean(), color=COLOUR[arm], label=LABEL[arm])
        axes[1].plot(x, pd.Series(updates).rolling(50, min_periods=1).mean(), color=COLOUR[arm])
    axes[0].set_ylabel("Approximate KL")
    axes[1].set_ylabel("Completed PPO updates")
    axes[1].set_xlabel("Training batch")
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("R05  PPO optimization diagnostics (mean across five runs)", fontweight="bold")
    save(fig, out, "R05_ppo_diagnostics", "PPO approximate KL and completed updates; these are optimization diagnostics, not independent outcomes.", index)


def endpoint_counts(episodes):
    return episodes[episodes.checkpoint == 1500].groupby(["arm", "paper_run"]).success_common.sum().unstack("arm")


def r06_endpoint_paired(episodes, out, index):
    counts = endpoint_counts(episodes)
    fig, ax = plt.subplots(figsize=(6.7, 5.4))
    for run in RUNS:
        ys = [counts.loc[run, "O1_sham"], counts.loc[run, "O2"]]
        ax.plot([0, 1], ys, color=GREY, alpha=.65, lw=1.2)
        ax.scatter(0, ys[0], color=COLOUR["O1_sham"], marker="s", s=55, zorder=3)
        ax.scatter(1, ys[1], color=COLOUR["O2"], marker="o", s=55, zorder=3)
        ax.text(1.04, ys[1], f"run {run}", va="center", fontsize=8)
    ax.axhline(10, color=RED, ls="--", lw=1, label="run-level discovery threshold")
    ax.set_xticks([0, 1], ["O1-sham", "O2"])
    ax.set_ylabel("Common-criterion successes / 20")
    ax.set_ylim(-.7, 21.4)
    ax.set_title("R06  Primary endpoint: matched formal runs", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    save(fig, out, "R06_primary_endpoint_paired_success", "Checkpoint-1500 success counts; each connecting line is one matched training run.", index)


def r07_effect(episodes, out, index):
    counts = endpoint_counts(episodes)
    diff = (counts.O2 - counts.O1_sham) / 20.0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axvline(0, color=GREY, lw=1)
    for run in RUNS:
        ax.plot([0, diff.loc[run]], [run, run], color=COLOUR["O2"] if diff.loc[run] >= 0 else COLOUR["O1_sham"], lw=2)
        ax.scatter(diff.loc[run], run, color=COLOUR["O2"] if diff.loc[run] >= 0 else COLOUR["O1_sham"], s=55)
        ax.text(diff.loc[run], run + .17, f"{diff.loc[run]:+.2f}", ha="center", fontsize=8)
    ax.set_yticks(RUNS, [f"run {r}" for r in RUNS])
    ax.set_xlabel("Paired success-proportion difference (O2 − O1-sham)")
    ax.set_title("R07  Run-level endpoint effects", fontweight="bold")
    ax.grid(axis="x", alpha=.2)
    save(fig, out, "R07_endpoint_run_level_effects", "Five paired run-level effects; episode rows are nested and are not treated as n=100 independent replicates.", index)


def r08_outcome_matrix(episodes, out, index):
    endpoint = episodes[episodes.checkpoint == 1500]
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
    for ax, arm in zip(axes, ARMS):
        matrix = np.zeros((5, 20))
        for run in RUNS:
            sub = endpoint[(endpoint.arm == arm) & (endpoint.paper_run == run)].set_index("reset_seed")
            matrix[run] = sub.loc[list(RESET_SEEDS), "success_common"].to_numpy()
        ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        for y in RUNS:
            for x in range(20):
                ax.text(x, y, "●" if matrix[y, x] else "×", ha="center", va="center", fontsize=7)
        ax.set_yticks(RUNS, [f"run {r}" for r in RUNS])
        ax.set_title(LABEL[arm], color=COLOUR[arm], loc="left", fontweight="bold")
    axes[-1].set_xticks(range(20), range(1, 21))
    axes[-1].set_xlabel("Paired reset block")
    fig.suptitle("R08  All paired endpoint episode decisions", fontweight="bold")
    save(fig, out, "R08_endpoint_episode_outcome_matrix", "All checkpoint-1500 success/failure decisions arranged by matched run and reset.", index)


def r09_metrics(episodes, out, index):
    endpoint = episodes[episodes.checkpoint == 1500]
    metrics = [("desired_net_rotation_deg", "Desired net rotation (deg)", 360),
               ("desired_direction_fraction", "Direction fraction", .70),
               ("forward_body_lengths", "Forward displacement (body lengths)", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, (metric, label, threshold) in zip(axes, metrics):
        rng = np.random.default_rng(1701)
        for ai, arm in enumerate(ARMS):
            for run in RUNS:
                values = endpoint[(endpoint.arm == arm) & (endpoint.paper_run == run)][metric].astype(float).to_numpy()
                x = ai + (run - 2) * .045 + rng.normal(0, .009, len(values))
                ax.scatter(x, values, color=COLOUR[arm], marker=MARKER[arm], alpha=.20, s=12)
                ax.scatter(ai + (run - 2) * .045, np.mean(values), color=COLOUR[arm], marker=MARKER[arm], edgecolor="white", s=44)
        ax.axhline(threshold, color=RED, ls="--", lw=1)
        ax.set_xticks([0, 1], ["O1-sham", "O2"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("R09  Endpoint kinematic components (points: episodes; large marks: run means)", fontweight="bold")
    save(fig, out, "R09_endpoint_kinematic_components", "The three components of the common rolling criterion at checkpoint 1500.", index)


def r10_geometry(episodes, out, index):
    endpoint = episodes[episodes.checkpoint == 1500]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    size = np.clip(endpoint.forward_body_lengths.astype(float), 0, 5) * 15 + 8
    for ax, arm in zip(axes, ARMS):
        sub = endpoint[endpoint.arm == arm]
        s = size[sub.index]
        colours = np.where(sub.success_common.astype(int).to_numpy() == 1, GREEN, GREY)
        ax.scatter(sub.desired_net_rotation_deg, sub.desired_direction_fraction, s=s, c=colours, alpha=.55, edgecolor="none")
        ax.axvline(360, color=RED, ls="--", lw=1)
        ax.axhline(.70, color=RED, ls="--", lw=1)
        ax.set_title(LABEL[arm], color=COLOUR[arm], fontweight="bold")
        ax.set_xlabel("Desired net rotation (deg)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Desired-direction fraction")
    fig.suptitle("R10  Endpoint criterion geometry (marker area ∝ forward body lengths)", fontweight="bold")
    save(fig, out, "R10_endpoint_criterion_geometry", "Rotation-direction geometry; green marks meet all three common-criterion components.", index)


def checkpoint_counts(episodes):
    return episodes.groupby(["arm", "paper_run", "checkpoint"]).success_common.sum().rename("successes").reset_index()


def r11_checkpoint_paths(episodes, out, index):
    counts = checkpoint_counts(episodes)
    fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True, sharey=True)
    for run, ax in zip(RUNS, axes):
        for arm in ARMS:
            sub = counts[(counts.arm == arm) & (counts.paper_run == run)]
            ax.plot(sub.checkpoint, sub.successes, color=COLOUR[arm], marker=MARKER[arm], ms=3, label=LABEL[arm])
        ax.axhline(10, color=RED, ls="--", lw=.8)
        ax.set_ylabel(f"run {run}")
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Training checkpoint")
    fig.text(.02, .5, "Successes / 20", rotation=90, va="center")
    fig.legend(handles=legend_handles(), frameon=False, loc="upper right")
    fig.suptitle("R11  Rolling-discovery trajectories", fontweight="bold")
    save(fig, out, "R11_checkpoint_rolling_discovery", "Common-criterion success counts at all predeclared checkpoints.", index)


def r12_heatmaps(episodes, out, index):
    counts = checkpoint_counts(episodes)
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
    for ax, arm in zip(axes, ARMS):
        pivot = counts[counts.arm == arm].pivot(index="paper_run", columns="checkpoint", values="successes").loc[list(RUNS), list(CHECKPOINTS)]
        image = ax.imshow(pivot, vmin=0, vmax=20, cmap="viridis", aspect="auto")
        for y in RUNS:
            for x in range(15):
                value = int(pivot.iloc[y, x])
                ax.text(x, y, str(value), ha="center", va="center", fontsize=7, color="white" if value < 7 or value > 14 else "black")
        ax.set_yticks(RUNS, [f"run {r}" for r in RUNS])
        ax.set_title(LABEL[arm], loc="left", color=COLOUR[arm], fontweight="bold")
    axes[-1].set_xticks(range(15), CHECKPOINTS, rotation=45)
    axes[-1].set_xlabel("Training checkpoint")
    fig.colorbar(image, ax=axes, label="Successes / 20", fraction=.02)
    fig.suptitle("R12  Run-by-checkpoint rolling map", fontweight="bold")
    save(fig, out, "R12_checkpoint_success_heatmaps", "Arm-specific rolling success heat maps across all runs and checkpoints.", index)


def r13_discovery(episodes, out, index):
    counts = checkpoint_counts(episodes)
    records = []
    for arm in ARMS:
        for run in RUNS:
            sub = counts[(counts.arm == arm) & (counts.paper_run == run)].sort_values("checkpoint")
            passing = sub[sub.successes >= 10]
            first = float(passing.checkpoint.iloc[0]) if len(passing) else np.nan
            persistence = int((sub[sub.checkpoint >= first].successes >= 10).sum()) if np.isfinite(first) else 0
            records.append((arm, run, first, persistence))
    derived = pd.DataFrame(records, columns=["arm", "run", "first", "persistence"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for run in RUNS:
        pair = derived[derived.run == run].set_index("arm")
        for ax, metric in zip(axes, ("first", "persistence")):
            vals = [pair.loc[arm, metric] for arm in ARMS]
            plot_vals = [1600 if metric == "first" and not np.isfinite(v) else v for v in vals]
            ax.plot([0, 1], plot_vals, color=GREY, alpha=.6)
            for x, arm, value in zip([0, 1], ARMS, plot_vals):
                ax.scatter(x, value, color=COLOUR[arm], marker=MARKER[arm], s=46)
    axes[0].set_xticks([0, 1], ["O1-sham", "O2"])
    axes[0].set_yticks(list(CHECKPOINTS[::2]) + [1600], [str(v) for v in CHECKPOINTS[::2]] + ["not found"])
    axes[0].set_ylabel("First checkpoint with ≥10/20")
    axes[1].set_xticks([0, 1], ["O1-sham", "O2"])
    axes[1].set_ylabel("Passing checkpoints from first discovery onward")
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.suptitle("R13  Discovery timing and persistence", fontweight="bold")
    save(fig, out, "R13_discovery_timing_and_persistence", "First run-level discovery and subsequent persistence under the fixed ≥10/20 threshold.", index)


def sensitivity_counts(endpoint, arm=None, run=None):
    sub = endpoint.copy()
    if arm is not None: sub = sub[sub.arm == arm]
    if run is not None: sub = sub[sub.paper_run == run]
    output = {}
    for forward in (.5, 1.0, 1.5):
        matrix = np.zeros((3, 3))
        for yi, direction in enumerate((.6, .7, .8)):
            for xi, rotation in enumerate((270., 360., 450.)):
                matrix[yi, xi] = ((sub.desired_net_rotation_deg >= rotation)
                                  & (sub.desired_direction_fraction >= direction)
                                  & (sub.forward_body_lengths >= forward)).sum()
        output[forward] = matrix
    return output


def r14_sensitivity(episodes, out, index):
    endpoint = episodes[episodes.checkpoint == 1500]
    sets = {arm: sensitivity_counts(endpoint, arm=arm) for arm in ARMS}
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    rows = [("O1_sham", "O1-sham successes", 0, 100, "viridis"), ("O2", "O2 successes", 0, 100, "viridis"),
            ("diff", "O2 − O1-sham", -100, 100, "coolwarm")]
    for ri, (kind, row_label, vmin, vmax, cmap) in enumerate(rows):
        for ci, forward in enumerate((.5, 1.0, 1.5)):
            matrix = sets[kind][forward] if kind in sets else sets["O2"][forward] - sets["O1_sham"][forward]
            ax = axes[ri, ci]
            ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
            for y in range(3):
                for x in range(3): ax.text(x, y, f"{int(matrix[y, x]):+d}" if kind == "diff" else str(int(matrix[y, x])), ha="center", va="center", fontsize=8)
            ax.set_xticks(range(3), [270, 360, 450])
            ax.set_yticks(range(3), [.60, .70, .80])
            if ri == 2: ax.set_xlabel("Rotation threshold (deg)")
            if ci == 0: ax.set_ylabel(row_label + "\nDirection threshold")
            if ri == 0: ax.set_title(f"Forward ≥ {forward:g} BL")
    fig.suptitle("R14  Endpoint criterion-sensitivity cube", fontweight="bold")
    save(fig, out, "R14_aggregate_criterion_sensitivity", "All 27 prespecified kinematic threshold combinations for each arm and their difference.", index)


def r15_run_sensitivity(episodes, out, index):
    endpoint = episodes[episodes.checkpoint == 1500]
    fig, axes = plt.subplots(5, 3, figsize=(10, 13))
    for run in RUNS:
        a = sensitivity_counts(endpoint, "O1_sham", run)
        b = sensitivity_counts(endpoint, "O2", run)
        for ci, forward in enumerate((.5, 1.0, 1.5)):
            matrix = b[forward] - a[forward]
            ax = axes[run, ci]
            ax.imshow(matrix, vmin=-20, vmax=20, cmap="coolwarm", aspect="equal")
            for y in range(3):
                for x in range(3): ax.text(x, y, f"{int(matrix[y, x]):+d}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(3), [270, 360, 450])
            ax.set_yticks(range(3), [.60, .70, .80])
            if run == 0: ax.set_title(f"Forward ≥ {forward:g} BL")
            if ci == 0: ax.set_ylabel(f"run {run}\nDirection")
            if run == 4: ax.set_xlabel("Rotation (deg)")
    fig.suptitle("R15  Run-specific sensitivity: O2 minus O1-sham successes", fontweight="bold")
    save(fig, out, "R15_run_specific_criterion_sensitivity", "Run-specific differences across all 27 criterion variants.", index)


def r16_reward_alignment(training, episodes, out, index):
    counts = checkpoint_counts(episodes)
    selected = training[training.batch.isin(CHECKPOINTS)][["arm", "paper_run", "batch", "reward_mean"]].rename(columns={"batch": "checkpoint"})
    merged = counts.merge(selected, on=["arm", "paper_run", "checkpoint"], validate="one_to_one")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arm in ARMS:
        sub = merged[merged.arm == arm]
        ax.scatter(sub.reward_mean, sub.successes, c=COLOUR[arm], marker=MARKER[arm], alpha=.65, label=LABEL[arm])
    ax.axhline(10, color=RED, ls="--", lw=1)
    ax.set_xlabel("Logged HPR mean at checkpoint batch")
    ax.set_ylabel("Independent rolling successes / 20")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    ax.set_title("R16  Reward–behaviour alignment across checkpoints", fontweight="bold")
    save(fig, out, "R16_reward_behaviour_alignment", "Checkpoint HPR logging versus independently classified rolling outcomes.", index)


def heatmap_from_joint(joint, value, title, stem, out, index, vmin=None, vmax=None, cmap="magma"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        sub = joint[joint.arm == arm].groupby(["paper_run", "joint"])[value].mean().unstack("joint").loc[list(RUNS), range(1, 9)]
        image = ax.imshow(sub, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(8), [f"J{j}" for j in range(1, 9)])
        ax.set_yticks(RUNS, [f"run {r}" for r in RUNS])
        ax.set_title(LABEL[arm], color=COLOUR[arm], fontweight="bold")
    fig.colorbar(image, ax=axes, fraction=.025)
    fig.suptitle(title, fontweight="bold")
    save(fig, out, stem, title + " (predeclared representative reset).", index)


def r17_gains(joint, out, index):
    copy = joint.copy()
    copy["abs_k1"] = copy.k1.abs()
    copy["abs_k2"] = copy.k2.abs()
    heatmap_from_joint(copy, "abs_k1", "R17a  Mean absolute K1 by run and joint", "R17a_jointwise_abs_k1", out, index)
    heatmap_from_joint(copy, "abs_k2", "R17b  Mean absolute K2 by run and joint", "R17b_jointwise_abs_k2", out, index)


def r18_saturation(joint, out, index):
    copy = joint.copy()
    copy["sat"] = [int(parse_bool(v)) for v in copy.torque_saturated]
    heatmap_from_joint(copy, "sat", "R18  Torque saturation fraction by run and joint", "R18_torque_saturation_maps", out, index, 0, 1, "viridis")


def r19_components(joint, out, index):
    copy = joint.copy()
    copy["abs_k1_component"] = copy.k1_component.abs()
    copy["abs_k2_component"] = copy.k2_component.abs()
    grouped = copy.groupby(["arm", "paper_run", "joint"])[["abs_k1_component", "abs_k2_component"]].mean().reset_index()
    grouped["k2_share"] = grouped.abs_k2_component / (grouped.abs_k1_component + grouped.abs_k2_component + 1e-12)
    heatmap_from_joint(grouped, "k2_share", "R19  Fraction of controller magnitude attributable to K2 angular-velocity feedback", "R19_controller_component_balance", out, index, 0, 1, "cividis")


def r20_timeseries(joint, out, index):
    for run in RUNS:
        fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
        for arm in ARMS:
            sub = joint[(joint.arm == arm) & (joint.paper_run == run)]
            global_step = sub.groupby("step").first()
            components = sub.groupby("step")[["k1_component", "k2_component"]].apply(lambda x: x.abs().mean())
            saturation = sub.groupby("step").torque_saturated.apply(lambda s: np.mean([parse_bool(v) for v in s]))
            axes[0].plot(global_step.time_s, global_step.body_rotation_deg, color=COLOUR[arm], label=LABEL[arm])
            axes[1].plot(global_step.time_s, global_step.forward_body_lengths, color=COLOUR[arm])
            axes[2].plot(global_step.time_s, global_step.direction_fraction, color=COLOUR[arm])
            axes[3].plot(global_step.time_s, saturation, color=COLOUR[arm])
        axes[0].set_ylabel("Body rotation (deg)")
        axes[1].set_ylabel("Forward (BL)")
        axes[2].set_ylabel("Direction fraction")
        axes[3].set_ylabel("Torque saturated\n(joint fraction)")
        axes[3].set_xlabel("Time (s)")
        for ax in axes: ax.grid(alpha=.2)
        axes[0].legend(frameon=False)
        fig.suptitle(f"R20  Representative paired time series — run {run}, paired reset 1", fontweight="bold")
        save(fig, out, f"R20_run{run}_representative_time_series", f"Representative checkpoint-1500 time series for run {run} at predeclared paired reset 1.", index)


def r21_com(joint, out, index):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.7))
    for run, ax in zip(RUNS, axes):
        for arm in ARMS:
            sub = joint[(joint.arm == arm) & (joint.paper_run == run)].groupby("step").first()
            ax.plot(sub.com_x - sub.com_x.iloc[0], sub.com_y - sub.com_y.iloc[0], color=COLOUR[arm], label=LABEL[arm])
            ax.scatter([0], [0], color=COLOUR[arm], marker=MARKER[arm], s=20)
        ax.set_title(f"run {run}")
        ax.set_xlabel("Δx")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Δy")
    axes[-1].legend(handles=legend_handles(), frameon=False)
    fig.suptitle("R21  Representative centre-of-mass trajectories", fontweight="bold")
    save(fig, out, "R21_representative_com_trajectories", "Paired checkpoint-1500 COM trajectories for the predeclared reset.", index)


def r22_morphology(node, out, index):
    selected_steps = np.linspace(0, 999, 10).round().astype(int)
    for run in RUNS:
        fig, axes = plt.subplots(2, 10, figsize=(18, 4.2), sharex=False, sharey=False)
        for ri, arm in enumerate(ARMS):
            sub = node[(node.arm == arm) & (node.paper_run == run)]
            for ci, step in enumerate(selected_steps):
                frame = sub[sub.step == step].sort_values("node")
                x = frame.x.to_numpy() - frame.x.mean()
                y = frame.y.to_numpy() - frame.ground_height.to_numpy()
                ax = axes[ri, ci]
                ax.plot(x, y, color=COLOUR[arm], marker=MARKER[arm], ms=3, lw=1.4)
                ax.axhline(0, color=GREY, lw=.7)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0: ax.set_title(f"t={step}", fontsize=8)
                if ci == 0: ax.set_ylabel(LABEL[arm], color=COLOUR[arm], fontweight="bold")
        fig.suptitle(f"R22  Morphology montage — run {run}, checkpoint 1500, paired reset 1", fontweight="bold")
        save(fig, out, f"R22_run{run}_morphology_montage", f"Node-data morphology montage for run {run} at ten fixed normalized times.", index)


def r23_probe(probe, out, index):
    spatial_values = sorted(probe.spatial_difference.unique())
    chosen = min(spatial_values, key=lambda value: abs(float(value)))
    sub = probe[np.isclose(probe.spatial_difference, chosen)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for arm in ARMS:
        grouped = sub[sub.arm == arm].groupby("angular_velocity")
        x = np.array(sorted(grouped.groups), dtype=float)
        for ax, metric, label in zip(axes, ("k1", "k2"), ("Actor output K1", "Actor output K2")):
            med = grouped[metric].median().reindex(x).to_numpy()
            q1 = grouped[metric].quantile(.25).reindex(x).to_numpy()
            q3 = grouped[metric].quantile(.75).reindex(x).to_numpy()
            ax.plot(x, med, color=COLOUR[arm], marker=MARKER[arm], label=LABEL[arm])
            ax.fill_between(x, q1, q3, color=COLOUR[arm], alpha=.15)
            ax.set_ylabel(label)
            ax.grid(alpha=.2)
    for ax in axes: ax.set_xlabel("Probed angular velocity")
    axes[0].legend(frameon=False)
    fig.suptitle(f"R23  Actor angular-velocity probe at spatial difference {chosen:g} (median and IQR)", fontweight="bold")
    save(fig, out, "R23_actor_angular_velocity_probe", "Checkpoint-1500 actor output sensitivity to angular velocity on the frozen probe grid.", index)


def r24_sham(probe, out, index):
    sham = probe[probe.arm == "O1_sham"]
    ranges = sham.groupby(["paper_run", "joint", "spatial_difference"])[["k1", "k2"]].agg(lambda s: s.max() - s.min()).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, metric in zip(axes, ("k1", "k2")):
        for run in RUNS:
            sub = ranges[ranges.paper_run == run]
            ax.scatter(sub.joint + (run - 2) * .035, sub[metric], s=18, label=f"run {run}")
        ax.axhline(1e-8, color=RED, ls="--", lw=1)
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_xlabel("Joint")
        ax.set_title(f"Range of {metric.upper()} across angular-velocity sweep")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Output range (must be ≤1e-8)")
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("R24  O1-sham actor-invariance audit", fontweight="bold")
    save(fig, out, "R24_o1_sham_invariance_audit", "O1-sham outputs must be invariant to the masked angular-velocity probe coordinate.", index)


def r25_signflip(episodes, out, index):
    counts = endpoint_counts(episodes)
    effects = ((counts.O2 - counts.O1_sham) / 20.0).to_numpy()
    permutations = np.array([np.mean(effects * np.array(signs)) for signs in itertools.product([-1, 1], repeat=5)])
    observed = effects.mean()
    p = np.mean(np.abs(permutations) >= abs(observed) - 1e-15)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    unique, frequencies = np.unique(np.round(permutations, 12), return_counts=True)
    ax.vlines(unique, 0, frequencies, color=GREY, lw=2)
    ax.scatter(unique, frequencies, color=GREY, s=22)
    ax.axvline(observed, color=RED, lw=2, label=f"observed mean={observed:+.3f}; two-sided p={p:.4f}")
    ax.set_xlabel("Mean paired effect under all 32 sign flips")
    ax.set_ylabel("Multiplicity")
    ax.legend(frameon=False)
    ax.set_title("R25  Exact five-pair randomisation distribution", fontweight="bold")
    save(fig, out, "R25_exact_sign_flip_distribution", "Exact sign-flip distribution over five matched run-level effects; inferential resolution is limited by n=5.", index)


def generate(data_root: Path, receipt_path: Path, out: Path) -> dict:
    receipt = verify_receipt(data_root, receipt_path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Result figure directory must be empty: {out}")
    temp = out.parent / f".{out.name}.building"
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir(parents=True)
    setup()
    manifest = table(data_root, "run_manifest.csv")
    training = table(data_root, "training_metrics.csv")
    episodes = as_success(table(data_root, "checkpoint_episode_metrics.csv"))
    joint = table(data_root, "trajectory_joint.csv")
    node = table(data_root, "trajectory_node.csv")
    probe = table(data_root, "actor_probe.csv")
    for frame in (manifest, training, episodes, joint, node, probe):
        frame["paper_run"] = frame.paper_run.astype(int)
    episodes["checkpoint"] = episodes.checkpoint.astype(int)
    training["batch"] = training.batch.astype(int)
    index = []
    r01_completeness(episodes, temp, index)
    r02_hash_audit(manifest, temp, index)
    training_facets(training, "reward_mean", "Logged HPR mean", "R03_training_hpr_trajectories", "R03  HPR training trajectories", temp, index)
    training_facets(training, "speed_mean", "Logged speed mean", "R04_training_speed_trajectories", "R04  Speed training trajectories", temp, index)
    r05_ppo(training, temp, index)
    r06_endpoint_paired(episodes, temp, index)
    r07_effect(episodes, temp, index)
    r08_outcome_matrix(episodes, temp, index)
    r09_metrics(episodes, temp, index)
    r10_geometry(episodes, temp, index)
    r11_checkpoint_paths(episodes, temp, index)
    r12_heatmaps(episodes, temp, index)
    r13_discovery(episodes, temp, index)
    r14_sensitivity(episodes, temp, index)
    r15_run_sensitivity(episodes, temp, index)
    r16_reward_alignment(training, episodes, temp, index)
    r17_gains(joint, temp, index)
    r18_saturation(joint, temp, index)
    r19_components(joint, temp, index)
    r20_timeseries(joint, temp, index)
    r21_com(joint, temp, index)
    r22_morphology(node, temp, index)
    r23_probe(probe, temp, index)
    r24_sham(probe, temp, index)
    r25_signflip(episodes, temp, index)
    for entry in index:
        entry["sha256"] = {Path(path).suffix.lstrip("."): sha256_file(Path(path)) for path in entry["files"]}
    result = {
        "schema": "o1_o2_formal_figure_index/v1",
        "status": "complete",
        "validation_receipt": str(receipt_path),
        "validation_receipt_sha256": sha256_file(receipt_path),
        "figure_count": len(index),
        "files_count": 2 * len(index),
        "figures": index,
    }
    write_json(temp / "FIGURE_INDEX.json", result)
    if out.exists(): out.rmdir()
    temp.replace(out)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.data_root.resolve(), args.receipt.resolve(), args.out.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
