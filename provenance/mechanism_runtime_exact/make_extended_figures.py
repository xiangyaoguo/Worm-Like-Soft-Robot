"""Create extended post-result descriptive figures from sealed analysis tables.

This module deliberately reads only the completed analysis manifest and the two
analysis CSV tables named in ``INPUT_NAMES``.  It never opens raw ``results/``
files.  The figures are post-result descriptive visualizations and provide no
independent inferential evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from condition_matrix import build_conditions, canonical_sha256


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
FIGURE_DIR = ANALYSIS / "extended_figures"
MANIFEST_PATH = ANALYSIS / "EXTENDED_FIGURES_MANIFEST.json"
ANALYZE_SOURCE = ROOT / "analyze_results.py"
CONDITION_SOURCE = ROOT / "condition_matrix.py"
IMPLEMENTATION_STATUS = "post_result_transparent_implementation"
INPUT_NAMES = (
    "analysis/episode_results.csv",
    "analysis/condition_seed_summary.csv",
    "analysis/analysis_manifest.json",
)
JOINTS = tuple(f"J{index:02d}" for index in range(1, 9))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FIGURE_NOTE = (
    "Post-result descriptive visualization only; not independent inference."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str), f"{label} must be a SHA-256 string")
    digest = value.lower()
    require(SHA256_RE.fullmatch(digest) is not None, f"Invalid SHA-256 at {label}")
    return digest


def require_regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"Missing/non-regular {label}: {path}")
    try:
        path.resolve().relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes study root: {path}") from error


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"Expected JSON object: {path}")
    return payload


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV has no header: {path}")
        fields = list(reader.fieldnames)
        require(len(fields) == len(set(fields)), f"Duplicate CSV header fields: {path}")
        rows = [dict(row) for row in reader]
    return fields, rows


def require_fields(fields: Iterable[str], expected: Iterable[str], label: str) -> None:
    missing = sorted(set(expected) - set(fields))
    require(not missing, f"{label} missing columns: {missing}")


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid numeric value at {label}: {value!r}") from error
    require(math.isfinite(result), f"Non-finite numeric value at {label}")
    return result


def strict_int(value: Any, label: str) -> int:
    number = finite_float(value, label)
    require(number.is_integer(), f"Expected integer at {label}: {value!r}")
    return int(number)


def validate_analysis_manifest(manifest: Mapping[str, Any]) -> None:
    require(
        manifest.get("schema") == "obs2_v2_1_k_analysis_manifest/v1",
        "Analysis manifest schema mismatch",
    )
    require(
        manifest.get("implementation_status") == IMPLEMENTATION_STATUS
        and manifest.get("analysis_timing_status") == IMPLEMENTATION_STATUS,
        "Analysis manifest is not post_result_transparent_implementation",
    )
    require(manifest.get("episode_rows") == 5900, "Analysis manifest must seal 5900 episode rows")
    require(manifest.get("condition_seed_rows") == 295, "Analysis manifest must seal 295 condition-seed rows")
    require(manifest.get("condition_rows") == 59, "Analysis manifest must seal 59 conditions")
    require(manifest.get("fixed_evaluation_seed_count") == 20, "Expected 20 fixed evaluation seeds")
    fixed_seeds = manifest.get("fixed_evaluation_seeds")
    require(isinstance(fixed_seeds, list) and len(fixed_seeds) == 20, "Invalid fixed evaluation-seed inventory")
    require(len(set(fixed_seeds)) == 20 and all(type(value) is int for value in fixed_seeds), "Evaluation seeds must be 20 unique integers")

    expected_source_hash = require_sha256(
        manifest.get("analysis_source_sha256"), "analysis_manifest.analysis_source_sha256"
    )
    require_regular_file(ANALYZE_SOURCE, "analysis source")
    require(
        sha256_file(ANALYZE_SOURCE) == expected_source_hash,
        "analyze_results.py no longer matches the source sealed by analysis_manifest.json",
    )

    integrity = manifest.get("integrity")
    require(isinstance(integrity, Mapping), "Analysis manifest has no integrity object")
    require(
        integrity.get("implementation_status") == IMPLEMENTATION_STATUS,
        "Analysis integrity status mismatch",
    )
    result_hashes = integrity.get("result_file_sha256")
    require(
        integrity.get("result_file_count") == 295
        and isinstance(result_hashes, Mapping)
        and len(result_hashes) == 295,
        "Analysis integrity does not seal exactly 295 result files",
    )
    for relative, digest in result_hashes.items():
        require(
            isinstance(relative, str)
            and relative.startswith("results/")
            and relative.endswith(".json"),
            f"Invalid sealed result path: {relative!r}",
        )
        require_sha256(digest, f"integrity.result_file_sha256[{relative!r}]")
    for key in (
        "result_inventory_sha256",
        "validator_results_manifest_sha256",
        "frozen_evaluator_sha256",
        "main_execution_complete_sha256",
    ):
        require_sha256(integrity.get(key), f"integrity.{key}")
    audit_hashes = integrity.get("audit_sha256")
    require(isinstance(audit_hashes, Mapping) and len(audit_hashes) == 3, "Expected three sealed audit SHA-256 values")
    for name, digest in audit_hashes.items():
        require(isinstance(name, str) and name.endswith(".json"), f"Invalid audit name: {name!r}")
        require_sha256(digest, f"integrity.audit_sha256[{name!r}]")


def validate_tables(
    episode_fields: Sequence[str],
    episodes: list[dict[str, str]],
    summary_fields: Sequence[str],
    summaries: list[dict[str, str]],
    manifest: Mapping[str, Any],
) -> tuple[tuple[Any, ...], tuple[int, ...]]:
    conditions = build_conditions()
    require(len(conditions) == 59, "Condition matrix is not exactly 59 conditions")
    condition_by_id = {condition.id: condition for condition in conditions}
    require(len(condition_by_id) == 59, "Condition matrix contains duplicate IDs")
    require(
        Counter(condition.module for condition in conditions) == Counter({"A": 4, "B": 32, "C": 13, "D": 10}),
        "Condition matrix module counts have drifted",
    )

    episode_required = {
        "implementation_status", "condition_id", "module", "family",
        "training_seed", "evaluation_seed", "success",
    }
    for joint in JOINTS:
        for channel in ("K1", "K2"):
            episode_required.add(f"{joint}_{channel}_mean")
            episode_required.add(f"{joint}_{channel}_positive_fraction")
    require_fields(episode_fields, episode_required, "episode_results.csv")
    require_fields(
        summary_fields,
        {
            "implementation_status", "condition_id", "module", "family",
            "training_seed", "success_episodes", "success_rate",
        },
        "condition_seed_summary.csv",
    )
    require(len(episodes) == manifest["episode_rows"] == 5900, "Episode table row count mismatch")
    require(len(summaries) == manifest["condition_seed_rows"] == 295, "Summary table row count mismatch")

    training_seeds = tuple(sorted({strict_int(row["training_seed"], "episode.training_seed") for row in episodes}))
    require(len(training_seeds) == 5, "Expected exactly five unique training seeds")
    expected_evaluation_seeds = tuple(manifest["fixed_evaluation_seeds"])
    episode_groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(episodes, start=2):
        label = f"episode_results.csv row {index}"
        require(row["implementation_status"] == IMPLEMENTATION_STATUS, f"Status mismatch at {label}")
        condition_id = row["condition_id"]
        require(condition_id in condition_by_id, f"Unknown condition at {label}: {condition_id}")
        condition = condition_by_id[condition_id]
        require(row["module"] == condition.module and row["family"] == condition.family, f"Condition metadata mismatch at {label}")
        seed = strict_int(row["training_seed"], f"{label}.training_seed")
        evaluation_seed = strict_int(row["evaluation_seed"], f"{label}.evaluation_seed")
        require(evaluation_seed in expected_evaluation_seeds, f"Unknown evaluation seed at {label}")
        success = strict_int(row["success"], f"{label}.success")
        require(success in (0, 1), f"Success must be 0/1 at {label}")
        for joint in JOINTS:
            for channel in ("K1", "K2"):
                finite_float(row[f"{joint}_{channel}_mean"], f"{label}.{joint}_{channel}_mean")
                fraction = finite_float(
                    row[f"{joint}_{channel}_positive_fraction"],
                    f"{label}.{joint}_{channel}_positive_fraction",
                )
                require(0.0 <= fraction <= 1.0, f"Positive fraction outside [0,1] at {label}")
        episode_groups[(seed, condition_id)].append(row)

    expected_pairs = {(seed, condition.id) for seed in training_seeds for condition in conditions}
    require(set(episode_groups) == expected_pairs, "Episode table condition-seed inventory is incomplete")
    for pair, rows in episode_groups.items():
        evaluation_seeds = [strict_int(row["evaluation_seed"], "evaluation_seed") for row in rows]
        require(len(rows) == 20 and set(evaluation_seeds) == set(expected_evaluation_seeds), f"Evaluation-seed inventory mismatch for {pair}")

    summary_map: dict[tuple[int, str], dict[str, str]] = {}
    for index, row in enumerate(summaries, start=2):
        label = f"condition_seed_summary.csv row {index}"
        require(row["implementation_status"] == IMPLEMENTATION_STATUS, f"Status mismatch at {label}")
        condition_id = row["condition_id"]
        require(condition_id in condition_by_id, f"Unknown condition at {label}: {condition_id}")
        condition = condition_by_id[condition_id]
        require(row["module"] == condition.module and row["family"] == condition.family, f"Condition metadata mismatch at {label}")
        seed = strict_int(row["training_seed"], f"{label}.training_seed")
        key = (seed, condition_id)
        require(key not in summary_map, f"Duplicate condition-seed summary: {key}")
        count = strict_int(row["success_episodes"], f"{label}.success_episodes")
        rate = finite_float(row["success_rate"], f"{label}.success_rate")
        require(0 <= count <= 20 and 0.0 <= rate <= 1.0, f"Invalid success summary at {label}")
        require(math.isclose(rate, count / 20.0, rel_tol=0.0, abs_tol=1e-12), f"Success rate/count mismatch at {label}")
        episode_count = sum(strict_int(item["success"], "success") for item in episode_groups[key])
        require(count == episode_count, f"Episode/summary success mismatch for {key}")
        summary_map[key] = row
    require(set(summary_map) == expected_pairs, "Condition-seed summary inventory is incomplete")
    return conditions, training_seeds


def mean_rows(rows: Iterable[Mapping[str, str]], field: str) -> float:
    values = [finite_float(row[field], field) for row in rows]
    require(bool(values), f"No values for {field}")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def add_figure_note(fig: plt.Figure) -> None:
    fig.text(0.5, 0.006, FIGURE_NOTE, ha="center", va="bottom", fontsize=8, color="#555555")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        fig.savefig(temporary, dpi=220, bbox_inches="tight", facecolor="white")
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


def annotate_heatmap(ax: plt.Axes, data: np.ndarray, *, signed: bool = False, fontsize: int = 7) -> None:
    threshold = float(np.nanmax(np.abs(data))) * 0.55 if signed else 0.58
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            value = float(data[row, column])
            color = "white" if (abs(value) if signed else value) > threshold else "black"
            text = f"{value:+.2f}" if signed else f"{value:.2f}"
            ax.text(column, row, text, ha="center", va="center", fontsize=fontsize, color=color)


def c11_joint_matrices(
    episodes: Sequence[Mapping[str, str]], training_seeds: Sequence[int], field_suffix: str
) -> dict[str, np.ndarray]:
    c11 = [row for row in episodes if row["condition_id"] == "C11"]
    matrices: dict[str, np.ndarray] = {}
    for channel in ("K1", "K2"):
        matrices[channel] = np.asarray(
            [
                [
                    mean_rows(
                        (row for row in c11 if strict_int(row["training_seed"], "training_seed") == seed),
                        f"{joint}_{channel}_{field_suffix}",
                    )
                    for joint in JOINTS
                ]
                for seed in training_seeds
            ],
            dtype=np.float64,
        )
    return matrices


def figure_c11_means(episodes: Sequence[Mapping[str, str]], training_seeds: Sequence[int]) -> plt.Figure:
    matrices = c11_joint_matrices(episodes, training_seeds, "mean")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    x = np.arange(len(JOINTS))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(training_seeds)))
    for ax, channel in zip(axes, ("K1", "K2")):
        data = matrices[channel]
        for seed, values, color in zip(training_seeds, data, colors):
            ax.plot(x, values, marker="o", linewidth=1.2, alpha=0.72, color=color, label=f"seed {seed}")
        ax.plot(x, np.mean(data, axis=0), marker="D", linewidth=3.0, color="black", label="across-seed mean")
        ax.axhline(0.0, color="#888888", linewidth=0.8)
        ax.set_title(f"C11 {channel} signed mean by joint")
        ax.set_xticks(x, JOINTS)
        ax.set_xlabel("Joint")
        ax.set_ylabel(f"{channel} mean")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("C11 per-seed joint profiles and equally weighted across-seed mean")
    add_figure_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def figure_c11_positive_fraction(episodes: Sequence[Mapping[str, str]], training_seeds: Sequence[int]) -> plt.Figure:
    matrices = c11_joint_matrices(episodes, training_seeds, "positive_fraction")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    image = None
    for ax, channel in zip(axes, ("K1", "K2")):
        data = matrices[channel]
        image = ax.imshow(data, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"C11 {channel} positive-sign fraction")
        ax.set_xticks(np.arange(len(JOINTS)), JOINTS)
        ax.set_yticks(np.arange(len(training_seeds)), [str(seed) for seed in training_seeds])
        ax.set_xlabel("Joint")
        annotate_heatmap(ax, data)
    axes[0].set_ylabel("Training seed")
    require(image is not None, "Heatmap image was not created")
    fig.colorbar(image, ax=axes, label="Positive-sign fraction", fraction=0.025, pad=0.03)
    fig.suptitle("C11 K1/K2 sign prevalence (mean across 20 fixed evaluation seeds)")
    add_figure_note(fig)
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.16, top=0.82, wspace=0.12)
    return fig


def success_map(summaries: Sequence[Mapping[str, str]]) -> dict[tuple[int, str], float]:
    return {
        (strict_int(row["training_seed"], "training_seed"), row["condition_id"]):
        finite_float(row["success_rate"], "success_rate")
        for row in summaries
    }


def figure_all_success(
    summaries: Sequence[Mapping[str, str]], conditions: Sequence[Any], training_seeds: Sequence[int]
) -> plt.Figure:
    rates = success_map(summaries)
    data = np.asarray([[rates[(seed, condition.id)] for seed in training_seeds] for condition in conditions])
    fig, ax = plt.subplots(figsize=(9.2, 18.5))
    image = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(training_seeds)), [str(seed) for seed in training_seeds])
    ax.set_yticks(np.arange(len(conditions)), [f"{condition.module} | {condition.id}" for condition in conditions], fontsize=7)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Condition (canonical condition_matrix order)")
    ax.set_title("Success rate for all 59 conditions × 5 training seeds")
    for boundary in (4, 36, 49):
        ax.axhline(boundary - 0.5, color="black", linewidth=1.8)
    annotate_heatmap(ax, data, fontsize=5)
    fig.colorbar(image, ax=ax, label="Success rate (20 fixed evaluation seeds)", fraction=0.025, pad=0.02)
    add_figure_note(fig)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    return fig


def effect_matrix(
    rates: Mapping[tuple[int, str], float],
    training_seeds: Sequence[int],
    prefix: str,
    baseline: str,
    necessity: bool,
) -> np.ndarray:
    result = np.empty((len(training_seeds), len(JOINTS)), dtype=np.float64)
    for row, seed in enumerate(training_seeds):
        for column, joint in enumerate(JOINTS):
            intervention = rates[(seed, f"{prefix}_{joint}")]
            base = rates[(seed, baseline)]
            result[row, column] = base - intervention if necessity else intervention - base
    return result


def figure_joint_effects(summaries: Sequence[Mapping[str, str]], training_seeds: Sequence[int]) -> plt.Figure:
    rates = success_map(summaries)
    specs = (
        ("K1_SUFF", "C00", False, "K1 sufficiency: K1_SUFF − C00 (R0 background)"),
        ("K1_NEC", "C10", True, "K1 necessity: C10 − K1_NEC (background C10)"),
        ("K2_SUFF", "C00", False, "K2 sufficiency: K2_SUFF − C00 (R0 background)"),
        ("K2_NEC", "C11", True, "K2 necessity: C11 − K2_NEC (background C11)"),
    )
    matrices = [effect_matrix(rates, training_seeds, prefix, base, necessity) for prefix, base, necessity, _ in specs]
    limit = max(float(np.max(np.abs(matrix))) for matrix in matrices)
    limit = max(limit, 0.05)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharex=True, sharey=True)
    image = None
    for ax, matrix, spec in zip(axes.flat, matrices, specs):
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(spec[3], fontsize=10)
        ax.set_xticks(np.arange(len(JOINTS)), JOINTS)
        ax.set_yticks(np.arange(len(training_seeds)), [str(seed) for seed in training_seeds])
        annotate_heatmap(ax, matrix, signed=True)
    axes[0, 0].set_ylabel("Training seed")
    axes[1, 0].set_ylabel("Training seed")
    axes[1, 0].set_xlabel("Joint")
    axes[1, 1].set_xlabel("Joint")
    require(image is not None, "Effect heatmap image was not created")
    fig.colorbar(image, ax=axes, label="Oriented success-rate effect", fraction=0.025, pad=0.025)
    fig.suptitle("Per-joint sufficiency and necessity effects by training seed\nPositive values follow the stated sufficiency/necessity orientation")
    add_figure_note(fig)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.1, top=0.84, hspace=0.3, wspace=0.12)
    return fig


def grouped_success_figure(
    summaries: Sequence[Mapping[str, str]],
    conditions: Sequence[Any],
    training_seeds: Sequence[int],
    *,
    module: str,
    families: Sequence[str],
    title: str,
) -> plt.Figure:
    selected = [condition for condition in conditions if condition.module == module and condition.family in families]
    rates = success_map(summaries)
    data = np.asarray([[rates[(seed, condition.id)] for condition in selected] for seed in training_seeds])
    width = max(12.0, 1.05 * len(selected))
    fig, ax = plt.subplots(figsize=(width, 5.4))
    image = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(selected)), [condition.id for condition in selected], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(training_seeds)), [str(seed) for seed in training_seeds])
    ax.set_xlabel("Condition (canonical condition_matrix order)")
    ax.set_ylabel("Training seed")
    ax.set_title(title)
    previous = selected[0].family
    for index, condition in enumerate(selected[1:], start=1):
        if condition.family != previous:
            ax.axvline(index - 0.5, color="black", linewidth=1.6)
            previous = condition.family
    annotate_heatmap(ax, data)
    fig.colorbar(image, ax=ax, label="Success rate (20 fixed evaluation seeds)", fraction=0.035, pad=0.02)
    add_figure_note(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    input_paths = {name: ROOT / Path(name) for name in INPUT_NAMES}
    for name, path in input_paths.items():
        require_regular_file(path, name)
    require_regular_file(CONDITION_SOURCE, "condition matrix source")

    hashes_before = {name: sha256_file(path) for name, path in input_paths.items()}
    manifest = load_json(input_paths["analysis/analysis_manifest.json"])
    validate_analysis_manifest(manifest)
    episode_fields, episodes = read_csv_rows(input_paths["analysis/episode_results.csv"])
    summary_fields, summaries = read_csv_rows(input_paths["analysis/condition_seed_summary.csv"])
    conditions, training_seeds = validate_tables(
        episode_fields, episodes, summary_fields, summaries, manifest
    )
    hashes_after = {name: sha256_file(path) for name, path in input_paths.items()}
    require(hashes_before == hashes_after, "Analysis inputs changed while being read; refusing to plot")

    figure_specs = (
        ("01_c11_k1_k2_mean_by_seed.png", figure_c11_means(episodes, training_seeds), "C11 per-seed K1/K2 signed joint means and across-seed means."),
        ("02_c11_k1_k2_positive_fraction_heatmap.png", figure_c11_positive_fraction(episodes, training_seeds), "C11 K1/K2 positive-sign fractions by seed and joint."),
        ("03_all_conditions_success_rate_heatmap.png", figure_all_success(summaries, conditions, training_seeds), "Success rates for all 59 conditions and five training seeds in canonical order, with module boundaries."),
        ("04_per_joint_sufficiency_necessity_effects.png", figure_joint_effects(summaries, training_seeds), "Seed-by-joint oriented sufficiency/necessity effects; K1 necessity uses C10 and K2 necessity uses C11."),
        (
            "05_module_c_sign_space_success_heatmap.png",
            grouped_success_figure(
                summaries, conditions, training_seeds,
                module="C", families=("k1_sign_space",),
                title="Module C K1 sign/spatial controls: success rate by seed × condition",
            ),
            "Module C K1 sign/spatial-control success rates by seed and condition.",
        ),
        (
            "06_module_d_k2_controls_success_heatmap.png",
            grouped_success_figure(
                summaries, conditions, training_seeds,
                module="D",
                families=("k2_amplitude", "k2_sign", "k2_region", "k2_temporal_calibration"),
                title="Module D K2 amplitude/sign/region/timing controls: success rate by seed × condition",
            ),
            "Module D K2 amplitude, sign, region, and timing-control success rates by seed and condition.",
        ),
    )

    output_descriptions: dict[str, str] = {}
    with tempfile.TemporaryDirectory(
        prefix=".extended_figures_stage_", dir=str(ANALYSIS)
    ) as staging_name:
        staging_dir = Path(staging_name)
        for filename, figure, description in figure_specs:
            save_figure(figure, staging_dir / filename)
            final_path = FIGURE_DIR / filename
            output_descriptions[final_path.relative_to(ROOT).as_posix()] = description

        hashes_final = {name: sha256_file(path) for name, path in input_paths.items()}
        require(
            hashes_final == hashes_after,
            "Analysis inputs changed while figures were rendered; refusing to publish",
        )
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        for filename, _, _ in figure_specs:
            os.replace(staging_dir / filename, FIGURE_DIR / filename)

    source_sha256 = {
        Path(__file__).resolve().relative_to(ROOT).as_posix(): sha256_file(Path(__file__).resolve()),
        ANALYZE_SOURCE.relative_to(ROOT).as_posix(): sha256_file(ANALYZE_SOURCE),
        CONDITION_SOURCE.relative_to(ROOT).as_posix(): sha256_file(CONDITION_SOURCE),
    }
    output_sha256 = {
        name: sha256_file(ROOT / Path(name)) for name in sorted(output_descriptions)
    }
    output_manifest = {
        "schema": "obs2_v2_1_k_extended_figures_manifest/v1",
        "implementation_status": IMPLEMENTATION_STATUS,
        "analysis_timing_status": "post_result_descriptive_visualization",
        "study_id": manifest.get("study_id"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Post-result descriptive visualization only. These figures do not constitute "
            "preregistration, confirmatory analysis, independent inference, or additional evidence."
        ),
        "raw_results_accessed": False,
        "source_sha256": source_sha256,
        "input_sha256": hashes_after,
        "output_sha256": output_sha256,
        "output_descriptions": output_descriptions,
        "analysis_manifest_source_binding_sha256": manifest["analysis_source_sha256"],
        "analysis_result_inventory_sha256": manifest["integrity"]["result_inventory_sha256"],
        "condition_matrix_canonical_sha256": canonical_sha256(conditions),
        "condition_order": [condition.id for condition in conditions],
        "module_boundaries_after_condition_index_1_based": [4, 36, 49],
        "training_seeds": list(training_seeds),
        "joint_order": list(JOINTS),
    }
    atomic_json(MANIFEST_PATH, output_manifest)
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
