"""Shared path and runtime helpers for the portable thesis release."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = PROJECT_ROOT / "configs" / "paths.example.json"
LOCAL_PATHS = PROJECT_ROOT / "configs" / "paths.local.json"
PATH_KEYS = {
    "training_output_root",
    "formal_checkpoint_root",
    "avm_checkpoint_root",
    "formal_evaluation_output",
    "avm_evaluation_output",
    "gait_output",
    "hpr_output",
    "sgrr_output",
    "sgrr_legacy_root",
    "figure_output",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_paths() -> dict[str, Any]:
    override = os.environ.get("THESIS_REPRO_PATHS")
    source = resolve_project_path(override) if override else LOCAL_PATHS
    if not source.is_file():
        source = DEFAULT_PATHS
    config = load_json(source)
    if config.get("schema") != "thesis_repro_paths/v1":
        raise RuntimeError(f"Unsupported path-config schema: {source}")
    config["_source"] = str(source)
    for key in PATH_KEYS:
        if key not in config:
            raise KeyError(f"Missing {key!r} in {source}")
        config[key] = resolve_project_path(config[key])
    return config


def configured_python(config: dict[str, Any] | None = None) -> Path:
    config = config or load_paths()
    value = str(config.get("python", "auto"))
    if value.lower() == "auto":
        return Path(sys.executable).resolve()
    return resolve_project_path(value)


def python_site_packages(python: Path) -> Path:
    if python.resolve() == Path(sys.executable).resolve():
        return Path(sysconfig.get_paths()["purelib"]).resolve()
    output = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
    ).strip()
    return Path(output).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subprocess_environment(config: dict[str, Any] | None = None) -> dict[str, str]:
    config = config or load_paths()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "MPLBACKEND": "Agg",
            "CUDA_VISIBLE_DEVICES": str(config.get("cuda_visible_devices", "0")),
            "OMP_NUM_THREADS": str(config.get("threads_per_worker", 1)),
            "MKL_NUM_THREADS": str(config.get("threads_per_worker", 1)),
            "OPENBLAS_NUM_THREADS": str(config.get("threads_per_worker", 1)),
            "NUMEXPR_NUM_THREADS": str(config.get("threads_per_worker", 1)),
        }
    )
    existing = env.get("PYTHONPATH")
    additions = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "training"),
        str(PROJECT_ROOT / "packages" / "metamaterial_envs"),
    ]
    env["PYTHONPATH"] = os.pathsep.join(additions + ([existing] if existing else []))
    return env


def ensure_output_path(path: Path) -> Path:
    """Create a configured output directory while rejecting filesystem roots."""

    path = path.resolve()
    if path == Path(path.anchor) or path == PROJECT_ROOT:
        raise RuntimeError(f"Refusing to use a filesystem/project root as output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
