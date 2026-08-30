from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path


def load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("v2_1_regression_runner_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(seed: int, successes: int) -> dict:
    return {
        "stage": "regression",
        "seed": seed,
        "arm": "V2_1",
        "success_episodes": successes,
        "evaluation_episodes": 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    runner = load_runner(args.runner.resolve())
    config_path = args.config.resolve()
    setup_root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="v2_1_orchestrator_test_") as temporary:
        instance = runner.Orchestrator.__new__(runner.Orchestrator)
        instance.root = Path(temporary)
        instance.config = config
        instance.code = setup_root / config["runtime"]["code_snapshot_relative"]
        instance.reference_config = (
            setup_root / config["reference_v2"]["experiment_config_relative"]
        )
        instance.reference_evaluation_summary = (
            setup_root / config["reference_v2"]["evaluation_summary_relative"]
        )
        instance.verify_reference_contract()
        assert (instance.root / "_control" / "reference_contract_receipt.json").is_file()

        seeds = [9101, 9102, 9103]
        assert instance.gate_impossible_reason([], seeds) is None
        assert "seed 9101" in instance.gate_impossible_reason([result(9101, 4)], seeds)
        assert instance.gate_impossible_reason([result(9101, 5)], seeds) is None
        assert "maximum possible total 23/30" in instance.gate_impossible_reason(
            [result(9101, 5), result(9102, 8)], seeds
        )
        assert instance.gate_impossible_reason(
            [result(9101, 6), result(9102, 8)], seeds
        ) is None
        assert instance.gate_impossible_reason(
            [result(9101, 6), result(9102, 8), result(9103, 10)], seeds
        ) is None

        gate = instance.regression_gate_payload(
            [result(9101, 6), result(9102, 8), result(9103, 10)],
            passed=True,
            terminal_reason="self-test",
        )
        assert gate["passed"] is True
        assert gate["total_success_episodes"] == 24
        assert gate["success_episodes_by_seed"] == {"9101": 6, "9102": 8, "9103": 10}
        assert gate["formal_started"] is False

    print("PASS: v2.1 regression orchestrator reference, gate, and early-stop self-test")


if __name__ == "__main__":
    main()
