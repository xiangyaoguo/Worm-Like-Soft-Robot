from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = range(9201, 9206)
TAGS = ("DTH", "THDOT", "OBS", "HPR__O2shared", "R0", "Rroll")


class ReleaseLayoutTests(unittest.TestCase):
    def test_formal_training_matrix(self) -> None:
        config = json.loads((ROOT / "configs" / "formal_training.json").read_text(encoding="utf-8"))
        self.assertEqual(config["training_seeds"], list(SEEDS))
        self.assertEqual(config["formal_batches"], 1500)
        self.assertEqual(len(config["arms"]), 7)
        self.assertEqual(config["arms"]["SGRR-O2-JS"]["reward_func"], "obs2_roll_repro_v2_1")
        self.assertEqual(
            config["arms"]["HPR-O2-AVM-JS"]["actor_observation_mode"],
            "spatial_only_sham",
        )

    def test_all_bundled_endpoints_exist(self) -> None:
        required = ("checkpoint_1500.pt", "metadata.json", "training_log.csv", "training_summary.json")
        for seed in SEEDS:
            for tag in TAGS:
                run = ROOT / "checkpoints" / "formal_six" / "runs" / f"formal__seed{seed}__{tag}"
                for name in required:
                    self.assertGreater((run / name).stat().st_size, 0)
            avm = ROOT / "checkpoints" / "avm" / "runs" / f"formal__seed{seed}__HPR__O1sham"
            for name in required:
                self.assertGreater((avm / name).stat().st_size, 0)

    def test_formal_source_contains_sgrr_v2_1(self) -> None:
        trainer = (ROOT / "training" / "train_metamaterial.py").read_text(encoding="utf-8")
        environment = (
            ROOT
            / "packages"
            / "metamaterial_envs"
            / "metamaterial_envs"
            / "env"
            / "metamaterial.py"
        ).read_text(encoding="utf-8")
        self.assertIn("obs2_roll_repro_v2_1", trainer)
        self.assertIn("obs2_roll_repro_v2_1", environment)

    def test_fixed_reset_panel(self) -> None:
        config = json.loads(
            (ROOT / "evaluation" / "formal_endpoint" / "evaluator_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["evaluation"]["reset_seeds"], list(range(20264101, 20264121)))
        strict = config["evaluation"]["secondary_strict_common_kinematic"]
        self.assertEqual(strict["minimum_desired_net_rotation_degrees"], 360.0)
        self.assertEqual(strict["minimum_desired_active_rotation_fraction"], 0.7)
        self.assertEqual(strict["minimum_forward_body_lengths"], 1.0)

    def test_portable_path_contract(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "paths.example.json").read_text(encoding="utf-8")
        )
        required = {
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
        self.assertTrue(required.issubset(config))

    def test_checkpoint_hash_manifest_shape(self) -> None:
        with (ROOT / "CHECKPOINT_SHA256.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 35)
        self.assertEqual(len({row["relative_path"] for row in rows}), 35)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in rows))

    def test_hpr_archived_baseline_counts(self) -> None:
        expected_counts = {9201: 20, 9203: 7, 9205: 20}
        evaluation_root = (
            ROOT / "provenance" / "formal_parent_exact" / "formal" / "evaluations"
        )
        for seed, expected_count in expected_counts.items():
            payload = json.loads(
                (
                    evaluation_root
                    / f"formal__seed{seed}__R0__eval_attempt1.json"
                ).read_text(encoding="utf-8-sig")
            )
            episodes = payload["results"][0]["episodes"]
            actual_count = sum(
                float(row["desired_net_rotation_degrees"]) >= 360.0
                and float(row["desired_active_rotation_fraction"]) >= 0.70
                and float(row["forward_body_lengths"]) >= 1.0
                for row in episodes
            )
            self.assertEqual(actual_count, expected_count)

    def test_formal_evaluator_helper_closure(self) -> None:
        helper_root = ROOT / "evaluation" / "formal_endpoint"
        self.assertTrue((helper_root / "rotation_span_helper.py").is_file())
        self.assertTrue((helper_root / "r0_v21core_contract.py").is_file())


if __name__ == "__main__":
    unittest.main()
