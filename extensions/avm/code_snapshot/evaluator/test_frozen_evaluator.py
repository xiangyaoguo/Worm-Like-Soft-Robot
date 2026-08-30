from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frozen_evaluator_under_test", HERE / "frozen_evaluator.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError("frozen_evaluator.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FrozenEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((HERE / "evaluator_config.json").read_text(encoding="utf-8"))

    def test_frozen_config_matrix(self) -> None:
        MODULE.validate_config(self.config)
        tasks = MODULE.all_tasks(self.config)
        self.assertEqual(len(tasks), 150)
        self.assertEqual(len({task.task_id for task in tasks}), 150)
        self.assertEqual(tasks[0].task_id, "O2__run0__checkpoint0100")
        self.assertEqual(tasks[-1].task_id, "O1_sham__run4__checkpoint1500")

    def test_common_criterion_boundaries(self) -> None:
        criterion = self.config["evaluation"]["common_kinematic_criterion"]
        metrics = {
            "desired_net_rotation_degrees": 360.0,
            "desired_active_rotation_fraction": 0.7,
            "forward_body_lengths": 1.0,
        }
        self.assertTrue(MODULE.common_kinematic_success(metrics, criterion))
        for field, failing in (
            ("desired_net_rotation_degrees", 359.999),
            ("desired_active_rotation_fraction", 0.6999),
            ("forward_body_lengths", 0.9999),
        ):
            changed = dict(metrics)
            changed[field] = failing
            self.assertFalse(MODULE.common_kinematic_success(changed, criterion))

    def test_o2_locked_legacy_mode_is_narrow(self) -> None:
        arm = self.config["arms"]["O2"]
        metadata = {"training_args": {"run_name": "formal__seed9201__R0"}}
        mode, source = MODULE.infer_actor_observation_mode(metadata, "O2", arm, 9201)
        self.assertEqual(mode, "full_o2")
        self.assertEqual(source, "locked_archived_o2_contract")
        metadata["training_args"]["run_name"] = "wrong"
        with self.assertRaises(RuntimeError):
            MODULE.infer_actor_observation_mode(metadata, "O2", arm, 9201)

    def test_o1_requires_explicit_checkpoint_metadata(self) -> None:
        arm = self.config["arms"]["O1_sham"]
        metadata = {"training_args": {"run_name": "formal__seed9201__HPR__O1sham"}}
        with self.assertRaises(RuntimeError):
            MODULE.infer_actor_observation_mode(metadata, "O1_sham", arm, 9201)
        metadata["training_args"]["actor_observation_mode"] = "spatial_only_sham"
        mode, source = MODULE.infer_actor_observation_mode(metadata, "O1_sham", arm, 9201)
        self.assertEqual((mode, source), ("spatial_only_sham", "checkpoint_metadata"))

    def test_o1_path_is_segregated_formal_extension(self) -> None:
        task = MODULE.Task("O1_sham", 9205, 1500)
        path = MODULE.checkpoint_path(self.config, task)
        self.assertIn("formal_o1_sham_hpr_20260811", str(path))
        self.assertTrue(str(path).endswith("formal__seed9205__HPR__O1sham\\checkpoint_1500.pt"))

    def test_json_rejects_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.json_safe(float("nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
