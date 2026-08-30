"""Self-tests for the actor-only, capacity-matched observation intervention."""

from __future__ import annotations

import unittest

import torch

import actor_observation_shim as shim


class ActorObservationShamTests(unittest.TestCase):
    def make_backbone(self):
        return shim._BASE.MultiAgentMLP(
            n_agent_inputs=2,
            n_agent_outputs=4,
            n_agents=8,
            centralised=False,
            share_params=False,
            depth=2,
            num_cells=16,
            device=torch.device("cpu"),
            activation_class=torch.nn.Tanh,
        )

    def test_mask_preserves_first_channel_and_shape(self):
        value = torch.tensor([[[1.25, -7.0], [-3.0, 8.5]]], requires_grad=True)
        masked = shim.spatial_only_sham_tensor(value)
        self.assertEqual(masked.shape, value.shape)
        self.assertEqual(masked.dtype, value.dtype)
        self.assertTrue(torch.equal(masked[..., 0], value[..., 0]))
        self.assertTrue(torch.equal(masked[..., 1], torch.zeros_like(value[..., 1])))
        masked.sum().backward()
        self.assertIsNotNone(value.grad)

    def test_hook_changes_no_state_dict_entry(self):
        torch.manual_seed(123)
        network = self.make_backbone()
        before = {name: value.detach().clone() for name, value in network.state_dict().items()}
        captured = []
        shim.install_actor_observation_mode(network, "spatial_only_sham")
        network.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        observation = torch.randn(3, 8, 2)
        network(observation)
        after = network.state_dict()
        self.assertEqual(tuple(before), tuple(after))
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), name)
        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(captured[0][..., 0], observation[..., 0]))
        self.assertTrue(torch.equal(captured[0][..., 1], torch.zeros_like(observation[..., 1])))

    def test_full_o2_installs_no_mask(self):
        network = self.make_backbone()
        captured = []
        shim.install_actor_observation_mode(network, "full_o2")
        network.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().clone()))
        observation = torch.randn(2, 8, 2)
        network(observation)
        self.assertTrue(torch.equal(captured[0], observation))

    def test_bad_shape_fails_closed(self):
        with self.assertRaises(RuntimeError):
            shim.spatial_only_sham_tensor(torch.zeros(4, 8, 1))

    def test_flag_parser(self):
        mode, cleaned = shim.split_actor_observation_mode(
            ["--seed", "9201", "--actor-observation-mode", "spatial_only_sham"]
        )
        self.assertEqual(mode, "spatial_only_sham")
        self.assertEqual(cleaned, ["--seed", "9201"])
        with self.assertRaises(ValueError):
            shim.split_actor_observation_mode(
                ["--actor-observation-mode", "spatial_only_sham", "--actor-observation-mode=full_o2"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
