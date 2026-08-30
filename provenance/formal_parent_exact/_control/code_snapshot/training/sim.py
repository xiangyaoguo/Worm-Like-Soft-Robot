import argparse
import torch
from metamaterial_envs import metamaterial
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, TanhDelta
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
import numpy as np
import matplotlib.pyplot as plt
from tensordict import TensorDict
from benchmarks import get_best_ko


def internode_angle(pos: np.ndarray) -> np.ndarray:
    dp = np.roll(pos, -1, 1)-pos
    dp_norm = dp/np.absolute(dp)
    return np.angle(-dp_norm/np.roll(dp_norm, 1, 1))%(2*np.pi)


def get_checkpoint(path):
    saved_params = torch.load(path, weights_only=False)
    metadata = saved_params["metadata"]
    return metadata, saved_params["policy"]

class BiasedNormalParamExtractor(NormalParamExtractor):
    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        loc, scale = tensor.chunk(2, -1)
        scale = self.scale_mapping(scale) + self.scale_lb # add instead of min clamp
        return (loc, scale, *others)

class SafeTanhNormal(TanhNormal):
    def log_prob(self, value):
        # Get log-probabilities from Normal distribution.
        log_prob = super().log_prob(value)
        # Clip the log probabilities as a safeguard
        # log_prob = torch.clamp(log_prob, -20, 20)
        epsilon = 1e-6
        log_prob = torch.log(torch.exp(log_prob) + epsilon)
        return log_prob

def make_nn_policy(env, algorithm, policy_net_config, share_parameters_policy, normal_scale_lb=0.001):
    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],  # n_obs_per_agent
        n_agent_outputs=env.full_action_spec[env.action_key].shape[-1] * (2 if algorithm == 'ppo' else 1), #PPO needs 2 outputs: loc and scale
        n_agents=env.num_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=share_parameters_policy,
        activation_class=torch.nn.Tanh,
        **policy_net_config
    )
    if algorithm == 'ppo':
        policy_net = torch.nn.Sequential(
            policy_net,
            BiasedNormalParamExtractor(scale_lb=normal_scale_lb),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
        )
    temp_keys = [("agents", "loc"), ("agents", "scale")] if algorithm == 'ppo' else [("agents", "param")]
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=temp_keys,
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=temp_keys,
        out_keys=[env.action_key],
        distribution_class= (SafeTanhNormal if algorithm == 'ppo' else TanhDelta), # PPO learns a normal distribution
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob= (algorithm == 'ppo'), # PPO loss needs log probs, DDPG does not
    )
    return policy


class Sim:
    def __init__(self, window_width=1000, window_height=250, label=[], num_envs=1, **kwargs):
        self.window_width = window_width
        self.window_height = window_height
        self.label = label
        self.num_envs = num_envs

    def reset(self):
        self.td = self.env.reset()

    def reset_seed(self, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.td = self.env.reset()

    def step_render(self):
        with torch.no_grad():
            self.td = self.policy(self.td)
            self.td = self.env.step(self.td)["next"]
            ret = self.env.render()
        return ret

    def steps(self, n=1):
        with torch.no_grad():
            for _ in range(n):
                self.td = self.policy(self.td)
                self.td = self.env.step(self.td)["next"]
    
    def render_rainbow(self):
        return self.env.render_rainbow()
    
    def evaluate_speed(self, steps=1000):
        self.reset()
        speeds = []
        with torch.no_grad():
            for _ in range(steps):
                self.td = self.policy(self.td)
                self.td = self.env.step(self.td)["next"]
                speeds.append(self.td["log_info", "speed"].numpy())
        speeds = np.array(speeds)
        mean_speeds = np.mean(np.array(speeds), axis=0) # one element for each crawler; the next mean and std is then calculated over just crawlers, rather than over timesteps too
        return np.mean(mean_speeds), np.std(mean_speeds)

class TrainedPolicySim(Sim):
    """
    Instantiates a Sim from a pt checkpoint of a trained policy
    """
    def __init__(self, checkpoint_path, **kwargs):
        super().__init__(**kwargs)
        metadata, policy_params = get_checkpoint(checkpoint_path)
        env = metamaterial.torch_env(num_envs=self.num_envs, render_text_lines=self.label, material_shape=metadata['scenario'], num_particles=metadata['n_particles'], observation_func=metadata['observation_func'], terrain_type=metadata['terrain_type'], terrain_settings=metadata['terrain_settings'], render_mode="rgb_array", window_width=self.window_width, window_height=self.window_height)
        policy_net_config = {
            'depth': 2,
            'num_cells': 256
        }
        # policy_net_config = {
        #     'depth': 3,
        #     'num_cells': [256, 128, 128]
        # }
        policy = make_nn_policy(env, metadata['algorithm'], policy_net_config, share_parameters_policy=metadata['share_parameters_policy'])
        policy.load_state_dict(policy_params)
        self.env = env
        self.policy = policy
        self.reset()

class DefaultOddElasticitySim(Sim):
    def __init__(self, scenario="crawler", n_particles=13, terrain_type="flat", terrain_settings=None, ko=-12, use_best_ko=True, **kwargs):
        super().__init__(**kwargs)
        if use_best_ko:
            self.ko = get_best_ko(scenario, n_particles, terrain_type, terrain_settings)
        else:
            self.ko = ko
        env = metamaterial.torch_env(num_envs=self.num_envs, render_text_lines=self.label, material_shape=scenario, num_particles=n_particles, observation_func="dth_tot", terrain_type=terrain_type, terrain_settings=terrain_settings, render_mode="rgb_array", window_width=self.window_width, window_height=self.window_height)
        self.env = env
        self.reset()
        
    def policy(self, tensordict):
        obs = tensordict["agents", "observation"]
        act = self.ko * obs
        # act[:,0,:] = 9
        # act[:,-1,:] = 9
        act = torch.clip(act, -9, 9)
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "action": act
                    },
                    batch_size = tensordict["agents"].shape
                )
            },
            batch_size = tensordict.shape
        )

class DefaultOddElasticitySimRoll(DefaultOddElasticitySim):
    def __init__(self, roll_first=True, roll_last=True, **kwargs):
        super().__init__(**kwargs)
        self.roll_first = roll_first
        self.roll_last = roll_last
    
    def policy(self, tensordict):
        obs = tensordict["agents", "observation"]
        act = self.ko * obs
        if self.roll_first:
            act[:,0,:] = 9
        if self.roll_last:
            act[:,-1,:] = 9
        act = torch.clip(act, -9, 9)
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "action": act
                    },
                    batch_size = tensordict["agents"].shape
                )
            },
            batch_size = tensordict.shape
        )