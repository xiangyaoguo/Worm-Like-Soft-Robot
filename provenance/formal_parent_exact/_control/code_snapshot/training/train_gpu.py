import tempfile
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "metamaterial_envs"))

log_to_atlas = False

import torch

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement, RandomSampler
from torchrl.data.replay_buffers.storages import LazyTensorStorage, LazyMemmapStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, TanhDelta, AdditiveGaussianModule

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators, DDPGLoss, SoftUpdate

# Utils
from tensordict import TensorDictBase, TensorDict
from matplotlib import pyplot as plt
from tqdm import tqdm

from sacred import Experiment
if log_to_atlas:
    from mongo_atlas import mongo_atlas_observer
'''from metamaterial_envs import crawler_v0, ring_v0, metamaterial'''
'''from metamaterial_envs import metamaterial'''
from metamaterial_envs.env import metamaterial

import numpy as np
import math

import sys
print(sys.argv)

def td_contains_nan(tensordict: TensorDict) -> bool:
    for key, value in tensordict.items():
        if isinstance(value, TensorDict):
            if td_contains_nan(value):  # recursive check
                return True
        elif isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                return True
    return False
def td_contains_inf(tensordict: TensorDict) -> bool:
    for key, value in tensordict.items():
        if isinstance(value, TensorDict):
            if td_contains_inf(value):  # recursive check
                return True
        elif isinstance(value, torch.Tensor):
            if torch.isinf(value).any():
                return True
    return False

def process_batch(batch: TensorDictBase) -> TensorDictBase:
    """
    If the `(group, "terminated")` and `(group, "done")` keys are not present, create them by expanding
    `"terminated"` and `"done"`.
    This is needed to present them with the same shape as the reward to the loss.
    """
    keys = list(batch.keys(True, True))
    group_shape = batch.get_item_shape("agents")
    for key in ["done", "reward", "terminated"]:
        nested_key = ("next", "agents", key)
        if nested_key not in keys:
            batch.set(
            nested_key,
            batch.get(("next", key)).unsqueeze(-1).expand((*group_shape, 1)),
        )
    return batch

ex = Experiment('TorchRL-Metamaterial')
if log_to_atlas:
    ex.observers.append(mongo_atlas_observer)

@ex.config
def my_config():
    episodes = 10
    episode_steps = 100
    save_every = 50
    pretrained_model_path = None
    n_particles = 13
    scenario = 'ring'
    terrain_type = "flat"
    terrain_settings = None
    observation_func = 'dth_tot'
    memory_size = 1_000_000  # The replay buffer of each group can store this many frames
    frames_per_batch = 10_000  # Number of team frames collected per sampling iteration
    render = False

    share_parameters_policy = True
    share_parameters_critic = True
    centralised_critic = True  # IDDPG or IPPO if False

    # Training
    n_optimiser_steps = 10  # Number of optimization steps per training iteration
    minibatch_size = 128  # Size of the mini-batches in each optimization step
    lr = 3e-4  # Learning rate
    weight_decay = 1e-4 # L2 regularisation
    max_grad_norm = 1.0  # Maximum norm for the gradients

    algorithm = 'ppo' # ddpg or ppo
    gamma = 0.99  # Discount factor

    # DDPG
    polyak_tau = 0.005  # Tau for the soft-update of the target network
    expl_noise = [0.9, 0.1] # max and min exploration noise

    # PPO
    clip_epsilon = 0.2  # clip value for PPO loss
    lmbda = 0.9  # lambda for generalised advantage estimation
    entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

    policy_net_config = {
        'depth': 2,
        'num_cells': 256
    }
    critic_net_config = policy_net_config

    # implementation details
    buffer_storage = 'tensor' # tensor or memmap
    buffer_sample_with_replacement = (algorithm == 'ddpg')
    force_cpu = False  # GPU training: use CUDA automatically when available. Override with `with force_cpu=True` if needed.
    gaussian_activation = False
    normal_scale_lb = 1e-4

class FirstOrderGaussian(torch.nn.Module):

    def __init__(self, std=1):
        super().__init__()
        self.std = std

    def forward(self, x):
        return 0.5 * torch.pi * x * torch.exp((-x ** 2)/(2* self.std**2)) / self.std

class SafeTanhNormal(TanhNormal):
    def log_prob(self, value):
        # Get log-probabilities from Normal distribution.
        log_prob = super().log_prob(value)
        # Clip the log probabilities as a safeguard
        # log_prob = torch.clamp(log_prob, -20, 20)
        epsilon = 1e-6
        log_prob = torch.log(torch.exp(log_prob) + epsilon)
        return log_prob

minmax_scale = [[],[]]
class BiasedNormalParamExtractor(NormalParamExtractor):
    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        loc, scale = tensor.chunk(2, -1)
        scale = self.scale_mapping(scale) + self.scale_lb # add instead of min clamp
        # Avoid calling .item() inside every GPU forward pass: it synchronises CUDA and slows training heavily.
        # Keep the old logging behaviour only when the tensor is already on CPU.
        if scale.device.type == "cpu":
            minmax_scale[0].append(torch.min(scale).item())
            minmax_scale[1].append(torch.max(scale).item())
        return (loc, scale, *others)

@ex.capture
def components(env, device,
               algorithm, episodes, pretrained_model_path,
               share_parameters_policy, policy_net_config, expl_noise,
               share_parameters_critic, critic_net_config, centralised_critic,
               frames_per_batch, buffer_storage, buffer_sample_with_replacement, minibatch_size, memory_size,
               gamma, polyak_tau, clip_epsilon, entropy_eps, lmbda, lr, weight_decay, gaussian_activation, normal_scale_lb):
    """
    Returns the policies, critic, loss module, collector, and replay buffer
    """
    assert algorithm in ['ppo', 'ddpg']

    ### policy ###
    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],  # n_obs_per_agent
        n_agent_outputs=env.full_action_spec[env.action_key].shape[-1] * (2 if algorithm == 'ppo' else 1), #PPO needs 2 outputs: loc and scale
        n_agents=env.num_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=share_parameters_policy,
        device=device,
        activation_class=FirstOrderGaussian if gaussian_activation else torch.nn.Tanh,
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
    # The environment specs are created on CPU. When the neural networks run on CUDA,
    # move the actor spec bounds to the same device as the network outputs.
    action_spec = env.action_spec_unbatched.to(device) if hasattr(env.action_spec_unbatched, "to") else env.action_spec_unbatched
    action_low = env.full_action_spec_unbatched[env.action_key].space.low.to(device)
    action_high = env.full_action_spec_unbatched[env.action_key].space.high.to(device)

    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=temp_keys,
        out_keys=[env.action_key],
        distribution_class= (SafeTanhNormal if algorithm == 'ppo' else TanhDelta), # PPO learns a normal distribution
        distribution_kwargs={
            "low": action_low,
            "high": action_high,
        },
        return_log_prob= (algorithm == 'ppo'), # PPO loss needs log probs, DDPG does not
    )
    if algorithm == 'ddpg':
        # DDPG is deterministic and needs a separate exploration policy
        exploration_policy = TensorDictSequential(
            policy,
            AdditiveGaussianModule(
                spec=policy.spec,
                annealing_num_steps=frames_per_batch * episodes // 2,  # Number of frames after which sigma is sigma_end
                action_key=("agents", "action"),
                sigma_init=expl_noise[0],  # Initial value of the sigma
                sigma_end=expl_noise[1],  # Final value of the sigma
            ),
        )
    else:
        exploration_policy = policy
    

    ### critic ###
    critic_value_type = "state_action_value" if algorithm == 'ddpg' else 'state_value'
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1] + (env.full_action_spec["agents", "action"].shape[-1] if critic_value_type == 'state_action_value' else 0),
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.num_agents,
        centralised=centralised_critic,
        share_params=share_parameters_critic,
        device=device,
        activation_class=torch.nn.Tanh,
        **critic_net_config
    )
    if critic_value_type == 'state_action_value':
        critic = TensorDictSequential(
            TensorDictModule(
                lambda obs, action: torch.cat([obs, action], dim=-1),
                in_keys=[("agents", "observation"), ("agents", "action")],
                out_keys=[("agents", "obs_action")],
            ),
            TensorDictModule(
                module=critic_net,
                in_keys=[("agents", "obs_action")],
                out_keys=[("agents", "state_action_value")]
            )
        )
    else:
        critic = TensorDictModule(
            module=critic_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")]
        )
    
    if pretrained_model_path is not None:
        load_params({"policy": policy, "critic": critic}, pretrained_model_path)

    ### data collector ###
    collector = SyncDataCollector(
        env,
        exploration_policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * episodes,
    )


    ### replay buffer ###
    assert buffer_storage in ['memmap', 'tensor']
    buffer_sampler_class = RandomSampler if buffer_sample_with_replacement else SamplerWithoutReplacement
    buffer_memory_size = memory_size if buffer_sample_with_replacement else frames_per_batch
    if buffer_storage == 'tensor':
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                buffer_memory_size, device=device
            ),
            sampler=buffer_sampler_class(),
            batch_size=minibatch_size,  # We will sample minibatches of this size
        )
    else:
        scratch_dir = tempfile.TemporaryDirectory().name
        replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(
                buffer_memory_size, scratch_dir=scratch_dir,
            ),
            sampler=buffer_sampler_class(),
            batch_size=minibatch_size,  # We will sample batches of this size
        )
        if device.type != "cpu":
            replay_buffer.append_transform(lambda x: x.to(device))
    

    ### loss module ###
    if algorithm == 'ppo':
        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            clip_epsilon=clip_epsilon,
            entropy_coeff=entropy_eps,
            normalize_advantage=False,
        )
        loss_module.set_keys(  # We have to tell the loss where to find the keys
            reward=("agents", "reward"),
            action=env.action_key,
            value=("agents", "state_value"),
            # These last 2 keys will be expanded to match the reward shape
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
        )  # We build GAE
        GAE = loss_module.value_estimator
        optimiser = torch.optim.Adam(loss_module.parameters(), lr, weight_decay=weight_decay)
        target_updater = None
    elif algorithm == 'ddpg':
        loss_module = DDPGLoss(
            actor_network=policy,  # Use the non-explorative policies
            value_network=critic,
            delay_value=True,  # Whether to use a target network for the value
            loss_function="l2",
        )
        loss_module.set_keys(
            state_action_value=("agents", "state_action_value"),
            reward=("agents", "reward"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)

        target_updater = SoftUpdate(loss_module, tau=polyak_tau)
        optimiser = {
            "loss_actor": torch.optim.Adam(
                loss_module.actor_network_params.flatten_keys().values(), lr=lr, weight_decay=weight_decay
            ),
            "loss_value": torch.optim.Adam(
                loss_module.value_network_params.flatten_keys().values(), lr=lr, weight_decay=weight_decay
            ),
        }
        GAE = None
    
    return policy, exploration_policy, critic, collector, replay_buffer, loss_module, optimiser, GAE, target_updater

def save_params(obj, path):
    serialised = {}
    for k,v in obj.items():
        if k == "metadata":
            serialised[k] = v
        else:
            serialised[k] = v.state_dict()
    torch.save(serialised, path)

def load_params(obj, path):
    saved_params = torch.load(path, weights_only=True)
    if "metadata" in saved_params:
        del saved_params["metadata"]
    for module_name, module in obj.items():
        expected_state_dict = module.state_dict()
        saved_state_dict = saved_params[module_name]
        for parameter_name, parameter in expected_state_dict.items():
            if isinstance(parameter, torch.Tensor):
                expected_shape = parameter.shape
                saved_shape = saved_state_dict[parameter_name].shape
                if expected_shape == saved_shape:
                    pass
                elif expected_shape[1:] == saved_shape:
                    saved_params[module_name][parameter_name] = saved_params[module_name][parameter_name].unsqueeze(0).repeat([expected_shape[0]] + [1]*(len(expected_shape)-1))
                else:
                    raise ValueError(f"Cannot load parameter {module_name}/{parameter_name}: expected shape {expected_shape}, found {saved_shape}")
            else:
                saved_params[module_name][parameter_name] = parameter
        module.load_state_dict(saved_params[module_name])
        print(f"Loaded {module_name} params from {path}")

@ex.automain
def main(_run, force_cpu, episodes, episode_steps, n_particles, render, observation_func, seed, n_optimiser_steps,
         frames_per_batch, minibatch_size, scenario, terrain_type, terrain_settings, algorithm, max_grad_norm, save_every,
         policy_net_config, share_parameters_policy):
    save_dir = Path("results") / str(_run._id)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Devices
    # force_cpu=False means: use CUDA GPU if your PyTorch installation can see it.
    # The custom metamaterial physics environment still runs its NumPy/Numba simulation on CPU,
    # but the policy, critic, loss, replay buffer tensors, and optimisation will be placed on CUDA.
    is_fork = multiprocessing.get_start_method() == "fork"
    cuda_available = torch.cuda.is_available()

    if force_cpu:
        device = torch.device("cpu")
        print("force_cpu=True, using CPU.")
    elif cuda_available and not is_fork:
        device = torch.device("cuda:0")
        print("CUDA is available, using GPU.")
        print("CUDA device:", torch.cuda.get_device_name(device))
        print("CUDA version used by PyTorch:", torch.version.cuda)
    else:
        device = torch.device("cpu")
        if not cuda_available:
            print("CUDA is not available in this Python environment; falling back to CPU.")
        elif is_fork:
            print("Multiprocessing start method is 'fork'; falling back to CPU for safety.")

    print('Using device:', device)
    # Keep the global default tensor device on CPU. The custom metamaterial env constructs
    # tensors from NumPy internally, so setting the global default to CUDA can create
    # CPU/GPU device mismatches inside env reset/step. Networks and buffers use `device` explicitly.
    torch.set_default_device("cpu")
    
    # disable log-prob aggregation
    set_composite_lp_aggregate(False).set()

    # Sampling
    num_parallel_envs = (
        frames_per_batch // episode_steps
    )  # Number of vectorized environments. frames_per_batch collection will be divided among these environments

    base_env = metamaterial.env(num_envs=num_parallel_envs, material_shape=scenario, num_particles=n_particles, observation_func=observation_func, terrain_type=terrain_type, terrain_settings=terrain_settings, render=render)
    env = TransformedEnv(
        base_env,
        RewardSum(
            in_keys=base_env.reward_keys,
            reset_keys=["_reset"],
        ),
    )
    check_env_specs(env)
    print('PASSED CHECK!')

    policy, exploration_policy, critic, collector, replay_buffer, loss_module, optimiser, GAE, target_updater = components(env, device)

    pbar = tqdm(total=episodes, desc="episode_reward_mean = 0")

    episode_reward_mean_list = []

    current_episode = 0
    for iteration, batch in enumerate(collector):
        current_episode = iteration + 1
        log_info = batch["log_info"]
        _run.log_scalar("training.mean_speed", log_info["speed"].mean().item(), iteration)
        if len(minmax_scale[0]) > 0:
            _run.log_scalar("normal_scale.min", np.min(minmax_scale[0]), iteration)
            _run.log_scalar("normal_scale.max", np.min(minmax_scale[1]), iteration)
        minmax_scale[0] = []
        minmax_scale[1] = []
        with torch.no_grad():
            _run.log_scalar("policy_params.min", np.min([torch.min(p).item() for p in policy.parameters()]), iteration)
            _run.log_scalar("policy_params.max", np.max([torch.max(p).item() for p in policy.parameters()]), iteration)

        current_frames = batch.numel()
        batch = process_batch(batch)
        if GAE is not None:
            with torch.no_grad():
                GAE(
                    batch,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )  # Compute GAE and add it to the data, only for PPO
        replay_buffer.extend(batch.reshape(-1))

        for _ in range((n_optimiser_steps * frames_per_batch) // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            if algorithm == 'ppo':
                losses, params, optims = [
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]], [loss_module.parameters()], [optimiser]
            elif algorithm == 'ddpg':
                losses, params, optims = [
                    loss_vals[loss_name] for loss_name in ["loss_actor", "loss_value"]
                ], [
                    optimiser[loss_name].param_groups[0]["params"] for loss_name in ["loss_actor", "loss_value"]
                ], [
                    optimiser[loss_name] for loss_name in ["loss_actor", "loss_value"]
                ]
            
            for loss, param, optim in zip(losses, params, optims):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(param, max_grad_norm)
                optim.step()
                optim.zero_grad()
            
            if target_updater is not None:
                target_updater.step() # only for DDPG
        
        if algorithm == 'ddpg':
            # Exploration sigma anneal update
            exploration_policy[-1].step(current_frames)
        elif algorithm == 'ppo':
            collector.update_policy_weights_()
        
        if (current_episode % save_every) == 0:
            save_params({"policy": policy, "critic": critic, "metadata":{
                "scenario":scenario,
                "algorithm":algorithm,
                "n_particles":n_particles,
                "observation_func":observation_func,
                "terrain_type": terrain_type,
                "terrain_settings": terrain_settings,
                "policy_net_config": policy_net_config,
                "share_parameters_policy": share_parameters_policy
            }}, save_dir / f"checkpoint_{current_episode}.pt")

        done = batch.get(("next", "done"))
        episode_reward_mean = (
            batch.get(("next", "episode_reward"))[done].mean().item()
        )
        episode_reward_mean_list.append(episode_reward_mean)
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
        pbar.update()

    if not (current_episode % save_every) == 0:
        save_params({"policy": policy, "critic": critic, "metadata":{
            "scenario":scenario,
            "algorithm":algorithm,
            "n_particles":n_particles,
            "observation_func":observation_func,
            "terrain_type": terrain_type,
            "terrain_settings": terrain_settings,
            "policy_net_config": policy_net_config,
            "share_parameters_policy": share_parameters_policy
        }}, save_dir / f"checkpoint_{current_episode}.pt")