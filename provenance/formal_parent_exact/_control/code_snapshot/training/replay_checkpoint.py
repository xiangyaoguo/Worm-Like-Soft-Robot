import argparse
import torch
from metamaterial_envs import metamaterial
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, TanhDelta
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
import numpy as np
import matplotlib.pyplot as plt

def internode_angle(pos: np.ndarray) -> np.ndarray:
    dp = np.roll(pos, -1, 1)-pos
    dp_norm = dp/np.absolute(dp)
    return np.angle(-dp_norm/np.roll(dp_norm, 1, 1))%(2*np.pi)


def get_checkpoint(run, checkpoint):
    path = f"results/{run}/checkpoint_{checkpoint}.pt"
    saved_params = torch.load(path, weights_only=False)
    print(f"Loaded checkpoint {checkpoint} of run {run}:")
    metadata = saved_params["metadata"]
    for k, v in metadata.items():
        print(f"- {k}: {v}")
    return metadata, saved_params["policy"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('run')
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    metadata, policy_params = get_checkpoint(args.run, args.checkpoint)

    policy_net_config = {
        'depth': 2,
        'num_cells': 256
    }
    share_parameters_policy = True

    # init components based on metadata
    env = metamaterial.torch_env(num_envs=1, material_shape=metadata['scenario'], num_particles=metadata['n_particles'], observation_func=metadata['observation_func'], terrain_type=metadata['terrain_type'], terrain_settings=metadata['terrain_settings'], render=True)
    
    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],  # n_obs_per_agent
        n_agent_outputs=env.full_action_spec[env.action_key].shape[-1] * (2 if metadata['algorithm'] == 'ppo' else 1), #PPO needs 2 outputs: loc and scale
        n_agents=env.num_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=share_parameters_policy,
        device="cpu",
        activation_class=torch.nn.Tanh,
        **policy_net_config
    )
    if metadata['algorithm'] == 'ppo':
        policy_net = torch.nn.Sequential(
            policy_net,
            NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
        )
    temp_keys = [("agents", "loc"), ("agents", "scale")] if metadata['algorithm'] == 'ppo' else [("agents", "param")]
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
        distribution_class= (TanhNormal if metadata['algorithm'] == 'ppo' else TanhDelta), # PPO learns a normal distribution
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob= (metadata['algorithm'] == 'ppo'), # PPO loss needs log probs, DDPG does not
    )
    policy.load_state_dict(policy_params)

    angles = []
    torques = []

    n_steps = 1000
    n_agents = metadata['n_particles'] - 2 if metadata['scenario'] == 'crawler' else metadata['n_particles']

    td = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):
            td = policy(td)
            torques.append(td["agents","action"][0,:,0].numpy())
            td = env.step(td)["next"]
            angles.append(internode_angle(env.pos)[0])

    angles = np.array(angles)
    if metadata['scenario'] == 'crawler':
        angles = angles[:,1:-1]
    torques = np.array(torques)

    # fig, axs = plt.subplots(nrows=n_agents, ncols=1, sharex=True)
    # x_axis = np.arange(n_steps)
    # for i in range(len(axs)):
    #     axs[i].plot(x_axis, angles[:,i])
    #     axs[i].set_ylim((0, 2*np.pi))
    # plt.show()

    # print(np.max(torques), np.min(torques))

    fig, axs = plt.subplots(nrows=n_agents, ncols=1, sharex=True)
    x_axis = np.arange(n_steps)
    for i in range(len(axs)):
        axs[i].plot(x_axis, torques[:,i])
        axs[i].set_ylim((-10, 11))
        axs[i].set_yticks([0])
        axs[i].set_yticklabels([f"{i}"])
    plt.show()