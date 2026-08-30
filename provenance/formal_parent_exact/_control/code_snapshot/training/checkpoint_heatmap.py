import argparse
import torch
from metamaterial_envs import metamaterial
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, TanhDelta
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from tensordict import TensorDict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_checkpoint(run, checkpoint):
    path = f"results/{run}/checkpoint_{checkpoint}.pt"
    saved_params = torch.load(path)#, weights_only=True)
    print(f"Loaded checkpoint {checkpoint} of run {run}:")
    metadata = saved_params["metadata"]
    for k, v in metadata.items():
        print(f"- {k}: {v}")
    return metadata, saved_params["policy"]


@torch.no_grad()
def sample_activation_heatmap(policy, env, levels=20, minv=-torch.pi, maxv=torch.pi):
    assert env.observation_spec['agents', 'observation'].shape[-1] == 2
    print("asserted")

    batch_size = (levels, levels)
    n_agents = env.num_agents
    x = torch.linspace(minv, maxv, levels)
    obs = torch.stack(torch.meshgrid(x, x)).transpose(0, 2).unsqueeze(2).repeat(1, 1, n_agents, 1)
    td = TensorDict(
        {
            "agents": TensorDict(
                {
                    "observation": obs
                },
                batch_size = torch.Size([*batch_size, n_agents]),
                device = env.device
            )
        },
        batch_size = torch.Size([*batch_size]),
        device = env.device
    )

    print("before sample")
    heatmap = policy(td)['agents', 'action'].squeeze(-1).cpu().numpy()
    print("after sample")
    return heatmap

@torch.no_grad()
def sample_activation_heatmap_slice(policy, env, thdot, levels=20, minv=-torch.pi, maxv=torch.pi):
    assert env.observation_spec['agents', 'observation'].shape[-1] == 3

    batch_size = (levels, levels)
    n_agents = env.num_agents
    x = torch.linspace(minv, maxv, levels)
    grid = torch.meshgrid(x, x)
    obs = torch.stack(grid).transpose(0, 2).unsqueeze(2).repeat(1, 1, n_agents, 1)
    obs = torch.cat((obs, torch.full_like(obs[:,:,:,0:1], thdot)), dim=-1)
    print(obs.shape)
    td = TensorDict(
        {
            "agents": TensorDict(
                {
                    "observation": obs
                },
                batch_size = torch.Size([*batch_size, n_agents]),
                device = env.device
            )
        },
        batch_size = torch.Size([*batch_size]),
        device = env.device
    )

    heatmap = policy(td)['agents', 'action'].squeeze(-1).cpu().numpy()
    return heatmap

def load_policy_env(run_id, t):
    metadata, policy_params = get_checkpoint(run_id, t)


    # init components based on metadata
    env = metamaterial.torch_env(num_envs=10, material_shape=metadata['scenario'], num_particles=metadata['n_particles'], observation_func=metadata['observation_func'], terrain_type=metadata['terrain_type'], terrain_settings=metadata['terrain_settings'], render=True)
    
    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],  # n_obs_per_agent
        n_agent_outputs=env.full_action_spec[env.action_key].shape[-1] * (2 if metadata['algorithm'] == 'ppo' else 1), #PPO needs 2 outputs: loc and scale
        n_agents=env.num_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=metadata['share_parameters_policy'],
        device="cpu",
        activation_class=torch.nn.Tanh,
        **metadata['policy_net_config']
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
    print("Finished loading")
    return policy, env

def heatmap_of_checkpoint(run_id, t, levels=400):
    policy, env = load_policy_env(run_id, t)

    hm = sample_activation_heatmap(policy, env, levels=levels)[:,:,0]
    return hm

def heatmaps_of_checkpoint_indep(run_id, t, levels=400):
    policy, env = load_policy_env(run_id, t)

    hms = []
    s = sample_activation_heatmap(policy, env, levels=levels)#[:,:,0]
    for i in range(s.shape[-1]):
        hms.append(s[:,:,i])
    return hms

def heatmap_slices_of_checkpoint(run_id, t, thdot_min=-1, thdot_max=1, thdot_levels=2, levels=400):
    policy, env = load_policy_env(run_id, t)

    hms = []
    for thdot in np.linspace(thdot_min, thdot_max, thdot_levels):
        hms.append(sample_activation_heatmap_slice(policy, env, thdot, levels=levels)[:,:,0])
    return hms



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('run')
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    metadata, _ = get_checkpoint(args.run, args.checkpoint)

    if metadata["share_parameters_policy"] == False:
        hms = heatmaps_of_checkpoint_indep(args.run, args.checkpoint)
        fig, axs = plt.subplots(1,len(hms),figsize=(len(hms)*2, 2))
        for i in range(len(hms)):
            axs[i].pcolor(hms[i], cmap = "RdBu", vmin=-9, vmax=9)
        plt.show()
    elif metadata["observation_func"] == "dth_neighbours":
        hm = heatmap_of_checkpoint(args.run, args.checkpoint)
        fig = plt.figure()
        plt.pcolor(hm[::-1,:], cmap = "RdBu", vmin=-9, vmax=9)
        plt.show()
    # ax = sns.heatmap(hm, annot=False,  linewidths=0, cmap='RdBu')
    # ax.set_xticks(np.linspace(0, hm.shape[0], 3))
    # ax.set_yticks(np.linspace(0, hm.shape[0], 3))
    # ax.set_xticklabels(["{:.1f}".format(i) for i in np.linspace(-np.pi, np.pi, 3)])
    # ax.set_yticklabels(["{:.1f}".format(i) for i in np.linspace(-np.pi, np.pi, 3)])
    # ax.set(xlabel="Angle 1", ylabel="Angle 2", title="NN output value")
    # plt.show()
    else:
        hms = heatmap_slices_of_checkpoint(args.run, args.checkpoint, -10, 10, 10)
        fig, axs = plt.subplots(1,len(hms),figsize=(len(hms)*2, 2))
        for i in range(len(hms)):
            axs[i].pcolor(hms[i], cmap = "RdBu", vmin=-9, vmax=9)
        plt.show()
