from tensordict import TensorDict
from metamaterial_envs import crawler_v0
import torch
from tqdm import tqdm
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

num_envs = 10
max_steps = 1000
num_particles = 13

env = crawler_v0.torch_env(num_envs=num_envs, max_steps=max_steps, num_particles=num_particles)

levels = 10

k_range = (-20,20)
ko_range = (-50, 0)

file = f"temp/grid_search_result_{num_particles}_{levels}_{k_range}_{ko_range}.npy"
if os.path.isfile(file):
    results = np.load(file)
else:
    pbar = tqdm(total=levels**2)
    results = np.zeros((levels, levels))
    i = 0
    for k in np.linspace(*k_range, levels):
        j = 0
        for ko in np.linspace(*ko_range, levels):
            def actor(tensordict):
                device = "cpu"
                obs = tensordict["agents", "observation"]
                act = ko * obs[:,:,0] + k * obs[:,:,1]
                act = torch.clip(act, -9, 9)
                return TensorDict(
                    {
                        "agents": TensorDict(
                            {
                                "action": act
                            },
                            batch_size = tensordict["agents"].shape,
                            device = device
                        )
                    },
                    batch_size = tensordict.shape,
                    device = device
                )
            td = env.reset()
            n_iter = 1000
            for _ in range(n_iter):
                action = actor(td)
                td = env.step(action)["next"]
            
            results[i][j] = torch.real(env.pos).mean().item()

            pbar.update()
            j += 1
        i += 1
    pbar.close()
    np.save(file, results)

heatmap = plt.pcolor(results)
plt.yticks(np.arange(levels)+0.5,["{:.2f}".format(x) for x in np.linspace(*k_range, levels)])
plt.ylabel("$\kappa$")
plt.xticks(np.arange(levels)+0.5,["{:.2f}".format(x) for x in np.linspace(*ko_range, levels)], rotation=30)
plt.xlabel("$\kappa_o$")

formula = "$τ_1=Κδθ_1+Κ_oδθ_2$ and $τ_2=Κδθ_2-Κ_oδθ_1$" if num_particles == 4 else "$τ_i=Κδθ_i+Κ_o(δθ_{i+1} - δθ_{i-1})$"

plt.title(f"Distance travelled by odd crawler with {num_particles} nodes\n{formula}")
plt.colorbar(heatmap)
plt.tight_layout()
plt.savefig('grid_search.png')