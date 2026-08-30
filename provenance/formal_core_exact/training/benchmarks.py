from tensordict import TensorDict
from metamaterial_envs import metamaterial
import torch
import json
import os
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

cache_file_path = 'temp/benchmarks_cache.json'

class OddActor:
    def __init__(self, ko=-12):
        self.ko = ko
    
    def __call__(self, td: TensorDict):
        device = "cpu"
        obs = td["agents", "observation"]
        act = (self.ko * obs).type(torch.float32)
        act = torch.clip(act, -9, 9)
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "action": act
                    },
                    batch_size = td["agents"].shape,
                    device = device
                )
            },
            batch_size = td.shape,
            device = device
        )

class OddActorMultiKo:
    def __init__(self, ko_min=-50, ko_max=50, steps=101, num_envs_per_ko=10):
        self.kos = torch.tensor(np.repeat(np.linspace(ko_min, ko_max, steps), num_envs_per_ko)).unsqueeze(-1).unsqueeze(-1)
        self.num_envs_per_ko = num_envs_per_ko
        self.total_envs = steps * num_envs_per_ko
        self._ko_min = ko_min
        self._ko_max = ko_max
        self._steps = steps

    
    def __call__(self, td: TensorDict):
        device = "cpu"
        obs = td["agents", "observation"]
        act = (obs * self.kos).type(torch.float32)
        act = torch.clip(act, -9, 9)
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "action": act
                    },
                    batch_size = td["agents"].shape,
                    device = device
                )
            },
            batch_size = td.shape,
            device = device
        )
    
    def evaluate_speeds(self, speeds):
        speeds_ko = speeds.reshape((self._steps, self.num_envs_per_ko)).mean(axis=1)
        best_speed = np.max(speeds_ko)
        best_ko = np.linspace(self._ko_min, self._ko_max, self._steps)[np.argmax(speeds_ko)]
        return best_ko, best_speed

if os.path.isfile(cache_file_path):
    with open(cache_file_path, 'r') as f:
        cache = json.load(f)
else:
    cache = {}

def _run_episode(actor, scenario, nodes, terrain_type, terrain_settings, num_envs, steps=1000):
    env = metamaterial.torch_env(num_envs=num_envs, max_steps=steps, num_particles=nodes, material_shape=scenario, terrain_type=terrain_type, terrain_settings=terrain_settings, observation_func="dth_tot")
    td = env.reset()
    env.step(actor(td))
    td = env.reset()
    speeds = []
    for _ in tqdm(range(steps)):
        action = actor(td)
        td = env.step(action)["next"]
        speeds.append(td["log_info", "speed"].numpy())
        env.render()
    return np.array(speeds).mean(axis=0)


def _run_benchmark(scenario: str, n_particles: int, terrain_type="flat", terrain_settings=None):
    key = f"{scenario}_{n_particles}_{terrain_type}({json.dumps(terrain_settings)})"
    actor = OddActorMultiKo()
    speeds = _run_episode(actor, scenario, n_particles, terrain_type, terrain_settings, num_envs=actor.total_envs)
    best_ko, best_speed = actor.evaluate_speeds(speeds)
    best_speed = np.mean(_run_episode(OddActor(best_ko), scenario, n_particles, terrain_type, terrain_settings, num_envs=100))
    cache[key] = {'ko': float(best_ko), 'speed': float(best_speed)}
    with open(cache_file_path, 'w') as f:
        json.dump(cache, f)


def get_benchmark_speed(scenario: str, n_particles: int, terrain_type="flat", terrain_settings=None):
    key = f"{scenario}_{n_particles}_{terrain_type}({json.dumps(terrain_settings)})"
    if key not in cache:
        _run_benchmark(scenario, n_particles, terrain_type, terrain_settings)
    return cache[key]['speed']

def get_best_ko(scenario: str, n_particles: int, terrain_type="flat", terrain_settings=None):
    key = f"{scenario}_{n_particles}_{terrain_type}({json.dumps(terrain_settings)})"
    if key not in cache:
        _run_benchmark(scenario, n_particles, terrain_type, terrain_settings)
    return cache[key]['ko']

if __name__ == '__main__':
    for n in [5, 10, 15]:
    # for n in range(1,10):
    #     n = n * 5
        print(f"benchmark speed for ring with {n} nodes:", "{:.2f}".format(get_benchmark_speed("ring", n)*100))