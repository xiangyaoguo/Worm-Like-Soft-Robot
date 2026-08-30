import numpy as np
import cv2
from sim import DefaultOddElasticitySim, TrainedPolicySim, DefaultOddElasticitySimRoll
from copy import deepcopy

# import numpy as np
# import cv2
# size = 720*16//9, 720, 3
# duration = 2
# fps = 25
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
# for _ in range(fps * duration):
#     data = np.random.randint(0, 256, size, dtype='uint8')
#     out.write(data)
# out.release()

def make_movie(name, sims, steps=1000, fps=50):
    print('mm')
    window_width = sims[0].window_width
    window_height = sims[0].window_height
    total_height = window_height * len(sims)
    out = cv2.VideoWriter(f'videos/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (window_width, total_height), True)
    for i in range(steps):
        print(i)
        data = np.zeros((total_height, window_width, 3), dtype='uint8')
        for i in range(len(sims)):
            frame = sims[i].step_render()
            data[i*window_height:(i+1)*window_height,:,:] = frame
        out.write(cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
    out.release()

if __name__ == "__main__":
    # make_movie([
    #     DefaultOddElasticitySim(),
    #     DefaultOddElasticitySimRoll(roll_first=True, roll_last=True),
    #     DefaultOddElasticitySimRoll(roll_first=True, roll_last=False),
    #     DefaultOddElasticitySimRoll(roll_first=False, roll_last=True),
    #     TrainedPolicySim("results/974/checkpoint_300.pt")
    # ])
    # make_movie([
    #     DefaultOddElasticitySim(),
    #     # TrainedPolicySim("results/1045/checkpoint_500.pt"),
    #     TrainedPolicySim("results/1054/checkpoint_500.pt"),
    #     TrainedPolicySim("results/1058/checkpoint_500.pt"),
    #     TrainedPolicySim("results/1065/checkpoint_500.pt")
    # ])
    # make_movie("crawler_ddpg_stairs", [
    #     DefaultOddElasticitySim(terrain_type="mesh", terrain_settings={"type":"stairs"}, num_envs=5),
    #     TrainedPolicySim("results/1107/checkpoint_500.pt", num_envs=5),
    #     TrainedPolicySim("results/1109/checkpoint_500.pt", num_envs=5),
    #     TrainedPolicySim("results/1113/checkpoint_500.pt", num_envs=5),
    #     TrainedPolicySim("results/1119/checkpoint_500.pt", num_envs=5)
    # ], steps=3000, fps=100)
    # make_movie("crawler_ppo_stairs", [
    #     DefaultOddElasticitySim(terrain_type="mesh", terrain_settings={"type":"stairs"}, num_envs=5),
    #     TrainedPolicySim("results/1108/checkpoint_500.pt", num_envs=5),
    #     TrainedPolicySim("results/1111/checkpoint_500.pt", num_envs=5),
    #     TrainedPolicySim("results/1112/checkpoint_500.pt", num_envs=5),
    #     TrainedPolicySim("results/1115/checkpoint_500.pt", num_envs=5)
    # ], steps=3000, fps=100)
    # make_movie("ring_ddpg_stairs", [
    #     DefaultOddElasticitySim(terrain_type="mesh", terrain_settings={"type":"stairs"}, num_envs=5, scenario="ring", window_height=350),
    #     TrainedPolicySim("results/1104/checkpoint_500.pt", num_envs=5, window_height=350),
    #     TrainedPolicySim("results/1105/checkpoint_500.pt", num_envs=5, window_height=350),
    #     TrainedPolicySim("results/1106/checkpoint_500.pt", num_envs=5, window_height=350),
    #     TrainedPolicySim("results/1116/checkpoint_500.pt", num_envs=5, window_height=350)
    # ], steps=2000, fps=100)
    # make_movie("ring_ppo_stairs", [
    #     DefaultOddElasticitySim(terrain_type="mesh", terrain_settings={"type":"stairs"}, num_envs=5, scenario="ring", window_height=350),
    #     TrainedPolicySim("results/1110/checkpoint_500.pt", num_envs=5, window_height=350),
    #     TrainedPolicySim("results/1114/checkpoint_500.pt", num_envs=5, window_height=350),
    #     TrainedPolicySim("results/1117/checkpoint_500.pt", num_envs=5, window_height=350),
    #     TrainedPolicySim("results/1118/checkpoint_500.pt", num_envs=5, window_height=350)
    # ], steps=2000, fps=100)
    # make_movie("crawler_ppo_indep_2k", [
    #     DefaultOddElasticitySim(num_envs=5, n_particles=13),
    #     TrainedPolicySim("results/1124/checkpoint_800.pt", num_envs=5),
    #     DefaultOddElasticitySim(num_envs=5, n_particles=10),
    #     TrainedPolicySim("results/1129/checkpoint_475.pt", num_envs=5),
    #     DefaultOddElasticitySim(num_envs=5, n_particles=7),
    #     TrainedPolicySim("results/1142/checkpoint_300.pt", num_envs=5),
    #     DefaultOddElasticitySim(num_envs=5, n_particles=5),
    #     TrainedPolicySim("results/1141/checkpoint_400.pt", num_envs=5)
    # ], steps=2000, fps=100)
    # make_movie("ring_flat_15", [
    #     DefaultOddElasticitySim(num_envs=5, n_particles=15, scenario="ring", label=["default odd elasticity"], window_width=width),
    #     TrainedPolicySim("results/1733/checkpoint_550.pt", num_envs=5, label=["DDPG", "Shared parameters"], window_width=width),
    #     TrainedPolicySim("results/1772/checkpoint_600.pt", num_envs=5, label=["DDPG", "Independent parameters"], window_width=width),
    #     TrainedPolicySim("results/1708/checkpoint_200.pt", num_envs=5, label=["PPO", "Shared parameters"], window_width=width),
    #     TrainedPolicySim("results/1776/checkpoint_150.pt", num_envs=5, label=["PPO", "Independent parameters"], window_width=width)
    # ], steps=1000, fps=100)
    width = 1500
    height = 100
    best_checkpoints = {'flat': (2047, 650),
 'stairs': (2028, 750),
 'tunnel': (2025, 750),
 'mix': (2056, 950)}
    sims = {k: TrainedPolicySim(f"results/{v[0]}/checkpoint_{v[1]}.pt", num_envs=10, window_width=width, window_height=height, label=[f"Training env: {k}"]) for k, v in best_checkpoints.items()}
    envs = {k: v.env for k, v in sims.items()}
    for challenge in ["flat"]:
        for k, v in sims.items():
            v.env = deepcopy(envs[challenge])
            v.env.render_text_lines = v.label
        make_movie(f"tunnel_{challenge}", list(sims.values()), steps=1000, fps=100)