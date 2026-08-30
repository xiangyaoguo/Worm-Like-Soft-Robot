from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from copy import copy
from metamaterial_envs.env.crawler_simulation import pos_angleeq_to_obs_dth3, pos_angleeq_to_obs_dth_tot, pos_angleeq_to_obs_dth3_vel, get_initial, velocity_verlet_with_thdot, pos_angleeq_to_obs_small
import numpy as np
from enum import Enum
from types import MethodType
import pygame
import math

def get_obs_dth3(self):
    return dict(pos_angleeq_to_obs_dth3(self.pos, self.angle_eq))

def get_obs_dth_tot(self):
    return dict(pos_angleeq_to_obs_dth_tot(self.pos, self.angle_eq))

def get_obs_dth3_vel(self):
    return dict(pos_angleeq_to_obs_dth3_vel(self.pos, self.angle_eq, self.thdot))

def get_obs_small(self):
    return dict(pos_angleeq_to_obs_small(self.pos, self.angle_eq))

class ObsConfig(Enum):
    DTH3 = (Box(-np.pi, np.pi, shape=(3,), dtype=np.float32), get_obs_dth3)
    DTH_TOT = (Box(-np.pi, np.pi, shape=(1,), dtype=np.float32), get_obs_dth_tot)
    DTH3_VEL = (Box(-np.pi, np.pi, shape=(6,), dtype=np.float32), get_obs_dth3_vel)
    SMALL = (Box(-np.pi, np.pi, shape=(2,), dtype=np.float32), get_obs_small) # can only be used with 4 nodes (2 agents)

    def __init__(self, space, get_obs_func):
        self.space = space
        self.get_obs_func = get_obs_func
    
    def __str__(self):
        return self.name


def reward_func_delta_pos(self):
    reward = self.mean_speed * 100
    return reward

def reward_func_delta_pos_centre_node(self):
    reward = np.real(self.pos[self.n_particles//2] - self.last_pos[self.n_particles//2]) * 100
    return reward

def reward_func_dont_be_lazy(self):
    # give negative reward if angular velocity is low
    mean_abs_thdot = np.mean(np.abs(self.thdot))
    if mean_abs_thdot < 0.3:
        reward = -1
    else:
        reward = self.mean_speed * 100
    return reward

def reward_func_delta_pos_large(self):
    reward = self.mean_speed * 1e8
    return reward

class RewardFuncConfig(Enum):
    DELTA_POS = (reward_func_delta_pos,)
    DELTA_POS_CENTRE_NODE = (reward_func_delta_pos_centre_node,)
    DONT_BE_LAZY = (reward_func_dont_be_lazy,)
    DELTA_POS_LARGE = (reward_func_delta_pos_large,)

    def __init__(self, func):
        self.func = func
    
    def __str__(self):
        return self.name


class simple_env(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "crawler_v0",
        "render_fps": 50
    }

    def __init__(self, n_particles: int=13, max_steps: int=1000, observarion=ObsConfig.DTH3_VEL, reward_func=RewardFuncConfig.DELTA_POS, render_mode=None):
        self.timestep = None
        self.n_particles = n_particles
        self.max_steps = max_steps
        self.possible_agents = list(range(n_particles-2))

        self.pos = None
        self.vel = None
        self.camera_pos = None
        self.camera_vel = None
        self.thdot = None
        self.last_pos = None
        self.mean_speed = None
        self.last_reward = None
        self.reward_history = None
        self.render_mode = render_mode
        pygame.init()
        self.window = None
        self.window_size = 512
        self.clock = None

        # simulation parameters
        self.particle_randomness = 0.1
        self.particle_radius = 1/3.0
        self.particle_mass = 0.2

        self.dt = 3e-3 #integration time step
        self.physics_steps_per_timestep = 10 # amount of integration timesteps per observation-action

        self.max_motor_torque = 9.0

        self.gravity_constant = -1j*1
        self.background_friction = 0.0

        self.angle_eq = np.pi #equilibrium angles

        self.angle_stiffness = 4.0
        self.angle_damping = 0.42
        self.ground_stiffness = 1e3
        self.ground_damping = 5
        self.edge_stiffness = 1e3
        self.edge_damping = 5

        self.action_spaces = {a: Box(
            -self.max_motor_torque, self.max_motor_torque,
            shape=(1,),
            dtype=np.float32
        ) for a in self.possible_agents}
        self._get_obs = MethodType(observarion.get_obs_func, self)
        self._reward_func = MethodType(reward_func.func, self)

        self.observation_spaces = {a: observarion.space for a in self.possible_agents}

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.pos, self.vel = get_initial(self.n_particles, amplitude=self.particle_randomness, seed=options.get("init_seed", None))
        self.thdot = np.zeros((self.n_particles), dtype=np.float32)
        self.last_pos = self.pos.copy()
        self.mean_speed = 0.0
        self.camera_pos = np.mean(np.real(self.pos))
        self.camera_vel = 0.
        self.last_reward = 0.
        self.reward_history = np.zeros((100,))

        observations = self._get_obs()
        info = self._get_info()

        return observations, info

    def step(self, actions):
        self.timestep += 1
        joint_action = np.zeros((self.n_particles,), dtype=np.float32)
        joint_action[1:-1] = np.ravel(list(actions.values()))

        self.pos, self.vel, self.thdot = velocity_verlet_with_thdot(self.physics_steps_per_timestep, self.dt, self.particle_mass, joint_action, self.pos,  self.vel, self.angle_stiffness,self.angle_damping,self.edge_stiffness,self.edge_damping,self.background_friction,self.ground_stiffness,self.ground_damping,self.particle_radius,self.gravity_constant)
        self.mean_speed = np.mean(np.real(self.pos - self.last_pos))
        reward = self._reward_func()
        self.last_pos = self.pos.copy()
        self.last_reward = reward

        observations = self._get_obs()
        info = self._get_info()
        truncation = self.timestep >= self.max_steps

        return observations, reward, False, truncation, info

    def _get_info(self):
        return {'pos': self.pos, 'speed': self.mean_speed}

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        node_pos = self.pos - self.camera_pos

        scale = 20
        offset = self.window_size/2 + 1j * self.window_size/2
        node_pos = np.conjugate(node_pos * scale) + offset
        origin_pos = np.conjugate(-self.camera_pos * scale) + offset

        posx = np.real(node_pos)
        posy = np.imag(node_pos)

        # origin
        pygame.draw.circle(canvas, (200, 200, 200), np.array([np.real(origin_pos), np.imag(origin_pos)]), 5)
        font = pygame.font.Font(pygame.font.get_default_font(), 16)
        text_surface = font.render('(0, 0)', True, (200, 200, 200))
        canvas.blit(text_surface, dest=np.array([np.real(origin_pos), np.imag(origin_pos)])+4)

        # grid lines
        grid_line_spacing = 5
        linex = np.fmod(np.real(origin_pos), grid_line_spacing * scale)
        while linex < self.window_size:
            pygame.draw.line(canvas, (200, 200, 200), np.array([linex, 0]), np.array([linex, self.window_size]), width=1)
            linex += grid_line_spacing * scale
        liney = np.fmod(np.imag(origin_pos), grid_line_spacing * scale)
        while liney < self.window_size:
            pygame.draw.line(canvas, (200, 200, 200), np.array([0, liney]), np.array([self.window_size, liney]), width=1)
            liney += grid_line_spacing * scale

        # ground
        pygame.draw.line(canvas, 0, np.array([0, np.imag(origin_pos)]), np.array([self.window_size, np.imag(origin_pos)]), width=3)

        # nodes
        for i in np.arange(self.n_particles):
            centre = np.array([posx[i], posy[i]])
            pygame.draw.circle(canvas, (0, 0, 255), centre, self.particle_radius * scale)
        
        # edges
        c0 = np.array([posx[0], posy[0]])
        for i in np.arange(1, self.n_particles):
            c1 = np.array([posx[i], posy[i]])
            pygame.draw.line(canvas, 0, c0, c1, width=3)
            c0 = c1
        
        # reward
        text_surface = font.render(f'reward: {self.last_reward}', True, (255, 100, 100))
        canvas.blit(text_surface, dest=np.array([0,0])+4)
        self.reward_history = np.roll(self.reward_history, -1)
        self.reward_history[-1] = self.last_reward
        plot_on_pygame(canvas, self.reward_history, (0, 20, 200, 70))
        
        # move camera
        self.camera_pos += self.camera_vel
        self.camera_vel *= 0.99
        div = np.mean(np.real(self.pos)) - self.camera_pos

        dis_thresh = 1
        if np.abs(div) > dis_thresh:
            self.camera_vel += 0.0005 * div
        elif np.abs(self.camera_vel) < 0.01:
            self.camera_vel = 0

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def single_observation_space(self, agent):
        return self.observation_space(agent)

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def single_action_space(self, agent):
        return self.action_space(agent)


class parallel_env(simple_env):
    def step(self, actions):
        next_state, reward, termination, truncation, info = super().step(actions)
        return next_state, {a: reward for a in self.agents}, {a: termination for a in self.agents}, {a: truncation for a in self.agents}, info

    # def _get_info(self):
    #     info = super()._get_info()
    #     return {a: info for a in self.agents}



def plot_on_pygame(surface: pygame.Surface, data: np.array, box: tuple=(0,0,50,50,)):
    box = np.array(box)
    pygame.draw.rect(surface, (220, 220, 220), (box[0:2], box[2:4]-box[0:2])) 
    data_bounds = (np.min((np.min(data), 0)), np.max((np.max(data), 0)))
    anchor_y = np.interp(
        np.array([0] + list(range(max(math.ceil(data_bounds[0]), -10), 0)) + list(range(1, min(math.ceil(data_bounds[1]), 11))) ),
        data_bounds, (box[3], box[1]))

    pygame.draw.line(surface, (100, 100, 255), np.array([box[0], anchor_y[0]]), np.array([box[2], anchor_y[0]]), width=2)
    for y in anchor_y[1:]:
        pygame.draw.line(surface, (200, 200, 200), np.array([box[0], y]), np.array([box[2], y]), width=2)

    plot_points_y = np.interp(data, data_bounds, (box[3], box[1]))
    plot_points_x = np.interp(np.arange(len(data)), (0,len(data)-1), (box[0], box[2]))
    plot_points = np.vstack((plot_points_x, plot_points_y)).T
    point1 = plot_points[0]
    for point2 in plot_points[1:]:
        pygame.draw.line(surface, (255, 100, 0), point1, point2, width=3)
        point1 = point2