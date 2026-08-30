from collections import defaultdict
from typing import Optional

import numpy as np
from numba import jit
import torch
import tqdm
from tensordict import TensorDict
from types import MethodType

from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

import pygame
pygame.init()
from matplotlib.colors import hsv_to_rgb

observation_funcs = {}
def observation_func(dim=1):
    def decorator(func):
        name = func.__name__
        assert name.startswith('get_obs_')
        identifier = name[8:]
        observation_funcs[identifier] = {'func': func, 'dim': dim}
        return func
    return decorator


def terrain_density(x, y):
    return -y + np.sin(x)


class env(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = True

    def __init__(self, num_envs=1, num_particles = 4, max_steps=1000, device="cpu", render_mode=None, observation_func="dth_tot_plus_own"):
        super().__init__(device=device, batch_size=[num_envs])
        num_agents = num_particles
        max_torque = 9

        if observation_func not in observation_funcs:
            raise ValueError(f'Unknown observation func "{observation_func}". Available options: ' + ", ".join([f'"{k}"' for k in observation_funcs.keys()]))
        self._get_obs = MethodType(observation_funcs[observation_func]['func'], self)
        observation_size = observation_funcs[observation_func]['dim']
        self._reward_func = MethodType(reward_func_delta_pos, self)

        self.observation_spec = Composite(
            agents = Composite(
                observation = Bounded (
                    low = -torch.pi,
                    high = torch.pi,
                    shape = torch.Size([num_envs, num_agents, observation_size]),
                    dtype = torch.float32
                ),
                shape = torch.Size([num_envs, num_agents])
            ),
            log_info = Composite(
                trajectory = Unbounded(
                    shape = torch.Size([num_envs, num_particles, 2]),
                    dtype = torch.float32
                ),
                speed = Unbounded(
                    shape = torch.Size([num_envs, 1]),
                    dtype = torch.float32
                ),
                shape = torch.Size([num_envs])
            ),
            shape = torch.Size([num_envs])
        )

        self.action_spec = Composite(
            agents = Composite(
                action = Bounded (
                    low = -max_torque,
                    high = max_torque,
                    shape = torch.Size([num_envs, num_agents, 1]),
                    dtype = torch.float32
                ),
                shape = torch.Size([num_envs, num_agents])
            ),
            shape = torch.Size([num_envs])
        )

        self.num_envs = num_envs
        self.num_particles = num_particles
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.render_window = None
        # simulation parameters
        self.particle_randomness = 0.1
        self.particle_radius = 1/3.0
        self.particle_mass = 0.2

        self.dt = 3e-3 #integration time step
        self.physics_steps_per_timestep = 10 # amount of integration timesteps per observation-action

        self.max_motor_torque = 9.0

        self.gravity_constant = -1j*1
        self.background_friction = 0.0

        self.angle_eq = (np.pi * (num_particles - 2)) / num_particles #equilibrium angles

        self.angle_stiffness = 4.0
        self.angle_damping = 0.42
        self.ground_stiffness = 1e3
        self.ground_damping = 5
        self.edge_stiffness = 1e3
        self.edge_damping = 5
        self.edge_length = 1

    def _get_info(self):
        return TensorDict(
                    {
                        "trajectory": torch.stack((torch.real(self.pos), torch.imag(self.pos)), -1),
                        "speed": self.mean_speed
                    },
                    batch_size = self.observation_spec['log_info'].shape,
                    device = self.device
                )

    # Mandatory methods: _step, _reset and _set_seed
    def _reset(self, tensordict):
        self.steps = 0

        angles = torch.arange(self.num_particles, device=self.device)*2*torch.pi/self.num_particles
        radius = (self.edge_length / 2) / np.cos(((self.num_particles - 2)*np.pi)/(self.num_particles*2))
        pos_randomness = 0.01
    
        self.pos = (radius * (torch.cos(angles) + 1j * torch.sin(angles))).repeat(self.num_envs,1).type(torch.cfloat) + (radius + self.particle_radius) * 1j
        self.pos += (pos_randomness * torch.randn_like(self.pos) + pos_randomness*1j * torch.randn_like(self.pos) + 1j)
        self.mean_speed = torch.zeros((self.num_envs, 1,), device=self.device)
        self.last_pos = self.pos.clone()
        self.vel = torch.zeros((self.num_envs, self.num_particles,), device=self.device, dtype=torch.cfloat)
        self.thdot = torch.zeros((self.num_envs, self.num_particles,), device=self.device)
        self.camera_pos = 0j
        self.camera_vel = 0.
        self.camera_scale = 20.
        self.camera_scale_vel = 0.

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": self._get_obs()
                    },
                    batch_size = self.observation_spec['agents'].shape,
                    device = self.device
                ),
                "log_info": self._get_info()
            },
            batch_size = self.observation_spec.shape,
            device = self.device
        )

    def _step(self, tensordict):
        action = tensordict["agents", "action"]
        if action.requires_grad:
            raise ValueError("THE ACTION REQUIRES GRAD SO ITS PROBABLY A GOOD IDEA TO DO SOMETHING ABOUT THAT BEFORE PUTTING IT THROUGH PHYSICS SIM")

        joint_action = torch.zeros((self.num_envs, self.num_particles), device=self.device)
        joint_action[:,:] = action.squeeze(-1)
        np_pos, np_vel, np_thdot = velocity_verlet_with_thdot(self.physics_steps_per_timestep, self.dt, self.particle_mass, joint_action.cpu().numpy(), self.pos.cpu().numpy(),  self.vel.cpu().numpy(), self.angle_stiffness,self.angle_damping,self.edge_stiffness,self.edge_damping,self.background_friction,self.ground_stiffness,self.ground_damping,self.particle_radius,self.gravity_constant)
        self.pos, self.vel, self.thdot = torch.tensor(np_pos, device=self.device), torch.tensor(np_vel, device=self.device), torch.tensor(np_thdot, device=self.device)
        self.mean_speed = (torch.real(self.pos - self.last_pos)).mean(dim=1, keepdim=True)
        reward = self._reward_func()
        self.last_pos = self.pos.clone()

        self.steps += 1
        done = self.steps >= self.max_steps

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": self._get_obs()
                    },
                    batch_size = self.observation_spec['agents'].shape,
                    device = self.device
                ),
                "reward": self._reward_func(),
                "done": torch.full(self.done_spec.shape, fill_value=done, device = self.device),
                "log_info": self._get_info()
            },
            batch_size = self.observation_spec.shape,
            device = self.device
        )
    
    def render(self):
        if self.render_window is None:
            window_size = 512
            if self.render_mode == "human":
                pygame.display.init()
                self.render_window = {
                    "size": window_size,
                    "surface": pygame.display.set_mode((window_size, window_size)),
                    "clock": pygame.time.Clock(),
                    "font": pygame.font.Font(pygame.font.get_default_font(), 16)
                }
            else:
                self.render_window = {
                    "size": window_size,
                    "font": pygame.font.Font(pygame.font.get_default_font(), 16)
                }
        canvas = pygame.Surface((self.render_window['size'], self.render_window['size']))
        canvas.fill((255, 255, 255))

        node_pos = self.pos - self.camera_pos
        dispersion = torch.real(self.pos).max().item() - torch.real(self.pos).min().item()
        target_scale = 300 / dispersion
        scale = self.camera_scale
        offset = self.render_window['size']/2 + 1j * self.render_window['size']/2
        node_pos = torch.conj(node_pos * scale) + offset
        origin_pos = np.conjugate(-self.camera_pos * scale) + offset

        # ground
        ground_size = 0.5 * self.render_window['size'] / scale
        # print(np.real(self.camera_pos)-ground_size, np.real(self.camera_pos)+ground_size, '<>', np.imag(self.camera_pos)-ground_size, np.imag(self.camera_pos)+ground_size)
        ground_coords_x, ground_coords_y = np.meshgrid(
            np.linspace(np.real(self.camera_pos)-ground_size, np.real(self.camera_pos)+ground_size, self.render_window['size']),
            np.linspace(np.imag(self.camera_pos)+ground_size, np.imag(self.camera_pos)-ground_size, self.render_window['size'])
        )
        ground_image = np.repeat(
            ((terrain_density(ground_coords_x, ground_coords_y).T)[:, :, np.newaxis] < 0).astype(int) * 255,
            3, axis=2
        )
        pygame.surfarray.blit_array(canvas, ground_image)

        # origin
        pygame.draw.circle(canvas, (200, 200, 200), np.array([np.real(origin_pos), np.imag(origin_pos)]), 5)
        text_surface = self.render_window['font'].render('(0, 0)', True, (200, 200, 200))
        canvas.blit(text_surface, dest=np.array([np.real(origin_pos), np.imag(origin_pos)])+4)

        # grid lines
        grid_line_spacing = 5
        linex = np.fmod(np.real(origin_pos), grid_line_spacing * scale)
        while linex < self.render_window['size']:
            pygame.draw.line(canvas, (200, 200, 200), np.array([linex, 0]), np.array([linex, self.render_window['size']]), width=1)
            linex += grid_line_spacing * scale
        liney = np.fmod(np.imag(origin_pos), grid_line_spacing * scale)
        while liney < self.render_window['size']:
            pygame.draw.line(canvas, (200, 200, 200), np.array([0, liney]), np.array([self.render_window['size'], liney]), width=1)
            liney += grid_line_spacing * scale

        # ground
        pygame.draw.line(canvas, 0, np.array([0, np.imag(origin_pos)]), np.array([self.render_window['size'], np.imag(origin_pos)]), width=3)
        

        # step counter
        text_surface = self.render_window['font'].render(f'step {self.steps}/{self.max_steps}', True, (0, 0, 0))
        canvas.blit(text_surface, dest=np.array([0, 0])+4)

        for env in range(self.num_envs):
            hue = (env + 0.5) / self.num_envs
            circle_color = tuple(hsv_to_rgb((hue, 1, 0.75))*255)
            edge_color = tuple(hsv_to_rgb((hue, 1, 1))*255)

            posx = torch.real(node_pos[env])
            posy = torch.imag(node_pos[env])

            # nodes
            for i in np.arange(self.num_particles):
                centre = np.array([posx[i].item(), posy[i].item()])
                pygame.draw.circle(canvas, circle_color, centre, self.particle_radius * scale)
            
            # edges
            c0 = np.array([posx[-1].item(), posy[-1].item()])
            for i in np.arange(self.num_particles):
                c1 = np.array([posx[i].item(), posy[i].item()])
                pygame.draw.line(canvas, edge_color, c0, c1, width=3)
                c0 = c1

        # move camera
        self.camera_pos += self.camera_vel
        self.camera_vel *= 0.99
        target_camera_pos = torch.real(self.pos).min().item() + 0.5 * dispersion
        div = target_camera_pos - self.camera_pos
        dis_thresh = 1
        if np.abs(div) > dis_thresh:
            self.camera_vel += 0.001 * div
        elif np.abs(self.camera_vel) < 0.01:
            self.camera_vel = 0
        
        self.camera_scale += self.camera_scale_vel
        self.camera_scale_vel *= 0.8
        div = np.log(target_scale / scale)
        dis_thresh = 0.1
        if np.abs(div) > dis_thresh:
            self.camera_scale_vel += 0.03 * div
        elif np.abs(self.camera_scale_vel) < 0.01:
            self.camera_scale_vel = 0


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.render_window['surface'].blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.render_window['clock'].tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    


    def _set_seed(self, seed):
        pass


## helper func
#@torch.jit.script
def internode_angle(pos: torch.Tensor):
    dp = torch.roll(pos,(0, -1), (0, 1))-pos                                         #internode position vector
    dp_norm = dp/torch.absolute(dp)                                                  #internode unit vector
    return torch.angle(-dp_norm/torch.roll(dp_norm,(0, 1), (0, 1)))%(2*torch.pi)     #angle between nodes


## reward func
def reward_func_delta_pos(self):
    reward = self.mean_speed * 100
    return reward


## observation func
@observation_func(dim=1)
def get_obs_dth_tot(self):
    dth = internode_angle(self.pos) - self.angle_eq
    dthP=torch.roll(dth,(0, -1), (0, 1))
    dthM=torch.roll(dth,(0, 1), (0, 1))
    dth_tot = dthP-dthM
    return dth_tot.unsqueeze(-1)

@observation_func(dim=2)
def get_obs_dth_tot_plus_own(self):
    dth = internode_angle(self.pos) - self.angle_eq
    dthP=torch.roll(dth,(0, -1), (0, 1))
    dthM=torch.roll(dth,(0, 1), (0, 1))
    dth_tot = dthP-dthM
    return torch.stack((dth_tot, dth), dim=2)

@observation_func(dim=2)
def get_obs_dth_neighbours(self):
    dth = internode_angle(self.pos) - self.angle_eq
    dthP=torch.roll(dth,(0, -1), (0, 1))
    dthM=torch.roll(dth,(0, 1), (0, 1))
    return torch.stack((dthP, dthM), dim=2)

@observation_func(dim=3)
def get_obs_dth_neighbours_plus_own(self):
    dth = internode_angle(self.pos) - self.angle_eq
    dthP=torch.roll(dth,(0, -1), (0, 1))
    dthM=torch.roll(dth,(0, 1), (0, 1))
    return torch.stack((dthP, dth, dthM), dim=2)


## physics sim
#@jit(nopython=True)
def roll_pos1(arr): # numba doesn't support multi-dim rolling :(
    rolled = np.empty_like(arr)
    rolled[:, 1:] = arr[:, :-1]
    rolled[:, :1] = arr[:, -1:]
    return rolled
#@jit(nopython=True)
def roll_neg1(arr):
    rolled = np.empty_like(arr)
    rolled[:, :-1] = arr[:, 1:]
    rolled[:, -1:] = arr[:, :1]
    return rolled

#@jit(nopython=True)
def passive_hinge_force(ke: float, disp: float, o: np.ndarray, oC: np.ndarray, oM: np.ndarray,
                        dth: np.ndarray, dthM: np.ndarray, dthP: np.ndarray,
                        thdot: np.ndarray, thdotP: np.ndarray, thdotM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fE = ke * (dth * oC - dthP * o - dthM * oM)
    fD = disp * (thdot * oC - thdotP * o - thdotM * oM)

    # fE[:, 0] = ke * (-dthP[:, 0] * o[:, 0])
    # fE[:, -1] = ke * (-dthM[:, -1] * oM[:, -1])
    # fE[:, 1] = ke * (dth[:, 1] * oC[:, 1] - dthP[:, 1] * o[:, 1])
    # fE[:, -2] = ke * (dth[:, -2] * oC[:, -2] - dthM[:, -2] * oM[:, -2])

    # fD[:, 0] = disp * (-thdotP[:, 0] * o[:, 0])
    # fD[:, -1] = disp * (-thdotM[:, -1] * oM[:, -1])
    # fD[:, 1] = disp * (thdot[:, 1] * oC[:, 1] - thdotP[:, 1] * o[:, 1])
    # fD[:, -2] = disp * (thdot[:, -2] * oC[:, -2] - thdotM[:, -2] * oM[:, -2])

    return fE, fD

#@jit(nopython=True)
def convert_torque_to_force(tau: np.ndarray, o: np.ndarray, oM: np.ndarray) -> np.ndarray:
    fM = 0.5 * tau * oM
    fP = 0.5 * tau * o
    fself = -(fM + fP)
    fM = roll_neg1(fM)
    fP = roll_pos1(fP)
    fO = fself + fM + fP
    return fO

#@jit(nopython=True)
def edge_force(pos: np.ndarray, vel: np.ndarray, kr: float, gr: float, r: np.ndarray, dv: np.ndarray) -> np.ndarray:
    leq = 1
    fs = np.zeros_like(pos)
    mag = np.abs(r)

    # mag[:, -1] = 1
    # dv[:, -1] = 0.

    rh = r / mag
    SMag = kr * (mag - leq)
    DMag = gr * np.real(dv * np.conj(rh))
    T = rh * (SMag + DMag)
    fs = T - roll_pos1(T)

    return fs

#@jit(nopython=True)
def slide_force(vel: np.ndarray, gam: float) -> np.ndarray:
    ff = -gam * vel
    return ff

# @jit(nopython=True)
# def wall_force(pos: np.ndarray, vel: np.ndarray, ks: float, gs: float, rad: float) -> np.ndarray:
#     fw = np.zeros_like(pos)
#     msk = np.imag(pos) < rad
#     fw[msk] = -gs * vel[msk] - 1j * ks * (np.imag(pos[msk]) - rad)
#     return fw

#@jit(nopython=True)
def wall_force(pos: np.ndarray, vel: np.ndarray, ks: float, gs: float, rad: float):
    #ks = ground_stiffness, gs = ground_damping, rad = radius of nodes
    fw = np.zeros_like(pos)
    num_envs, num_particles = pos.shape

    for i in range(num_envs): # numba doesn't support multi-dim indexing mask :(
        msk = terrain_density(np.real(pos[i]), np.imag(pos[i]) - rad) > 0
        fw[i][msk] = -gs * vel[i][msk] + 1j * ks * terrain_density(np.real(pos[i]), np.imag(pos[i]) - rad)[msk]

    return fw


#@jit(nopython=True)
def grav_force(pos: np.ndarray, g: complex) -> np.ndarray:
    fg = g * np.ones_like(pos)
    return fg

#@jit(nopython=True)
def calculate(pos: np.ndarray, vel: np.ndarray, teq: float):
    r = roll_neg1(pos) - pos
    dv = roll_neg1(vel) - vel
    mag = np.abs(r)
    rh = r / mag
    th = np.angle(-rh / roll_pos1(rh)) % (2 * np.pi)
    dth = th - teq
    
    dthP = roll_neg1(dth)
    dthM = roll_pos1(dth)
    
    o = 1j * rh / mag
    oM = roll_pos1(o)
    oC = o + oM
    
    velP = roll_neg1(vel)
    velM = roll_pos1(vel)
    
    thdot = np.real(-oC * np.conj(vel) + o * np.conj(velP) + oM * np.conj(velM))
    thdotP = roll_neg1(thdot)
    thdotM = roll_pos1(thdot)
    
    return r, dv, mag, rh, th, dth, dthP, dthM, thdot, thdotP, thdotM, o, oC, oM


#@jit(nopython=True)
def get_forces(action: np.ndarray, pos: np.ndarray, vel: np.ndarray, angle_stiffness: float, angle_damping: float,
               edge_stiffness: float, edge_damping: float, background_friction: float, ground_stiffness: float,
               ground_damping: float, particle_radius: float, gravity_constant: complex) -> tuple[np.ndarray, np.ndarray]:

    r, dv, mag, rh, th, dth, dthP, dthM, thdot, thdotP, thdotM, o, oC, oM = calculate(pos, vel, teq=np.pi)

    fE, fD = passive_hinge_force(angle_stiffness, angle_damping, o, oC, oM, dth, dthM, dthP, thdot, thdotP, thdotM)

    tauO = action
    fO = convert_torque_to_force(tauO, o, oM)

    fs = edge_force(pos, vel, edge_stiffness, edge_damping, r, dv)
    ff = slide_force(pos, background_friction)
    fw = wall_force(pos, vel, ground_stiffness, ground_damping, particle_radius)
    fg = grav_force(pos, gravity_constant)

    ftot = fO + fs + ff + fw + fg + fE + fD

    return ftot, thdot

#@jit(nopython=True)
def velocity_verlet_with_thdot(steps: int, dt: float, particle_mass: float, action: np.ndarray, pos: np.ndarray,
                               vel: np.ndarray, angle_stiffness: float, angle_damping: float, edge_stiffness: float,
                               edge_damping: float, background_friction: float, ground_stiffness: float,
                               ground_damping: float, particle_radius: float, gravity_constant: complex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    for _ in range(steps):
        f, _ = get_forces(action, pos, vel, angle_stiffness, angle_damping, edge_stiffness, edge_damping,
                          background_friction, ground_stiffness, ground_damping, particle_radius, gravity_constant)

        vel += f * 0.5 * dt / particle_mass
        pos += vel * dt

        f, thdot = get_forces(action, pos, vel, angle_stiffness, angle_damping, edge_stiffness, edge_damping,
                              background_friction, ground_stiffness, ground_damping, particle_radius, gravity_constant)

        vel += f * 0.5 * dt / particle_mass

    return pos, vel, thdot
