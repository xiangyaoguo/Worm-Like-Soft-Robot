from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from types import MethodType

from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

import pygame
pygame.init()
from matplotlib.colors import hsv_to_rgb


class env(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 300,
    }
    batch_locked = True

    def __init__(self, num_envs=1, num_particles = 4, max_steps=1000, device="cpu", render_mode=None):
        super().__init__(device=device, batch_size=[num_envs])
        num_agents = num_particles - 2
        observation_size = 2
        max_torque = 9

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

        self.gravity_constant = torch.tensor([0,-1], device=self.device)
        self.background_friction = 0.0

        self.angle_eq = np.pi #equilibrium angles

        self.angle_stiffness = 4.0
        self.angle_damping = 0.42
        self.ground_stiffness = 1e3
        self.ground_damping = 5
        self.edge_stiffness = 1e3
        self.edge_damping = 5
        self.edge_length = 1

        self._get_obs = MethodType(get_obs_dth_tot_plus_own, self)
        self._reward_func = MethodType(reward_func_delta_pos, self)

    def _get_info(self):
        return TensorDict(
                    {
                        "trajectory": self.pos,
                        "speed": self.mean_speed
                    },
                    batch_size = self.observation_spec['log_info'].shape,
                    device = self.device
                )

    # Mandatory methods: _step, _reset and _set_seed
    def _reset(self, tensordict):
        self.steps = 0
    
        self.pos = torch.stack((
            ((torch.arange(self.num_particles, device=self.device) - self.num_particles/2) * self.edge_length).repeat(self.num_envs,1),
            torch.randn((self.num_envs, self.num_particles,), device=self.device) * 0.2 + 1 + 0.5 * self.particle_radius
        ), -1)
        
        self.mean_speed = torch.zeros((self.num_envs, 1,), device=self.device)
        self.last_pos = self.pos.clone()
        self.vel = torch.zeros_like(self.pos)
        self.thdot = torch.zeros((self.num_envs, self.num_particles,), device=self.device)
        self.camera_pos = torch.tensor([0,0],device=self.device)
        self.camera_vel = 0.0

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
        joint_action[:,1:-1] = action.squeeze(-1)
        self.pos, self.vel, self.thdot = velocity_verlet_with_thdot_vec(self.physics_steps_per_timestep, self.dt, self.particle_mass, joint_action, self.pos,  self.vel, self.angle_stiffness,self.angle_damping,self.edge_stiffness,self.edge_damping,self.background_friction,self.ground_stiffness,self.ground_damping,self.particle_radius,self.gravity_constant)
        self.mean_speed = (self.pos[..., 0] - self.last_pos[..., 0]).mean(dim=1, keepdim=True)
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

        self.camera_pos = np.array([float(torch.mean(self.pos[... ,0]).item()), 0])
        node_pos = (self.pos - self.camera_pos).numpy().astype(float)
        node_pos[..., 1] *= -1
        scale = 20
        offset = np.array([self.render_window['size']/2, self.render_window['size']/2])
        node_pos = node_pos * scale + offset
        origin_pos = np.array([-self.camera_pos[0], self.camera_pos[1]])* scale + offset

        # origin
        pygame.draw.circle(canvas, (200, 200, 200), origin_pos, 5)
        text_surface = self.render_window['font'].render('(0, 0)', True, (200, 200, 200))
        canvas.blit(text_surface, dest=origin_pos+4)

        # grid lines
        grid_line_spacing = 5
        linex = np.fmod(origin_pos[0], grid_line_spacing * scale)
        while linex < self.render_window['size']:
            pygame.draw.line(canvas, (200, 200, 200), np.array([linex, 0]), np.array([linex, self.render_window['size']]), width=1)
            linex += grid_line_spacing * scale
        liney = np.fmod(origin_pos[1], grid_line_spacing * scale)
        while liney < self.render_window['size']:
            pygame.draw.line(canvas, (200, 200, 200), np.array([0, liney]), np.array([self.render_window['size'], liney]), width=1)
            liney += grid_line_spacing * scale

        # ground
        pygame.draw.line(canvas, 0, np.array([0, origin_pos[1]]), np.array([self.render_window['size'], origin_pos[1]]), width=3)

        # step counter
        text_surface = self.render_window['font'].render(f'step {self.steps}/{self.max_steps}', True, (0, 0, 0))
        canvas.blit(text_surface, dest=np.array([0, 0])+4)

        for env in range(self.num_envs):
            hue = (env + 0.5) / self.num_envs
            circle_color = tuple(hsv_to_rgb((hue, 1, 0.75))*255)
            edge_color = tuple(hsv_to_rgb((hue, 1, 1))*255)

            pos_env = node_pos[env]

            # nodes
            for i in np.arange(self.num_particles):
                centre = pos_env[i]
                pygame.draw.circle(canvas, circle_color, centre, self.particle_radius * scale)
            
            # edges
            c0 = pos_env[0]
            for i in np.arange(1, self.num_particles):
                c1 = pos_env[i]
                pygame.draw.line(canvas, edge_color, c0, c1, width=3)
                c0 = c1

        # move camera
        # self.camera_pos[0] += self.camera_vel
        # self.camera_vel *= 0.99
        # div = torch.mean(self.pos[:,:,0]).item() - self.camera_pos[0]

        # dis_thresh = 1
        # if np.abs(div) > dis_thresh:
        #     self.camera_vel += 0.0005 * div
        # elif np.abs(self.camera_vel) < 0.01:
        #     self.camera_vel = 0


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


@torch.jit.script
def internode_angle_vec(pos: torch.Tensor):
    # Compute internode vectors
    dp = torch.roll(pos, shifts=-1, dims=1) - pos  # shape: (batch, N, 2)

    # Normalize
    dp_norm = dp / torch.norm(dp, dim=2, keepdim=True)

    # Previous (left) normalized vector
    dp_prev = torch.roll(dp_norm, shifts=1, dims=1)

    # Angle between using cross and dot product
    dot = (dp_prev * (-dp_norm)).sum(dim=2)
    cross = dp_prev[..., 0] * (-dp_norm[..., 1]) - dp_prev[..., 1] * (-dp_norm[..., 0])

    angle = torch.atan2(cross, dot) % (2 * torch.pi)
    return angle


## reward func
def reward_func_delta_pos(self):
    reward = self.mean_speed * 100
    return reward


def get_obs_dth_tot(self):
    dth = internode_angle_vec(self.pos) - self.angle_eq
    dth[:, 0] = 0.0
    dth[:, -1] = 0.0
    dthP = torch.roll(dth, shifts=-1, dims=1)
    dthM = torch.roll(dth, shifts=1, dims=1)
    dth_tot = dthP - dthM
    return dth_tot[:, 1:-1]

def get_obs_dth_tot_plus_own(self):
    dth = internode_angle_vec(self.pos) - self.angle_eq
    dth[:, 0] = 0.0
    dth[:, -1] = 0.0
    dthP = torch.roll(dth, shifts=-1, dims=1)
    dthM = torch.roll(dth, shifts=1, dims=1)
    dth_tot = dthP - dthM
    return torch.stack((dth_tot[:, 1:-1], dth[:, 1:-1]), dim=2)  # shape: (batch, N-2, 2)

@torch.jit.script
def vec_angle(v: torch.Tensor):
    return torch.atan2(v[..., 1], v[..., 0])

## physics sim
@torch.jit.script
def rotate90(v: torch.Tensor):
    return torch.stack([-v[..., 1], v[..., 0]], dim=-1)  # 90-degree counterclockwise rotation

@torch.jit.script
def dot2d(a: torch.Tensor, b: torch.Tensor):
    # (a + bi) and (c + di) -> (ac - bd) + (ad + bc)i
    # only care about real part (ac - bd)
    # take conjugate of b
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]

@torch.jit.script
def complex_mult(a: torch.Tensor, b: torch.Tensor):
    # (a + bi) and (c + di) -> (ac - bd) + (ad + bc)i
    return torch.stack((
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ), -1)

@torch.jit.script
def complex_div(a: torch.Tensor, b: torch.Tensor):
    # (a + bi) and (c + di) -> (ac + bd) / (c^2 + d^2) + i * (bc - ad) / (c^2 + d^2)
    denom = a[..., 0] ** 2 + b[..., 1] ** 2
    return torch.stack((
        a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1],
        a[..., 1] * b[..., 0] - a[..., 0] * b[..., 1],
    ), -1) / denom.unsqueeze(-1)

@torch.jit.script
def passive_hinge_force_vec(
    ke: float, disp: float,
    o: torch.Tensor, oC: torch.Tensor, oM: torch.Tensor,
    dth: torch.Tensor, dthM: torch.Tensor, dthP: torch.Tensor,
    thdot: torch.Tensor, thdotP: torch.Tensor, thdotM: torch.Tensor,
):
    fE = ke * (dth.unsqueeze(-1) * oC - dthP.unsqueeze(-1) * o - dthM.unsqueeze(-1) * oM)
    fD = disp * (thdot.unsqueeze(-1) * oC - thdotP.unsqueeze(-1) * o - thdotM.unsqueeze(-1) * oM)

    # boundaries
    fE[:, 0] = ke * (-dthP[:, 0].unsqueeze(-1) * o[:, 0])
    fE[:, -1] = ke * (-dthM[:, -1].unsqueeze(-1) * oM[:, -1])
    fE[:, 1] = ke * (dth[:, 1].unsqueeze(-1) * oC[:, 1] - dthP[:, 1].unsqueeze(-1) * o[:, 1])
    fE[:, -2] = ke * (dth[:, -2].unsqueeze(-1) * oC[:, -2] - dthM[:, -2].unsqueeze(-1) * oM[:, -2])
    
    fD[:, 0] = disp * (-thdotP[:, 0].unsqueeze(-1) * o[:, 0])
    fD[:, -1] = disp * (-thdotM[:, -1].unsqueeze(-1) * oM[:, -1])
    fD[:, 1] = disp * (thdot[:, 1].unsqueeze(-1) * oC[:, 1] - thdotP[:, 1].unsqueeze(-1) * o[:, 1])
    fD[:, -2] = disp * (thdot[:, -2].unsqueeze(-1) * oC[:, -2] - thdotM[:, -2].unsqueeze(-1) * oM[:, -2])

    return fE, fD

@torch.jit.script
def convert_torque_to_force_vec(tau: torch.Tensor, o: torch.Tensor, oM: torch.Tensor):
    fM = 0.5 * tau.unsqueeze(-1) * oM
    fP = 0.5 * tau.unsqueeze(-1) * o
    fself = -(fM + fP)
    fM = torch.roll(fM, shifts=(0, -1), dims=(0, 1))
    fP = torch.roll(fP, shifts=(0, 1), dims=(0, 1))
    fO = fself + fM + fP
    return fO

@torch.jit.script
def edge_force_vec(pos: torch.Tensor, vel: torch.Tensor, kr: float, gr: float, r: torch.Tensor, dv: torch.Tensor, mag: torch.Tensor):
    leq = 1.0
    fs = torch.zeros_like(pos)

    mag[:, -1] = 1.0
    dv[:, -1] = 0.0

    rh = r / mag.unsqueeze(-1)
    SMag = kr * (mag - leq)
    DMag = gr * dot2d(dv, rh)
    T = rh * (SMag + DMag).unsqueeze(-1)
    fs = T - torch.roll(T, shifts=(0, 1), dims=(0, 1))
    return fs

@torch.jit.script
def wall_force_vec(pos: torch.Tensor, vel: torch.Tensor, ks: float, gs: float, rad: float):
    fw = torch.zeros_like(pos)
    mask = pos[..., 1] < rad
    fw[mask] = -gs * vel[mask]
    fw[..., 1][mask] -= ks * (pos[..., 1][mask] - rad)
    return fw

@torch.jit.script
def grav_force_vec(pos: torch.Tensor, g: torch.Tensor):
    return g.unsqueeze(0).unsqueeze(1) * torch.ones_like(pos)

@torch.jit.script
def calculate_vec(pos: torch.Tensor, vel: torch.Tensor, teq: float):
    r = torch.roll(pos, shifts=-1, dims=1) - pos
    dv = torch.roll(vel, shifts=-1, dims=1) - vel

    mag = torch.norm(r, dim=-1)
    rh = r / mag.unsqueeze(-1)

    th = vec_angle(complex_div(-rh, torch.roll(rh, shifts=1, dims=1))) % (2*torch.pi)
    dth = th - teq

    dthP = torch.roll(dth, (0, -1), (0, 1))
    dthM = torch.roll(dth, (0, 1), (0, 1))

    o = rotate90(rh) / mag.unsqueeze(-1)
    oM = torch.roll(o, (0, 1), (0, 1))
    oC = o + oM

    velP = torch.roll(vel, (0, -1), (0, 1))
    velM = torch.roll(vel, (0, 1), (0, 1))

    thdot = dot2d(-oC, vel) + dot2d(o, velP) + dot2d(oM, velM)
    thdotP = torch.roll(thdot, (0, -1), (0, 1))
    thdotM = torch.roll(thdot, (0, 1), (0, 1))

    return r, dv, mag, rh, th, dth, dthP, dthM, thdot, thdotP, thdotM, o, oC, oM

@torch.jit.script
def get_forces_vec(
    action: torch.Tensor,
    pos: torch.Tensor,
    vel: torch.Tensor,
    angle_stiffness: float,
    angle_damping: float,
    edge_stiffness: float,
    edge_damping: float,
    background_friction: float,
    ground_stiffness: float,
    ground_damping: float,
    particle_radius: float,
    gravity_vector: torch.Tensor  # shape: (2,)
):
    r, dv, mag, rh, th, dth, dthP, dthM, thdot, thdotP, thdotM, o, oC, oM = calculate_vec(pos, vel, teq=torch.pi)

    fE, fD = passive_hinge_force_vec(angle_stiffness, angle_damping, o, oC, oM, dth, dthM, dthP, thdot, thdotP, thdotM)
    fO = convert_torque_to_force_vec(action, o, oM)
    fs = edge_force_vec(pos, vel, edge_stiffness, edge_damping, r, dv, mag)
    ff = -background_friction * vel
    fw = wall_force_vec(pos, vel, ground_stiffness, ground_damping, particle_radius)
    fg = grav_force_vec(pos, gravity_vector)

    ftot = fO + fs + ff + fw + fg + fE + fD
    return ftot, thdot

@torch.jit.script
def velocity_verlet_with_thdot_vec(
    steps: int,
    dt: float,
    particle_mass: float,
    action: torch.Tensor,
    pos: torch.Tensor,
    vel: torch.Tensor,
    angle_stiffness: float,
    angle_damping: float,
    edge_stiffness: float,
    edge_damping: float,
    background_friction: float,
    ground_stiffness: float,
    ground_damping: float,
    particle_radius: float,
    gravity_vector: torch.Tensor  # shape: (2,)
):
    for _ in range(steps - 1):
        f, _ = get_forces_vec(
            action, pos, vel,
            angle_stiffness, angle_damping,
            edge_stiffness, edge_damping,
            background_friction, ground_stiffness,
            ground_damping, particle_radius,
            gravity_vector
        )
        vel = vel + 0.5 * dt * f / particle_mass
        pos = pos + dt * vel

        f, thdot = get_forces_vec(
            action, pos, vel,
            angle_stiffness, angle_damping,
            edge_stiffness, edge_damping,
            background_friction, ground_stiffness,
            ground_damping, particle_radius,
            gravity_vector
        )
        vel = vel + 0.5 * dt * f / particle_mass

    # Final step
    f, _ = get_forces_vec(
        action, pos, vel,
        angle_stiffness, angle_damping,
        edge_stiffness, edge_damping,
        background_friction, ground_stiffness,
        ground_damping, particle_radius,
        gravity_vector
    )
    vel = vel + 0.5 * dt * f / particle_mass
    pos = pos + dt * vel

    f, thdot = get_forces_vec(
        action, pos, vel,
        angle_stiffness, angle_damping,
        edge_stiffness, edge_damping,
        background_friction, ground_stiffness,
        ground_damping, particle_radius,
        gravity_vector
    )
    vel = vel + 0.5 * dt * f / particle_mass

    return pos, vel, thdot
