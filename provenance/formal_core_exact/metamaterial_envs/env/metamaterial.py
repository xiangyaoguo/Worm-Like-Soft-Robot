from torchrl.envs import EnvBase
import numpy as np
import numba as nb
import torch
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from types import MethodType
from tensordict import TensorDict
import pygame
from matplotlib.colors import hsv_to_rgb
pygame.init()


class MeshTerrain:
    def __init__(self, settings):
        try:
            settings = settings.copy()
        except:
            pass
        if isinstance(settings, list):
            self.lines = np.array(settings, dtype=np.complex64)
        elif isinstance(settings, (str, dict)):
            # use a preset
            if isinstance(settings, str):
                preset_type = settings
                preset_options = {}
            else:
                preset_type = settings['type']
                preset_options = settings
                del preset_options['type']

            # Accept both the original MeshTerrain key `tunnel` and the
            # clearer CLI/config key `tunnel_length` for the tunnel preset.
            if preset_type == 'tunnel' and 'tunnel_length' in preset_options:
                preset_options['tunnel'] = preset_options.pop('tunnel_length')
            
            if preset_type == 'flat':
                self.lines = np.array([[-1000, 1000]], dtype=np.complex64)
            elif preset_type == 'stairs':
                self.lines = self._preset_stairs(**preset_options)
            elif preset_type == 'tunnel':
                self.lines = self._preset_tunnel(**preset_options)
            else:
                raise ValueError
        else:
            raise ValueError
    
    def _preset_stairs(self, start_stairs=10, step_width=5, step_height=1, steps=10):
        lines = [[-1000, start_stairs]]
        for i in range(steps):
            x = start_stairs + step_width * i
            y = step_height * i
            lines.append([x + y * 1j, x + (y+step_height) * 1j])
            lines.append([x + (y+step_height) * 1j, x + step_width + (y+step_height) * 1j])
        end_point = start_stairs + steps * step_width + (steps * step_height) * 1j
        lines.append([end_point, end_point + 1000])
        return np.array(lines, dtype=np.complex64)
    
    def _preset_tunnel(self, start=10, slope=5, slope_height=1, tunnel=10, tunnel_height=5.0):
        lines = [
            [-1000, start], # _
            [start, start + slope + slope_height * 1j], # /
            [start + (slope_height * 2 + tunnel_height) * 1j, start + slope + (slope_height + tunnel_height) * 1j], # \
            [start + (slope_height * 2 + tunnel_height) * 1j, start + (slope_height * 2 + tunnel_height + 100) * 1j], # |
            [start + slope + slope_height * 1j, start + slope + tunnel + slope_height * 1j], # -
            [start + slope + (slope_height + tunnel_height) * 1j, start + slope + tunnel + (slope_height + tunnel_height) * 1j], # -
            [start + slope + tunnel + slope_height * 1j, start + 2*slope + tunnel], # \
            [start + slope + tunnel + (slope_height + tunnel_height) * 1j, start + 2*slope + tunnel+ (slope_height * 2 + tunnel_height) * 1j], # /
            [start + 2*slope + tunnel+ (slope_height * 2 + tunnel_height) * 1j, start + 2*slope + tunnel+ (slope_height * 2 + tunnel_height + 100) * 1j], # |
            [start + 2*slope + tunnel, 1000] # _
        ]
        return np.array(lines, dtype=np.complex64)



reward_funcs = {}
def reward_func():
    def decorator(func):
        name = func.__name__
        assert name.startswith('reward_func_')
        identifier = name[12:]
        reward_funcs[identifier] = {'func': func}
        return func
    return decorator

@reward_func()
def reward_func_horizontal_speed(self):
    reward = np.real(self.mean_speed).astype(np.float32) * np.float32(100.0)
    return torch.tensor(reward, dtype=torch.float32, device=self.device)


@reward_func()
def reward_func_rolling_curriculum(self):
    """Two-stage dense reward: first form a loop, then roll it forward."""
    metrics = self._compute_rolling_metrics()
    curl_reward = (
        np.float32(0.60) * metrics["closure_score"]
        + np.float32(0.25) * metrics["circularity_score"]
        + np.float32(0.10) * metrics["speed_score"]
        - np.float32(0.05) * metrics["effort_penalty"]
    )
    rolling_reward = (
        np.float32(0.30) * metrics["speed_score"]
        + np.float32(0.30) * metrics["rotation_score"]
        + np.float32(0.20) * metrics["closure_score"]
        + np.float32(0.10) * metrics["circularity_score"]
        - np.float32(0.08) * metrics["slip_penalty"]
        - np.float32(0.02) * metrics["effort_penalty"]
    )
    progress = metrics["curriculum_progress"]
    blended_reward = (np.float32(1.0) - progress) * curl_reward + progress * rolling_reward
    blended_reward -= self.action_smoothness_weight * metrics["action_smoothness_penalty"]
    reward = self.rolling_reward_scale * blended_reward
    reward = np.asarray(reward, dtype=np.float32)
    metrics["rolling_reward"] = reward
    return torch.as_tensor(reward, dtype=torch.float32, device=self.device)


@reward_func()
def reward_func_tail_roll_curriculum(self):
    """Competence-gated tail-first rolling reward.

    This preset is intentionally opt-in.  Unlike ``rolling_curriculum``, it
    never pays an ungated forward-speed bonus.  Each stage uses potential
    progress plus a one-shot milestone bonus, so a static folded posture
    cannot collect the same reward forever.
    """
    metrics = self._compute_tail_roll_metrics()
    stage = int(self.tail_roll_stage)
    potential = metrics["tail_stage_potentials"][:, stage : stage + 1]
    if self._tail_previous_potential is None:
        progress_reward = np.zeros_like(potential, dtype=np.float32)
    else:
        progress_reward = (
            np.float32(self.tail_roll_potential_gamma) * potential
            - self._tail_previous_potential
        )
    self._tail_previous_potential = np.asarray(potential, dtype=np.float32).copy()

    stage_success = metrics["tail_stage_success"] > np.float32(0.5)
    new_success = stage_success & (~self._tail_stage_success_latched)
    self._tail_stage_success_latched |= stage_success
    metrics["tail_stage_success"] = self._tail_stage_success_latched.astype(np.float32)
    milestone_bonus = new_success.astype(np.float32)

    dynamic_reward = np.zeros_like(potential, dtype=np.float32)
    if stage >= 3:
        # Translation is valuable only while the body is loop-like and
        # rotating in the requested direction.  This removes the crawling
        # shortcut learned by the earlier rolling reward.
        dynamic_reward = (
            np.float32(0.55) * metrics["desired_rotation_increment"]
            + np.float32(0.25)
            * metrics["rolling_gate"]
            * metrics["forward_displacement_increment"]
            + np.float32(0.20) * metrics["contact_migration_increment"]
        )

    reward = (
        np.float32(5.0) * progress_reward
        + milestone_bonus
        + dynamic_reward
        - np.float32(0.02) * metrics["effort_penalty"]
        - self.action_smoothness_weight * metrics["action_smoothness_penalty"]
    )
    reward = np.asarray(self.tail_roll_reward_scale * reward, dtype=np.float32)
    metrics["tail_roll_reward"] = reward
    return torch.as_tensor(reward, dtype=torch.float32, device=self.device)



class env(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    batch_locked = True

    def __init__(self, num_envs=1, material_shape="ring", num_particles=4, max_steps=1000, terrain_type="flat", terrain_settings=None, 
                 render_mode=None, observation_func="dth_neighbours", reward_func="horizontal_speed", render=False, window_width=1000, window_height=500, render_text_lines=[],
                 control_mode="direct", feedback_gain=None, max_control_gain=9.0, action_mode=None, feedback_velocity_gain=None, coefficient_limit=None,
                 fixed_k1=-5.0, fix_k1=False, fixed_k2=0.0, fix_k2=False, k1_min=None, k1_max=None, k2_min=None, k2_max=None,
                 k_action_scale=1.0, min_k2_magnitude=1e-3, passive_kappa=4.0,
                 rolling_observation=False, rolling_direction="right", rolling_curl_episodes=500,
                 rolling_transition_episodes=300, rolling_speed_ref_x100=2.0, rolling_omega_ref=1.0,
                 rolling_reward_scale=3.0, init_pos_randomness=0.01, init_angle_range_degrees=0.0,
                 init_height_jitter=0.0, action_smoothness_weight=0.0,
                 tail_roll_observation=False, tail_side="left", tail_curl_sign="auto",
                 tail_roll_stage=0, tail_roll_reward_scale=3.0,
                 tail_roll_potential_gamma=1.0, tail_roll_contact_margin=0.05,
                 tail_roll_curl_reference=np.pi / 3.0,
                 tail_roll_init_assist_degrees=0.0, tail_roll_init_assist_segments=4,
                 tail_roll_init_assist_episodes=0,
                 tail_wave_amplitude_max=2.8, tail_wave_center_max=1.1,
                 tail_wave_width_min=0.03, tail_wave_width_max=0.35,
                 tail_wave_kp_max=12.0, tail_wave_kd_max=4.0):
        assert material_shape in ["ring", "crawler"]
        super().__init__(device="cpu", batch_size=[num_envs])
        num_agents = num_particles - 2 if material_shape == "crawler" else num_particles

        max_torque = np.float32(9.0) # mN/m
        # Compatibility aliases used by newer training/demo scripts and notebooks.
        # control_mode="formula" is the same as action_mode="formula".
        # feedback_gain is kept as a compatibility argument; F is fixed to 1.0.
        if action_mode is not None:
            control_mode = action_mode
        if feedback_velocity_gain is not None:
            feedback_gain = feedback_velocity_gain
        if coefficient_limit is not None:
            max_control_gain = coefficient_limit
        max_control_gain = np.float32(max_control_gain)
        if max_control_gain <= 0:
            raise ValueError("max_control_gain / coefficient_limit must be positive.")
        fixed_k1_value = np.float32(fixed_k1)
        fixed_k2_value = np.float32(fixed_k2)
        k_action_scale_value = np.float32(k_action_scale)
        min_k2_magnitude_value = np.float32(min_k2_magnitude)
        passive_kappa_value = np.float32(passive_kappa)
        if not np.isfinite(fixed_k1_value):
            raise ValueError("fixed_k1 must be finite.")
        if not np.isfinite(fixed_k2_value):
            raise ValueError("fixed_k2 must be finite.")
        if not np.isfinite(k_action_scale_value) or k_action_scale_value <= 0:
            raise ValueError("k_action_scale must be a positive finite value.")
        if not np.isfinite(passive_kappa_value) or passive_kappa_value <= 0:
            raise ValueError("passive_kappa must be a positive finite value.")
        if (
            not np.isfinite(min_k2_magnitude_value)
            or min_k2_magnitude_value <= 0
            or min_k2_magnitude_value >= max_control_gain
        ):
            raise ValueError("min_k2_magnitude must be > 0 and < max_control_gain.")

        def optional_bound(value, fallback, name):
            if value is None:
                return np.float32(fallback), False
            parsed = np.float32(value)
            if not np.isfinite(parsed):
                raise ValueError(f"{name} must be finite.")
            return parsed, True

        k1_min_value, k1_min_set = optional_bound(k1_min, -np.inf, "k1_min")
        k1_max_value, k1_max_set = optional_bound(k1_max, np.inf, "k1_max")
        k2_min_value, k2_min_set = optional_bound(k2_min, -np.inf, "k2_min")
        k2_max_value, k2_max_set = optional_bound(k2_max, np.inf, "k2_max")
        if k1_min_value >= k1_max_value:
            raise ValueError("k1_min must be smaller than k1_max.")
        if k2_min_value >= k2_max_value:
            raise ValueError("k2_min must be smaller than k2_max.")
        formula_fix_k1 = bool(fix_k1)
        formula_fix_k2 = bool(fix_k2)
        if formula_fix_k1 and (k1_min_set or k1_max_set) and not (k1_min_value <= fixed_k1_value <= k1_max_value):
            raise ValueError("fixed_k1 must be inside [k1_min, k1_max] when K1 bounds are provided.")
        if formula_fix_k2 and (k2_min_set or k2_max_set) and not (k2_min_value <= fixed_k2_value <= k2_max_value):
            raise ValueError("fixed_k2 must be inside [k2_min, k2_max] when K2 bounds are provided.")
        control_mode_aliases = {
            "direct": "direct",
            "obs": "direct",
            "observation": "direct",
            "theta": "direct",
            "torque": "direct",
            "raw": "direct",
            "single": "direct",
            "single_channel": "direct",
            "formula": "formula",
            "feedback": "formula",
            "action": "formula",
            "gain": "formula",
            "action_formula": "formula",
            "k1k2": "formula",
            "k1_k2": "formula",
            "two_channel": "formula",
            "two_channels": "formula",
            "wave_formula": "formula",
            "wave_feedback": "formula",
            "tail_wave": "tail_wave",
            "tail_curl_wave": "tail_wave",
            "curl_wave": "tail_wave",
            # Paper-formula channel. The passive term -kappa*dtheta_i is
            # supplied by passive_hinge_force; the policy outputs kappa_alpha.
            "paper": "nonreciprocity",
            "paper_formula": "nonreciprocity",
            "nonreciprocity": "nonreciprocity",
            "non_reciprocity": "nonreciprocity",
            "nonreciprocal": "nonreciprocity",
            "odd": "nonreciprocity",
            "odd_elasticity": "nonreciprocity",
            "kappa_alpha": "nonreciprocity",
            # Proposed formula with fixed k1 and a strictly signed k2.
            "k2_positive": "fixed_k1_k2_positive",
            "k2_pos": "fixed_k1_k2_positive",
            "positive_k2": "fixed_k1_k2_positive",
            "fixed_k1_k2_positive": "fixed_k1_k2_positive",
            "fixed_k1_positive": "fixed_k1_k2_positive",
            "k2_negative": "fixed_k1_k2_negative",
            "k2_neg": "fixed_k1_k2_negative",
            "negative_k2": "fixed_k1_k2_negative",
            "fixed_k1_k2_negative": "fixed_k1_k2_negative",
            "fixed_k1_negative": "fixed_k1_k2_negative",
        }
        control_mode_key = str(control_mode).strip().lower().replace("-", "_")
        if control_mode_key not in control_mode_aliases:
            raise ValueError(
                f"Unsupported control_mode: {control_mode!r}. "
                "Use 'direct', 'formula', 'tail_wave', 'nonreciprocity', "
                "'fixed_k1_k2_positive', or 'fixed_k1_k2_negative'."
            )
        control_mode = control_mode_aliases[control_mode_key]
        num_controlled_joints = num_particles - 2 if material_shape == "crawler" else num_particles
        # The wave controller is one global policy agent with six parameters.
        # Legacy modes retain exactly one policy agent per physical joint.
        num_agents = 1 if control_mode == "tail_wave" else num_controlled_joints
        if control_mode == "fixed_k1_k2_positive":
            signed_k2_min_value = k2_min_value if k2_min_set else min_k2_magnitude_value
            signed_k2_max_value = k2_max_value if k2_max_set else max_control_gain
            if signed_k2_min_value <= 0 or signed_k2_max_value <= signed_k2_min_value:
                raise ValueError("k2_positive requires 0 < k2_min < k2_max.")
        elif control_mode == "fixed_k1_k2_negative":
            signed_k2_min_value = k2_min_value if k2_min_set else -max_control_gain
            signed_k2_max_value = k2_max_value if k2_max_set else -min_k2_magnitude_value
            if signed_k2_max_value >= 0 or signed_k2_min_value >= signed_k2_max_value:
                raise ValueError("k2_negative requires k2_min < k2_max < 0.")
        else:
            signed_k2_min_value = k2_min_value
            signed_k2_max_value = k2_max_value
        formula_action_names = []
        if control_mode == "formula":
            if not formula_fix_k1:
                formula_action_names.append("k1")
            if not formula_fix_k2:
                formula_action_names.append("k2")
        particle_radius = np.float32(1/3.0) # edge length
        particle_mass = np.float32(0.2) # kg

        dt = np.float32(3e-3) #integration time step
        physics_steps_per_timestep = 10 # increase this rather than dt to prevent instabilities
        gravity_constant = np.complex64(-1j*1)
        background_friction = np.float32(0.0)
        angle_eq = np.float32(((np.pi * (num_particles - 2)) / num_particles) if material_shape == "ring" else np.pi)

        angle_stiffness = passive_kappa_value
        angle_damping = np.float32(0.42)
        ground_stiffness = np.float32(1e3)
        ground_damping = np.float32(5)
        edge_stiffness = np.float32(1e3)
        edge_damping = np.float32(5)
        edge_length = np.float32(1)
        # Keep the feedback coefficient identical in every experiment.  The
        # command-line/API argument is accepted only for checkpoint compatibility,
        # but it deliberately cannot change the physical model.
        feedback_gain_value = np.float32(1.0)


        self.angle_eq = angle_eq
        self.num_particles = num_particles
        self.num_agents = num_agents
        self.num_controlled_joints = num_controlled_joints
        self.num_envs = num_envs
        self.material_shape = material_shape
        self.edge_length = edge_length
        self.particle_radius = particle_radius
        self.max_steps = max_steps
        self.control_mode = control_mode
        self.action_mode = control_mode
        self.feedback_gain = float(feedback_gain_value)
        # Expose the original-paper contact parameters for metadata/tests.
        self.background_friction = float(background_friction)
        self.ground_stiffness = float(ground_stiffness)
        self.ground_damping = float(ground_damping)
        self.max_torque = float(max_torque)
        self.coefficient_limit = float(max_control_gain)
        self.max_control_gain = float(max_control_gain)  # backward-compatible alias
        self.k1_min = float(k1_min_value)
        self.k1_max = float(k1_max_value)
        self.k2_min = float(signed_k2_min_value if control_mode in {"fixed_k1_k2_positive", "fixed_k1_k2_negative"} else k2_min_value)
        self.k2_max = float(signed_k2_max_value if control_mode in {"fixed_k1_k2_positive", "fixed_k1_k2_negative"} else k2_max_value)
        self.k_action_scale = float(k_action_scale_value)
        self.formula_action_scale = float(k_action_scale_value)
        self.fix_k1 = bool(formula_fix_k1)
        self.formula_fix_k1 = bool(formula_fix_k1)
        self.fixed_k1 = float(fixed_k1_value)
        self.fix_k2 = bool(formula_fix_k2)
        self.formula_fix_k2 = bool(formula_fix_k2)
        self.fixed_k2 = float(fixed_k2_value)
        self.formula_action_names = tuple(formula_action_names)
        self.min_k2_magnitude = float(min_k2_magnitude_value)
        self.passive_kappa = float(passive_kappa_value)

        reward_func = str(reward_func).strip().lower().replace("-", "_")
        if reward_func not in reward_funcs:
            raise ValueError(f"Unsupported reward_func {reward_func!r}. Available: {sorted(reward_funcs)}")
        direction_key = str(rolling_direction).strip().lower()
        if direction_key not in {"right", "left"}:
            raise ValueError("rolling_direction must be 'right' or 'left'.")
        if int(rolling_curl_episodes) < 0 or int(rolling_transition_episodes) < 0:
            raise ValueError("rolling curriculum episode counts must be non-negative.")
        if float(rolling_speed_ref_x100) <= 0 or float(rolling_omega_ref) <= 0 or float(rolling_reward_scale) <= 0:
            raise ValueError("rolling reference values and reward scale must be positive.")
        if float(init_pos_randomness) < 0 or float(init_angle_range_degrees) < 0 or float(init_height_jitter) < 0:
            raise ValueError("initial-state randomization magnitudes must be non-negative.")
        if not np.isfinite(float(action_smoothness_weight)) or float(action_smoothness_weight) < 0:
            raise ValueError("action_smoothness_weight must be a non-negative finite value.")
        tail_side_key = str(tail_side).strip().lower()
        if tail_side_key not in {"left", "right"}:
            raise ValueError("tail_side must be 'left' or 'right'.")
        if str(tail_curl_sign).strip().lower() == "auto":
            # Positions are ordered from left to right at reset.  Reversing
            # that material order reverses the signed joint curvature.
            tail_curl_sign_value = np.float32(1.0 if tail_side_key == "left" else -1.0)
        else:
            tail_curl_sign_value = np.float32(tail_curl_sign)
            if tail_curl_sign_value not in {-1.0, 1.0}:
                raise ValueError("tail_curl_sign must be 'auto', -1, or 1.")
        if int(tail_roll_stage) not in {0, 1, 2, 3}:
            raise ValueError("tail_roll_stage must be one of 0, 1, 2, or 3.")
        if not np.isfinite(float(tail_roll_reward_scale)) or float(tail_roll_reward_scale) <= 0:
            raise ValueError("tail_roll_reward_scale must be a positive finite value.")
        if not (0.0 < float(tail_roll_potential_gamma) <= 1.0):
            raise ValueError("tail_roll_potential_gamma must be in (0, 1].")
        if not np.isfinite(float(tail_roll_contact_margin)) or float(tail_roll_contact_margin) < 0:
            raise ValueError("tail_roll_contact_margin must be a non-negative finite value.")
        if not np.isfinite(float(tail_roll_curl_reference)) or float(tail_roll_curl_reference) <= 0:
            raise ValueError("tail_roll_curl_reference must be a positive finite value.")
        if not np.isfinite(float(tail_roll_init_assist_degrees)) or not (0.0 <= float(tail_roll_init_assist_degrees) < 180.0):
            raise ValueError("tail_roll_init_assist_degrees must be in [0, 180).")
        if int(tail_roll_init_assist_segments) < 1:
            raise ValueError("tail_roll_init_assist_segments must be positive.")
        if int(tail_roll_init_assist_episodes) < 0:
            raise ValueError("tail_roll_init_assist_episodes must be non-negative.")
        if material_shape != "crawler" and control_mode == "tail_wave":
            raise ValueError("tail_wave control is defined only for the crawler material shape.")
        if not np.isfinite(float(tail_wave_amplitude_max)) or float(tail_wave_amplitude_max) <= 0:
            raise ValueError("tail_wave_amplitude_max must be positive.")
        if not np.isfinite(float(tail_wave_center_max)) or float(tail_wave_center_max) <= 0:
            raise ValueError("tail_wave_center_max must be positive.")
        if not (0 < float(tail_wave_width_min) < float(tail_wave_width_max)):
            raise ValueError("tail_wave widths must satisfy 0 < min < max.")
        if float(tail_wave_kp_max) <= 0 or float(tail_wave_kd_max) <= 0:
            raise ValueError("tail_wave gain limits must be positive.")
        if material_shape != "crawler" and (tail_roll_observation or reward_func == "tail_roll_curriculum"):
            raise ValueError("Tail-first rolling is defined only for the crawler material shape.")

        self.reward_func = reward_func
        self.rolling_observation = bool(rolling_observation)
        self.tail_roll_observation = bool(tail_roll_observation)
        self.tail_roll_metrics_enabled = self.tail_roll_observation or reward_func == "tail_roll_curriculum"
        self.rolling_metrics_enabled = (
            self.rolling_observation
            or reward_func == "rolling_curriculum"
            or self.tail_roll_metrics_enabled
        )
        self.rolling_direction = direction_key
        self.rolling_direction_sign = np.float32(1.0 if direction_key == "right" else -1.0)
        self.rolling_curl_episodes = int(rolling_curl_episodes)
        self.rolling_transition_episodes = int(rolling_transition_episodes)
        self.rolling_speed_ref_x100 = np.float32(rolling_speed_ref_x100)
        self.rolling_omega_ref = np.float32(rolling_omega_ref)
        self.rolling_reward_scale = np.float32(rolling_reward_scale)
        self.action_smoothness_weight = np.float32(action_smoothness_weight)
        self.tail_side = tail_side_key
        self.tail_index = 0 if tail_side_key == "left" else num_particles - 1
        self.head_index = num_particles - 1 if tail_side_key == "left" else 0
        self.tail_curl_sign = tail_curl_sign_value
        self.tail_roll_stage = int(tail_roll_stage)
        self.tail_roll_reward_scale = np.float32(tail_roll_reward_scale)
        self.tail_roll_potential_gamma = np.float32(tail_roll_potential_gamma)
        self.tail_roll_contact_margin = np.float32(tail_roll_contact_margin)
        self.tail_roll_curl_reference = np.float32(tail_roll_curl_reference)
        self.tail_roll_init_assist_radians = np.float32(np.deg2rad(tail_roll_init_assist_degrees))
        self.tail_roll_init_assist_segments = min(int(tail_roll_init_assist_segments), num_particles - 1)
        self.tail_roll_init_assist_episodes = int(tail_roll_init_assist_episodes)
        self.tail_roll_init_assist_fraction = np.float32(0.0)
        self.tail_wave_action_names = ("amplitude", "center", "width", "hold", "kp", "kd")
        self.tail_wave_action_low = np.asarray(
            [0.0, 0.0, tail_wave_width_min, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.tail_wave_action_high = np.asarray(
            [tail_wave_amplitude_max, tail_wave_center_max, tail_wave_width_max, 1.0, tail_wave_kp_max, tail_wave_kd_max],
            dtype=np.float32,
        )
        self.init_pos_randomness = np.float32(init_pos_randomness)
        self.init_angle_range_radians = np.float32(np.deg2rad(init_angle_range_degrees))
        self.init_height_jitter = np.float32(init_height_jitter)
        self.curriculum_episode = 0
        self._rolling_metrics_cache = None
        self._tail_roll_metrics_cache = None
        self._tail_previous_potential = None
        self._tail_stage_success_latched = np.zeros((num_envs, 1), dtype=bool)
        self._tail_previous_centered_shape = None
        self._tail_cumulative_rotation = np.zeros((num_envs, 1), dtype=np.float32)
        self._tail_previous_support_x = None
        self._tail_initial_com_x = None
        self._tail_initial_relative_x = None
        initial_action_size = 6 if control_mode == "tail_wave" else max(1, len(formula_action_names))
        self._last_action_for_reward = np.zeros((num_envs, num_agents, initial_action_size), dtype=np.float32)
        self._previous_action_for_reward = np.zeros_like(self._last_action_for_reward, dtype=np.float32)

        self._reward_func = MethodType(reward_funcs[reward_func]['func'], self)
        self._renderer_inited = False
        self._render_flag = render
        self._render_env_colors = None
        if render and render_mode is None:
            render_mode = "human"
        self.render_mode = render_mode
        self.render_text_lines = render_text_lines
        self.window_width = window_width
        self.window_height = window_height

        ####################################### Observation funcs ####################################
        has_boundary_conditions = material_shape == "crawler"

        twopi = np.float32(2*np.pi)

        # roll arrays, numba doesn't support multi-dim rolling :(
        def func(arr):
            rolled = np.empty_like(arr)
            rolled[:, 1:] = arr[:, :-1]
            rolled[:, :1] = arr[:, -1:]
            return rolled
        roll_pos1_complex = nb.njit(
            nb.complex64[:,:](nb.complex64[:,:])
        )(func)
        roll_pos1_real = nb.njit(
            nb.float32[:,:](nb.float32[:,:])
        )(func)
        def func(arr):
            rolled = np.empty_like(arr)
            rolled[:, :-1] = arr[:, 1:]
            rolled[:, -1:] = arr[:, :1]
            return rolled
        roll_neg1_complex = nb.njit(
            nb.complex64[:,:](nb.complex64[:,:])
        )(func)
        roll_neg1_real = nb.njit(
            nb.float32[:,:](nb.float32[:,:])
        )(func)


        @nb.jit(nb.float32[:,:](nb.complex64[:,:]), nopython=False)
        def internode_angle(pos: np.ndarray) -> np.ndarray:
            dp = roll_neg1_complex(pos)-pos
            dp_norm = dp/np.absolute(dp)
            return np.angle(-dp_norm/roll_pos1_complex(dp_norm))%twopi

        observation_funcs = {}
        def obs_func(dim=1, mag=torch.pi):
            def decorator(func):
                name = func.__name__
                assert name.startswith('get_obs_')
                identifier = name[8:]
                observation_funcs[identifier] = {'func': func, 'dim': dim, 'mag': mag}
                return func
            return decorator
        

        @obs_func(dim=1)
        def get_obs_dth_tot(pos, thdot):
            dth = internode_angle(pos) - angle_eq
            if has_boundary_conditions:
                dth[:,0] = 0.
                dth[:,-1] = 0.
            dthP=roll_neg1_real(dth)
            dthM=roll_pos1_real(dth)
            dth_tot = dthP-dthM
            stacked = np.expand_dims(dth_tot, -1)
            if has_boundary_conditions:
                stacked = stacked[:,1:-1]
            return stacked


        @obs_func(dim=2, mag=100)
        def get_obs_dth_tot_plus_feedback_thdot(pos, thdot):
            dth = internode_angle(pos) - angle_eq
            if has_boundary_conditions:
                dth[:, 0] = 0.
                dth[:, -1] = 0.
            dthP = roll_neg1_real(dth)
            dthM = roll_pos1_real(dth)
            dth_tot = dthP - dthM
            feedback_term = feedback_gain_value * thdot
            stacked = np.stack((dth_tot, feedback_term), axis=2)
            if has_boundary_conditions:
                stacked = stacked[:, 1:-1]
            return stacked

        @obs_func(dim=2, mag=100)
        def get_obs_dth_tot_plus_friction_thdot(pos, thdot):
            """
            Observation for RL wave-like locomotion:
                obs[..., 0] = theta(i+1) - theta(i-1)
                obs[..., 1] = F * theta_dot

            F is controlled by the environment argument `feedback_gain` and the
            training/demo CLI option `--feedback-gain`.
            """
            dth = internode_angle(pos) - angle_eq
            if has_boundary_conditions:
                dth[:, 0] = 0.
                dth[:, -1] = 0.
            dthP = roll_neg1_real(dth)
            dthM = roll_pos1_real(dth)
            dth_tot = dthP - dthM
            friction_term = feedback_gain_value * thdot
            stacked = np.stack((dth_tot, friction_term), axis=2)
            if has_boundary_conditions:
                stacked = stacked[:, 1:-1]
            return stacked

        @obs_func(dim=1, mag=100)
        def get_obs_dth_wave_feedback(pos, thdot):
            """
            Single-channel formula-like observation:
                obs = theta(i+1) - theta(i-1) + F * theta_dot

            This is closer to your proposed expression. For training, the two-channel
            dth_tot_plus_friction_thdot version is often more flexible.
            """
            dth = internode_angle(pos) - angle_eq
            if has_boundary_conditions:
                dth[:, 0] = 0.
                dth[:, -1] = 0.
            dthP = roll_neg1_real(dth)
            dthM = roll_pos1_real(dth)
            dth_tot = dthP - dthM
            wave_signal = dth_tot + feedback_gain_value * thdot
            stacked = np.expand_dims(wave_signal, -1)
            if has_boundary_conditions:
                stacked = stacked[:, 1:-1]
            return stacked

        @obs_func(dim=2)
        def get_obs_dth_tot_plus_own(pos, thdot):
            dth = internode_angle(pos) - angle_eq
            if has_boundary_conditions:
                dth[:,0] = 0.
                dth[:,-1] = 0.
            dthP=roll_neg1_real(dth)
            dthM=roll_pos1_real(dth)
            dth_tot = dthP-dthM
            stacked = np.stack((dth_tot, dth), axis=2)
            if has_boundary_conditions:
                stacked = stacked[:,1:-1]
            return stacked

        @obs_func(dim=2)
        def get_obs_dth_neighbours(pos, thdot):
            dth = internode_angle(pos) - angle_eq
            if has_boundary_conditions:
                dth[:,0] = 0.
                dth[:,-1] = 0.
            dthP=roll_neg1_real(dth)
            dthM=roll_pos1_real(dth)
            stacked = np.stack((dthP, dthM), axis=2)
            if has_boundary_conditions:
                stacked = stacked[:,1:-1]
            return stacked

        @obs_func(dim=3)
        def get_obs_dth_neighbours_plus_own(pos, thdot):
            dth = internode_angle(pos) - angle_eq
            if has_boundary_conditions:
                dth[:,0] = 0.
                dth[:,-1] = 0.
            dthP=roll_neg1_real(dth)
            dthM=roll_pos1_real(dth)
            stacked = np.stack((dthP, dth, dthM), axis=2)
            if has_boundary_conditions:
                stacked = stacked[:,1:-1]
            return stacked

        @obs_func(dim=3, mag=100)
        def get_obs_dth_neighbours_plus_thdot(pos, thdot):
            dth = internode_angle(pos) - angle_eq
            if has_boundary_conditions:
                dth[:,0] = 0.
                dth[:,-1] = 0.
            dthP=roll_neg1_real(dth)
            dthM=roll_pos1_real(dth)
            stacked = np.stack((dthP, dthM, thdot), axis=2)
            if has_boundary_conditions:
                stacked = stacked[:,1:-1]
            return stacked
        
        observation_aliases = {
            "theta": "dth_tot",
            "obs": "dth_tot",
            # Exact observation functions used in the thesis/paper.
            "dth": "dth_neighbours",
            "paper_dth": "dth_neighbours",
            "neighbours": "dth_neighbours",
            "neighbors": "dth_neighbours",
            "thdot": "dth_neighbours_plus_thdot",
            "paper_thdot": "dth_neighbours_plus_thdot",
            "neighbours_thdot": "dth_neighbours_plus_thdot",
            "neighbors_thdot": "dth_neighbours_plus_thdot",
            "theta_diff": "dth_tot",
            "single": "dth_tot",
            "one_channel": "dth_tot",
            "1ch": "dth_tot",
            "action": "dth_tot",
            "formula": "dth_tot",
            "action_formula": "dth_tot",
            "theta_feedback": "dth_tot_plus_friction_thdot",
            "feedback_obs": "dth_tot_plus_friction_thdot",
            "theta_dot": "dth_tot_plus_friction_thdot",
            "theta_friction": "dth_tot_plus_friction_thdot",
            "friction": "dth_tot_plus_friction_thdot",
            "dth_tot_plus_feedback_thdot": "dth_tot_plus_friction_thdot",
            "two_channel": "dth_tot_plus_friction_thdot",
            "two_channels": "dth_tot_plus_friction_thdot",
            "2ch": "dth_tot_plus_friction_thdot",
            "two_channel_obs": "dth_tot_plus_friction_thdot",
            "wave": "dth_wave_feedback",
            "wave_feedback": "dth_wave_feedback",
            # action/formula are control-mode aliases; when passed as an
            # observation alias, keep the prompt observation obs=theta(i+1)-theta(i-1).
            "action": "dth_tot",
            "formula": "dth_tot",
            "gain": "dth_tot",
            "wave_sum": "dth_wave_feedback",
            "combined": "dth_wave_feedback",
            "formula_signal": "dth_wave_feedback",
        }
        observation_key = str(observation_func).lower().replace("-", "_")
        observation_func = observation_aliases.get(observation_key, observation_func)
        if observation_func not in observation_funcs:
            raise ValueError(
                f"Unsupported observation_func/channel: {observation_func!r}. "
                f"Available observation functions: {sorted(observation_funcs)}"
            )
        self.observation_func = observation_func
        self._obs_func = nb.njit(nb.float32[:,:,:](nb.complex64[:,:],nb.float32[:,:]))(observation_funcs[observation_func]['func'])

        # Allow user-facing aliases such as terrain_type="tunnel" or "stairs".
        # Internally both are mesh terrains, matching the training metadata format.
        if terrain_type in ["stairs", "tunnel"]:
            preset_type = terrain_type
            if terrain_settings is None:
                terrain_settings = {"type": preset_type}
            elif isinstance(terrain_settings, str):
                terrain_settings = {"type": preset_type}
            elif isinstance(terrain_settings, dict):
                terrain_settings = terrain_settings.copy()
                terrain_settings["type"] = preset_type
            else:
                raise ValueError(f"terrain_settings for {preset_type!r} must be None, str, or dict.")
            terrain_type = "mesh"

        ######################################## Physics Sim ###########################################

        # passive hinge force
        def func(o: np.ndarray, oC: np.ndarray, oM: np.ndarray,
                dth: np.ndarray, dthM: np.ndarray, dthP: np.ndarray,
                thdot: np.ndarray, thdotP: np.ndarray, thdotM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            fE = angle_stiffness * (dth * oC - dthP * o - dthM * oM)
            fD = angle_damping * (thdot * oC - thdotP * o - thdotM * oM)

            if has_boundary_conditions:
                fE[:, 0] = angle_stiffness * (-dthP[:, 0] * o[:, 0])
                fE[:, -1] = angle_stiffness * (-dthM[:, -1] * oM[:, -1])
                fE[:, 1] = angle_stiffness * (dth[:, 1] * oC[:, 1] - dthP[:, 1] * o[:, 1])
                fE[:, -2] = angle_stiffness * (dth[:, -2] * oC[:, -2] - dthM[:, -2] * oM[:, -2])

                fD[:, 0] = angle_damping * (-thdotP[:, 0] * o[:, 0])
                fD[:, -1] = angle_damping * (-thdotM[:, -1] * oM[:, -1])
                fD[:, 1] = angle_damping * (thdot[:, 1] * oC[:, 1] - thdotP[:, 1] * o[:, 1])
                fD[:, -2] = angle_damping * (thdot[:, -2] * oC[:, -2] - thdotM[:, -2] * oM[:, -2])

            return fE, fD
        passive_hinge_force = nb.njit(
            nb.types.UniTuple(nb.complex64[:,:], 2)
            (nb.complex64[:,:],nb.complex64[:,:],nb.complex64[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32[:,:])
        )(func)

        # convert torque to force
        def func(tau: np.ndarray, o: np.ndarray, oM: np.ndarray) -> np.ndarray:
            fM = np.float32(0.5) * tau * oM
            fP = np.float32(0.5) * tau * o
            fself = -(fM + fP)
            fM = roll_neg1_complex(fM)
            fP = roll_pos1_complex(fP)
            fO = fself + fM + fP
            return fO
        convert_torque_to_force = nb.njit(
            nb.complex64[:,:](nb.float32[:,:],nb.complex64[:,:],nb.complex64[:,:])
        )(func)

        # edge force
        def func(pos: np.ndarray, r: np.ndarray, dv: np.ndarray) -> np.ndarray:
            fs = np.zeros_like(pos)
            mag = np.abs(r)

            if has_boundary_conditions:
                mag[:, -1] = 1
                dv[:, -1] = 0.

            rh = r / mag
            SMag = edge_stiffness * (mag - edge_length)
            DMag = edge_damping * np.real(dv * np.conj(rh))
            T = rh * (SMag + DMag)
            fs = T - roll_pos1_complex(T)

            return fs
        edge_force = nb.njit(
            nb.complex64[:,:](nb.complex64[:,:],nb.complex64[:,:],nb.complex64[:,:])
        )(func)
        
        # slide force
        def func(vel: np.ndarray) -> np.ndarray:
            ff = -background_friction * vel
            return ff
        slide_force = nb.njit(
            nb.complex64[:,:](nb.complex64[:,:])
        )(func)

        # wall force
        if terrain_type == "flat":
            def func(pos: np.ndarray, vel: np.ndarray):
                fw = np.zeros_like(pos)

                for i in range(num_envs): # numba doesn't support multi-dim indexing mask :(
                    msk = np.imag(pos[i]) < particle_radius
                    fw[i][msk] = -ground_damping * vel[i][msk] - 1j * ground_stiffness * (np.imag(pos[i][msk]) - particle_radius)

                return fw
            self._render_terrain = self._render_terrain_flat
        elif terrain_type == "mesh":
            terrain_mesh = MeshTerrain(terrain_settings).lines
            self.terrain_mesh = terrain_mesh
            self._render_terrain = self._render_terrain_mesh
            def func(pos: np.ndarray, vel: np.ndarray):
                fw = np.zeros_like(pos)

                u = terrain_mesh[:,1] - terrain_mesh[:,0]
                            

                for i in range(num_envs):
                    for j in range(num_particles):
                        p = pos[i,j]
                        v = -terrain_mesh[:,0] + p
                        v_onto_u = u * np.clip(np.real(v / u), 0, 1) # do not exceed end points of line piece
                        p_onto_u = terrain_mesh[:,0] + v_onto_u
                        dist = np.abs(p - p_onto_u)
                        collide = dist < particle_radius
                        if not np.any(collide):
                            continue
                        closest = np.argmin(dist)
                        depth = particle_radius - dist[closest]
                        normal = (p - p_onto_u[closest]) / dist[closest]
                        fw[i,j] = -ground_damping * vel[i,j] + normal * ground_stiffness * depth

                # for i in range(num_envs): # numba doesn't support multi-dim indexing mask :(
                #     msk = terrain_density(np.real(pos[i]), np.imag(pos[i]) - rad) > 0
                #     fw[i][msk] = -gs * vel[i][msk] + 1j * ks * terrain_density(np.real(pos[i]), np.imag(pos[i]) - rad)[msk]

                return fw
        elif terrain_type == "mesh_cycle":
            terrain_mesh = [MeshTerrain(s).lines for s in terrain_settings]
            self.terrain_mesh = terrain_mesh
            self._render_terrain = self._render_terrain_mesh_cycle
            # mesh_list = nb.typed.Dict(enumerate([m[:,1] - m[:,0] for m in terrain_mesh]))
            # I cannot figure out how to compile a list or dict into numba so I'm
            # just gonna hardcode 3 elements, sorry :(
            mesh0 = terrain_mesh[0]
            mesh1 = terrain_mesh[1]
            mesh2 = terrain_mesh[2]
            def func(pos: np.ndarray, vel: np.ndarray):
                fw = np.zeros_like(pos)
                
                u0 = mesh0[:,1] - mesh0[:,0]
                u1 = mesh1[:,1] - mesh1[:,0]
                u2 = mesh2[:,1] - mesh2[:,0]
                            
                for i in range(num_envs):
                    mesh_index = i % 3
                    if mesh_index == 0:
                        mesh = mesh0
                        u = u0
                    if mesh_index == 1:
                        mesh = mesh1
                        u = u1
                    if mesh_index == 2:
                        mesh = mesh2
                        u = u2
                    for j in range(num_particles):
                        p = pos[i,j]
                        v = -mesh[:,0] + p
                        v_onto_u = u * np.clip(np.real(v / u), 0, 1) # do not exceed end points of line piece
                        p_onto_u = mesh[:,0] + v_onto_u
                        dist = np.abs(p - p_onto_u)
                        collide = dist < particle_radius
                        if not np.any(collide):
                            continue
                        closest = np.argmin(dist)
                        depth = particle_radius - dist[closest]
                        normal = (p - p_onto_u[closest]) / dist[closest]
                        fw[i,j] = -ground_damping * vel[i,j] + normal * ground_stiffness * depth

                return fw

        else:
            raise ValueError(f"Unsupported terrain_type: {terrain_type!r}.")

        wall_force = nb.njit(
            nb.complex64[:,:](nb.complex64[:,:],nb.complex64[:,:])
        )(func)

        # gravity
        def func(pos: np.ndarray) -> np.ndarray:
            fg = gravity_constant * np.ones_like(pos)
            return fg
        grav_force = nb.njit(
            nb.complex64[:,:](nb.complex64[:,:])
        )(func)

        # calculate useful quantities
        def func(pos: np.ndarray, vel: np.ndarray):
            r = roll_neg1_complex(pos) - pos
            dv = roll_neg1_complex(vel) - vel
            mag = np.abs(r)
            rh = r / mag
            th = np.angle(-rh / roll_pos1_complex(rh)) % twopi
            dth = th - angle_eq # angle_eq was set as np.pi, even for the ring?????
            
            dthP = roll_neg1_real(dth)
            dthM = roll_pos1_real(dth)
            
            o = np.complex64(1j) * rh / mag
            oM = roll_pos1_complex(o)
            oC = o + oM
            
            velP = roll_neg1_complex(vel)
            velM = roll_pos1_complex(vel)
            
            thdot = np.real(-oC * np.conj(vel) + o * np.conj(velP) + oM * np.conj(velM))
            thdotP = roll_neg1_real(thdot)
            thdotM = roll_pos1_real(thdot)
            
            return r, dv, mag, rh, th, dth, dthP, dthM, thdot, thdotP, thdotM, o, oC, oM
        calculate = nb.njit(
            nb.types.Tuple((nb.complex64[:,:],nb.complex64[:,:],nb.float32[:,:],nb.complex64[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32[:,:],nb.float32[:,:],nb.complex64[:,:],nb.complex64[:,:],nb.complex64[:,:]))(nb.complex64[:,:],nb.complex64[:,:])
        )(func)

        if control_mode == "tail_wave":
            action_size = 6
        else:
            action_size = len(formula_action_names) if control_mode == "formula" and formula_action_names else 1

        # get total forces
        if control_mode == "tail_wave":
            # One global action defines a smooth cumulative curl front plus a
            # local travelling bump.  A target-curvature PD term creates
            # start-up torque even from an exactly straight, stationary body.
            if tail_side_key == "left":
                joint_coordinate = np.linspace(0.0, 1.0, num_controlled_joints, dtype=np.float32)[None, :]
            else:
                joint_coordinate = np.linspace(1.0, 0.0, num_controlled_joints, dtype=np.float32)[None, :]

            def func(action_params: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                r, dv, mag, rh, th, dth, dthP, dthM, thdot, thdotP, thdotM, o, oC, oM = calculate(pos, vel)
                fE, fD = passive_hinge_force(o, oC, oM, dth, dthM, dthP, thdot, thdotP, thdotM)

                amplitude = action_params[:, 0, 0:1]
                center_value = action_params[:, 0, 1:2]
                width = np.maximum(action_params[:, 0, 2:3], np.float32(tail_wave_width_min))
                hold = action_params[:, 0, 3:4]
                kp = action_params[:, 0, 4:5]
                kd = action_params[:, 0, 5:6]
                distance = (center_value - joint_coordinate) / width
                front = np.float32(0.5) * (np.float32(1.0) + np.tanh(distance))
                bump = np.exp(np.float32(-0.5) * ((joint_coordinate - center_value) / width) ** 2)
                profile = hold * front + (np.float32(1.0) - hold) * bump
                target_curvature = tail_curl_sign_value * amplitude * profile
                current_curvature = dth[:, 1:-1]
                current_thdot = thdot[:, 1:-1]
                controlled_tau = np.clip(
                    kp * (target_curvature - current_curvature) - kd * current_thdot,
                    -max_torque,
                    max_torque,
                )
                tauO = np.zeros_like(dth)
                tauO[:, 1:-1] = controlled_tau
                fO = convert_torque_to_force(tauO, o, oM)
                fs = edge_force(pos, r, dv)
                ff = slide_force(vel)
                fw = wall_force(pos, vel)
                fg = grav_force(pos)
                return fO + fs + ff + fw + fg + fE + fD, thdot

            get_forces = nb.njit(
                nb.types.Tuple((nb.complex64[:,:],nb.float32[:,:]))(nb.float32[:,:,:],nb.complex64[:,:],nb.complex64[:,:])
            )(func)

            def func(action_params: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                for _ in range(physics_steps_per_timestep):
                    f, _ = get_forces(action_params, pos, vel)
                    vel += f * 0.5 * dt / particle_mass
                    pos += vel * dt
                    f, thdot = get_forces(action_params, pos, vel)
                    vel += f * 0.5 * dt / particle_mass
                return pos, vel, thdot

            velocity_verlet = nb.njit(
                nb.types.Tuple((nb.complex64[:,:],nb.complex64[:,:],nb.float32[:,:]))(nb.float32[:,:,:],nb.complex64[:,:],nb.complex64[:,:])
            )(func)
        elif control_mode == "formula":
            # In formula mode, the policy outputs normalized coefficients
            # per controlled joint by default: u1 and u2. They are converted to
            # physical gains by K = k_action_scale * u, then optional K bounds
            # are applied. Either coefficient can also be fixed, in which case
            # it is removed from the policy output. The environment turns the
            # resulting physical gains into the physical torque
            #     tau = k1 * (theta(i+1)-theta(i-1)) + k2 * F * theta_dot
            # at each physics sub-step, then clips tau to the actuator range.
            k1_action_index = 0
            k2_action_index = 0 if formula_fix_k1 else 1

            def func(action_params: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

                r, dv, mag, rh, th, dth, dthP, dthM, thdot, thdotP, thdotM, o, oC, oM = calculate(pos, vel)

                fE, fD = passive_hinge_force(o, oC, oM, dth, dthM, dthP, thdot, thdotP, thdotM)

                dth_for_action = dth.copy()
                if has_boundary_conditions:
                    dth_for_action[:, 0] = 0.
                    dth_for_action[:, -1] = 0.
                dth_tot = roll_neg1_real(dth_for_action) - roll_pos1_real(dth_for_action)
                if formula_fix_k1:
                    k1 = np.zeros_like(dth_tot) + fixed_k1_value
                else:
                    raw_k1 = action_params[:, :, k1_action_index] * k_action_scale_value
                    k1 = np.minimum(
                        np.maximum(raw_k1, k1_min_value),
                        k1_max_value,
                    )
                if formula_fix_k2:
                    k2 = np.zeros_like(dth_tot) + fixed_k2_value
                else:
                    raw_k2 = action_params[:, :, k2_action_index] * k_action_scale_value
                    k2 = np.minimum(
                        np.maximum(raw_k2, k2_min_value),
                        k2_max_value,
                    )
                tauO = np.clip(k1 * dth_tot + k2 * feedback_gain_value * thdot, -max_torque, max_torque)
                if has_boundary_conditions:
                    tauO[:, 0] = 0.
                    tauO[:, -1] = 0.
                fO = convert_torque_to_force(tauO, o, oM)

                fs = edge_force(pos, r, dv)
                ff = slide_force(vel)
                fw = wall_force(pos, vel)
                fg = grav_force(pos)

                ftot = fO + fs + ff + fw + fg + fE + fD

                return ftot, thdot
            get_forces = nb.njit(
                nb.types.Tuple((nb.complex64[:,:],nb.float32[:,:]))(nb.float32[:,:,:],nb.complex64[:,:],nb.complex64[:,:])
            )(func)

            # velocity verlet
            def func(action_params: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                if has_boundary_conditions:
                    a = np.zeros((num_envs, num_particles, action_size), dtype=np.float32)
                    a[:, 1:-1, :] = action_params
                    action_params = a
                for _ in range(physics_steps_per_timestep):
                    f, _ = get_forces(action_params, pos, vel)
                    vel += f * 0.5 * dt / particle_mass
                    pos += vel * dt
                    f, thdot = get_forces(action_params, pos, vel)
                    vel += f * 0.5 * dt / particle_mass
                return pos, vel, thdot
            velocity_verlet = nb.njit(
                nb.types.Tuple((nb.complex64[:,:],nb.complex64[:,:],nb.float32[:,:]))(nb.float32[:,:,:],nb.complex64[:,:],nb.complex64[:,:])
            )(func)
        elif control_mode in {
            "nonreciprocity",
            "fixed_k1_k2_positive",
            "fixed_k1_k2_negative",
        }:
            # All three modes use a single policy output per joint.
            #
            # nonreciprocity:
            #   total torque = -kappa*dtheta_i
            #                  + kappa_alpha*(dtheta_{i+1}-dtheta_{i-1})
            #   The passive -kappa*dtheta_i term is already supplied by fE, while
            #   the policy output is the learnable kappa_alpha coefficient.
            #
            # fixed_k1_k2_positive / fixed_k1_k2_negative:
            #   active torque = fixed_k1*(theta_{i+1}-theta_{i-1})
            #                   + k2*F*theta_dot_i
            #   Only k2 is learned, and its sign is enforced here in addition to
            #   the action-spec bounds.
            is_nonreciprocity = control_mode == "nonreciprocity"

            def func(action_coefficient: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

                r, dv, mag, rh, th, dth, dthP, dthM, thdot, thdotP, thdotM, o, oC, oM = calculate(pos, vel)

                fE, fD = passive_hinge_force(o, oC, oM, dth, dthM, dthP, thdot, thdotP, thdotM)

                dth_for_action = dth.copy()
                if has_boundary_conditions:
                    dth_for_action[:, 0] = 0.
                    dth_for_action[:, -1] = 0.
                dth_tot = roll_neg1_real(dth_for_action) - roll_pos1_real(dth_for_action)

                if is_nonreciprocity:
                    kappa_alpha = np.minimum(
                        np.maximum(action_coefficient, -max_control_gain),
                        max_control_gain,
                    )
                    tauO = np.clip(kappa_alpha * dth_tot, -max_torque, max_torque)
                else:
                    k2 = np.minimum(
                        np.maximum(action_coefficient, signed_k2_min_value),
                        signed_k2_max_value,
                    )
                    tauO = np.clip(
                        fixed_k1_value * dth_tot + k2 * feedback_gain_value * thdot,
                        -max_torque,
                        max_torque,
                    )

                if has_boundary_conditions:
                    tauO[:, 0] = 0.
                    tauO[:, -1] = 0.
                fO = convert_torque_to_force(tauO, o, oM)

                fs = edge_force(pos, r, dv)
                ff = slide_force(vel)
                fw = wall_force(pos, vel)
                fg = grav_force(pos)

                ftot = fO + fs + ff + fw + fg + fE + fD

                return ftot, thdot
            get_forces = nb.njit(
                nb.types.Tuple((nb.complex64[:,:],nb.float32[:,:]))(nb.float32[:,:],nb.complex64[:,:],nb.complex64[:,:])
            )(func)

            # velocity verlet
            def func(action_coefficient: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                if has_boundary_conditions:
                    a = np.zeros((num_envs, num_particles), dtype=np.float32)
                    a[:, 1:-1] = action_coefficient
                    action_coefficient = a
                for _ in range(physics_steps_per_timestep):
                    f, _ = get_forces(action_coefficient, pos, vel)
                    vel += f * 0.5 * dt / particle_mass
                    pos += vel * dt
                    f, thdot = get_forces(action_coefficient, pos, vel)
                    vel += f * 0.5 * dt / particle_mass
                return pos, vel, thdot
            velocity_verlet = nb.njit(
                nb.types.Tuple((nb.complex64[:,:],nb.complex64[:,:],nb.float32[:,:]))(nb.float32[:,:],nb.complex64[:,:],nb.complex64[:,:])
            )(func)
        else:
            def func(action: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

                r, dv, mag, rh, th, dth, dthP, dthM, thdot, thdotP, thdotM, o, oC, oM = calculate(pos, vel)

                fE, fD = passive_hinge_force(o, oC, oM, dth, dthM, dthP, thdot, thdotP, thdotM)

                tauO = action
                fO = convert_torque_to_force(tauO, o, oM)

                fs = edge_force(pos, r, dv)
                ff = slide_force(vel)
                fw = wall_force(pos, vel)
                fg = grav_force(pos)

                ftot = fO + fs + ff + fw + fg + fE + fD

                return ftot, thdot
            get_forces = nb.njit(
                nb.types.Tuple((nb.complex64[:,:],nb.float32[:,:]))(nb.float32[:,:],nb.complex64[:,:],nb.complex64[:,:])
            )(func)

            # velocity verlet
            def func(action: np.ndarray, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                if has_boundary_conditions:
                    a = np.zeros((num_envs, num_particles), dtype=np.float32)
                    a[:,1:-1] = action
                    action = a
                for _ in range(physics_steps_per_timestep):
                    f, _ = get_forces(action, pos, vel)
                    vel += f * 0.5 * dt / particle_mass
                    pos += vel * dt
                    f, thdot = get_forces(action, pos, vel)
                    vel += f * 0.5 * dt / particle_mass
                return pos, vel, thdot
            velocity_verlet = nb.njit(
                nb.types.Tuple((nb.complex64[:,:],nb.complex64[:,:],nb.float32[:,:]))(nb.float32[:,:],nb.complex64[:,:],nb.complex64[:,:])
            )(func)

        self._physics_sim = velocity_verlet
        ####################################### TorchRL Interface ####################################
        base_observation_size = observation_funcs[observation_func]['dim']
        if control_mode == "tail_wave":
            # The sole global agent observes every physical joint.
            base_observation_size *= num_controlled_joints
        self.base_observation_size = int(base_observation_size)
        self.rolling_observation_size = 6 if self.rolling_observation else 0
        # 14 kinematic/shape features + 4-stage one-hot + 1 open-chain
        # joint coordinate measured from the configured tail.
        self.tail_roll_observation_size = 19 if self.tail_roll_observation else 0
        observation_size = (
            base_observation_size
            + self.rolling_observation_size
            + self.tail_roll_observation_size
        )
        observation_mag = observation_funcs[observation_func]['mag']
        log_info_specs = {
            "speed": Unbounded(
                shape=torch.Size([num_envs, 1]),
                dtype=torch.float32,
            )
        }
        if self.rolling_metrics_enabled:
            for metric_name in (
                "closure_score",
                "circularity_score",
                "body_omega",
                "slip_penalty",
                "action_smoothness_penalty",
                "curriculum_progress",
                "rolling_reward",
            ):
                log_info_specs[metric_name] = Unbounded(
                    shape=torch.Size([num_envs, 1]),
                    dtype=torch.float32,
                )
        if self.tail_roll_metrics_enabled:
            for metric_name in (
                "tail_lift_score",
                "tail_forward_score",
                "head_contact_score",
                "curl_prefix_progress",
                "curl_order_penalty",
                "total_signed_curvature",
                "closure_ratio",
                "support_margin",
                "cumulative_rotation",
                "rolling_gate",
                "tail_roll_stage",
                "tail_stage_success",
                "tail_roll_reward",
            ):
                log_info_specs[metric_name] = Unbounded(
                    shape=torch.Size([num_envs, 1]),
                    dtype=torch.float32,
                )
        if control_mode == "tail_wave":
            for metric_name in (
                "wave_amplitude", "wave_center", "wave_width",
                "wave_hold", "wave_kp", "wave_kd",
            ):
                log_info_specs[metric_name] = Unbounded(
                    shape=torch.Size([num_envs, 1]), dtype=torch.float32
                )
        self.observation_spec = Composite(
            agents = Composite(
                observation = Bounded (
                    low = -observation_mag,
                    high = observation_mag,
                    shape = torch.Size([num_envs, num_agents, observation_size]),
                    dtype = torch.float32
                ),
                shape = torch.Size([num_envs, num_agents])
            ),
            log_info = Composite(**log_info_specs, shape=torch.Size([num_envs])),
            shape = torch.Size([num_envs])
        )

        self.action_size = action_size
        if control_mode == "tail_wave":
            action_low = torch.as_tensor(self.tail_wave_action_low, dtype=torch.float32).view(1, 1, action_size).expand(num_envs, num_agents, action_size).clone()
            action_high = torch.as_tensor(self.tail_wave_action_high, dtype=torch.float32).view(1, 1, action_size).expand(num_envs, num_agents, action_size).clone()
            action_spec = Bounded
        elif control_mode == "formula":
            action_low_values = []
            action_high_values = []
            if not formula_fix_k1:
                action_low_values.append(float(k1_min_value / k_action_scale_value))
                action_high_values.append(float(k1_max_value / k_action_scale_value))
            if not formula_fix_k2:
                action_low_values.append(float(k2_min_value / k_action_scale_value))
                action_high_values.append(float(k2_max_value / k_action_scale_value))
            if not action_low_values:
                action_low_values = [-1.0]
                action_high_values = [1.0]
            if np.all(np.isfinite(action_low_values)) and np.all(np.isfinite(action_high_values)):
                action_low = torch.as_tensor(action_low_values, dtype=torch.float32).view(1, 1, action_size).expand(num_envs, num_agents, action_size).clone()
                action_high = torch.as_tensor(action_high_values, dtype=torch.float32).view(1, 1, action_size).expand(num_envs, num_agents, action_size).clone()
                action_spec = Bounded
            else:
                action_low = None
                action_high = None
                action_spec = Unbounded
        elif control_mode == "nonreciprocity":
            action_low = -max_control_gain
            action_high = max_control_gain
            action_spec = Bounded
        elif control_mode == "fixed_k1_k2_positive":
            action_low = signed_k2_min_value
            action_high = signed_k2_max_value
            action_spec = Bounded
        elif control_mode == "fixed_k1_k2_negative":
            action_low = signed_k2_min_value
            action_high = signed_k2_max_value
            action_spec = Bounded
        else:
            action_low = -max_torque
            action_high = max_torque
            action_spec = Bounded
        action_shape = torch.Size([num_envs, num_agents, action_size])
        if action_spec is Bounded:
            action_tensor_spec = Bounded (
                low = action_low,
                high = action_high,
                shape = action_shape,
                dtype = torch.float32
            )
        else:
            action_tensor_spec = Unbounded (
                shape = action_shape,
                dtype = torch.float32
            )
        self.action_spec = Composite(
            agents = Composite(
                action = action_tensor_spec,
                shape = torch.Size([num_envs, num_agents])
            ),
            shape = torch.Size([num_envs])
        )

    def set_curriculum_episode(self, episode):
        self.curriculum_episode = max(0, int(episode))

    def set_tail_roll_stage(self, stage):
        stage = int(stage)
        if stage not in {0, 1, 2, 3}:
            raise ValueError("tail roll stage must be one of 0, 1, 2, or 3")
        if stage != self.tail_roll_stage:
            self.tail_roll_stage = stage
            self._tail_previous_potential = None
            self._tail_stage_success_latched[...] = False
            self._tail_roll_metrics_cache = None

    @staticmethod
    def _smooth_step(value):
        value = np.clip(np.asarray(value, dtype=np.float32), -20.0, 20.0)
        return np.float32(1.0) / (np.float32(1.0) + np.exp(-value))

    def _compute_rolling_metrics(self):
        if self._rolling_metrics_cache is not None:
            return self._rolling_metrics_cache

        pos = np.asarray(self.pos, dtype=np.complex64)
        vel = np.asarray(self.vel, dtype=np.complex64)
        center = np.mean(pos, axis=1, keepdims=True)
        rel = pos - center
        segments = np.diff(pos, axis=1)
        body_length = np.sum(np.abs(segments), axis=1, keepdims=True)
        end_distance = np.abs(pos[:, -1:] - pos[:, :1])
        closure_score = np.float32(1.0) - np.clip(end_distance / np.maximum(body_length, np.float32(1e-6)), 0.0, 1.0)

        radius = np.abs(rel)
        radius_mean = np.mean(radius, axis=1, keepdims=True)
        radius_cv = np.std(radius, axis=1, keepdims=True) / np.maximum(radius_mean, np.float32(1e-6))
        circularity_score = np.float32(1.0) - np.clip(radius_cv, 0.0, 1.0)

        inertia = np.sum(np.abs(rel) ** 2, axis=1, keepdims=True)
        body_omega = np.sum(np.imag(np.conj(rel) * vel), axis=1, keepdims=True) / np.maximum(inertia, np.float32(1e-6))
        com_velocity = np.real(np.mean(vel, axis=1, keepdims=True))
        rolling_surface_velocity = radius_mean * body_omega
        slip_penalty = np.abs(com_velocity + rolling_surface_velocity) / np.maximum(
            np.abs(com_velocity) + np.abs(rolling_surface_velocity), np.float32(1e-6)
        )

        speed_x100 = np.asarray(self.mean_speed, dtype=np.float32) * np.float32(100.0)
        speed_score = np.tanh(self.rolling_direction_sign * speed_x100 / self.rolling_speed_ref_x100)
        desired_omega_sign = -self.rolling_direction_sign
        rotation_score = np.tanh(desired_omega_sign * body_omega / self.rolling_omega_ref)

        action = np.asarray(self._last_action_for_reward, dtype=np.float32)
        if self.control_mode == "tail_wave":
            low = self.tail_wave_action_low.reshape(1, 1, -1)
            span = np.maximum((self.tail_wave_action_high - self.tail_wave_action_low).reshape(1, 1, -1), np.float32(1e-6))
            normalized_action = np.float32(2.0) * (action - low) / span - np.float32(1.0)
            effort_penalty = np.mean(np.clip(normalized_action ** 2, 0.0, 1.0), axis=(1, 2))[:, None]
        else:
            effort_penalty = np.mean(np.tanh(action ** 2), axis=tuple(range(1, action.ndim)), keepdims=False)[:, None]
        previous_action = np.asarray(self._previous_action_for_reward, dtype=np.float32)
        if previous_action.shape != action.shape:
            previous_action = np.reshape(previous_action, action.shape)
        action_delta = action - previous_action
        if self.control_mode == "tail_wave":
            span = np.maximum((self.tail_wave_action_high - self.tail_wave_action_low).reshape(1, 1, -1), np.float32(1e-6))
            action_delta = action_delta / span
        elif self.control_mode == "direct":
            action_delta = action_delta / np.float32(max(self.max_torque, 1e-6))
        else:
            action_delta = np.tanh(action_delta)
        action_smoothness_penalty = np.mean(
            np.clip(action_delta ** 2, 0.0, 1.0),
            axis=tuple(range(1, action_delta.ndim)),
            keepdims=False,
        )[:, None]

        if self.rolling_transition_episodes <= 0:
            curriculum_value = np.float32(1.0 if self.curriculum_episode >= self.rolling_curl_episodes else 0.0)
        else:
            curriculum_value = np.float32(
                np.clip(
                    (self.curriculum_episode - self.rolling_curl_episodes) / self.rolling_transition_episodes,
                    0.0,
                    1.0,
                )
            )
        curriculum_progress = np.full((self.num_envs, 1), curriculum_value, dtype=np.float32)

        self._rolling_metrics_cache = {
            "closure_score": np.asarray(closure_score, dtype=np.float32),
            "circularity_score": np.asarray(circularity_score, dtype=np.float32),
            "body_omega": np.asarray(body_omega, dtype=np.float32),
            "slip_penalty": np.asarray(slip_penalty, dtype=np.float32),
            "action_smoothness_penalty": np.asarray(action_smoothness_penalty, dtype=np.float32),
            "speed_score": np.asarray(speed_score, dtype=np.float32),
            "rotation_score": np.asarray(rotation_score, dtype=np.float32),
            "effort_penalty": np.asarray(effort_penalty, dtype=np.float32),
            "curriculum_progress": curriculum_progress,
            "rolling_reward": np.zeros((self.num_envs, 1), dtype=np.float32),
        }
        return self._rolling_metrics_cache

    def _compute_tail_roll_metrics(self):
        if self._tail_roll_metrics_cache is not None:
            return self._tail_roll_metrics_cache

        pos = np.asarray(self.pos, dtype=np.complex64)
        if self.tail_side == "left":
            ordered = pos
        else:
            ordered = pos[:, ::-1]

        x = np.real(ordered)
        y = np.imag(ordered)
        center = np.mean(ordered, axis=1, keepdims=True)
        centered = ordered - center
        segments = np.diff(ordered, axis=1)
        body_length = np.maximum(
            np.sum(np.abs(segments), axis=1, keepdims=True),
            np.float32(1e-6),
        )
        tail = ordered[:, :1]
        head = ordered[:, -1:]

        tail_lift = (np.imag(tail) - np.imag(head)) / (np.float32(0.35) * body_length)
        tail_lift_score = np.clip(tail_lift, 0.0, 1.0)
        head_height = np.maximum(np.imag(head) - self.particle_radius, 0.0)
        contact_scale = np.maximum(self.tail_roll_contact_margin * body_length, np.float32(1e-4))
        head_contact_score = np.exp(-np.square(head_height / contact_scale))
        head_lift_score = np.clip(head_height / (np.float32(0.25) * body_length), 0.0, 1.0)

        com_x = np.real(center)
        tail_relative_x = np.real(tail) - com_x
        if self._tail_initial_relative_x is None:
            self._tail_initial_relative_x = tail_relative_x.copy()
        direction = self.rolling_direction_sign
        tail_forward = direction * (tail_relative_x - self._tail_initial_relative_x)
        tail_forward_score = np.clip(tail_forward / (np.float32(0.35) * body_length), 0.0, 1.0)

        if segments.shape[1] >= 2:
            joint_curvature = np.angle(segments[:, 1:] * np.conj(segments[:, :-1])).astype(np.float32)
            signed_curvature = self.tail_curl_sign * joint_curvature
            activation = np.clip(signed_curvature / self.tail_roll_curl_reference, 0.0, 1.0)
            prefix_activation = np.minimum.accumulate(activation, axis=1)
            curl_prefix_progress = np.mean(prefix_activation, axis=1, keepdims=True)
            curl_order_penalty = np.mean(
                np.maximum(activation[:, 1:] - activation[:, :-1], 0.0),
                axis=1,
                keepdims=True,
            ) if activation.shape[1] > 1 else np.zeros((self.num_envs, 1), dtype=np.float32)
            total_signed_curvature = np.sum(signed_curvature, axis=1, keepdims=True)
        else:
            curl_prefix_progress = np.zeros((self.num_envs, 1), dtype=np.float32)
            curl_order_penalty = np.zeros((self.num_envs, 1), dtype=np.float32)
            total_signed_curvature = np.zeros((self.num_envs, 1), dtype=np.float32)

        closure_ratio = np.abs(tail - head) / body_length
        closure_score = np.float32(1.0) - np.clip(closure_ratio, 0.0, 1.0)
        turn_progress = np.clip(total_signed_curvature / np.float32(1.5 * np.pi), 0.0, 1.0)
        loop_gate = self._smooth_step((np.float32(0.25) - closure_ratio) / np.float32(0.05))
        loop_gate *= self._smooth_step(
            (total_signed_curvature - np.float32(1.5 * np.pi)) / np.float32(0.25 * np.pi)
        )

        contact_softness = np.maximum(np.float32(0.02) * body_length, np.float32(1e-4))
        contact_weights = self._smooth_step(
            (self.particle_radius + self.tail_roll_contact_margin * body_length - y) / contact_softness
        )
        support_x = np.sum(contact_weights * x, axis=1, keepdims=True) / np.maximum(
            np.sum(contact_weights, axis=1, keepdims=True),
            np.float32(1e-6),
        )
        support_margin = direction * (com_x - support_x) / body_length
        pivot_gate = self._smooth_step(support_margin / np.float32(0.05))

        if self._tail_previous_centered_shape is None:
            rotation_increment = np.zeros((self.num_envs, 1), dtype=np.float32)
        else:
            fit = np.sum(
                np.conj(self._tail_previous_centered_shape) * centered,
                axis=1,
                keepdims=True,
            )
            rotation_increment = np.angle(fit).astype(np.float32)
        self._tail_previous_centered_shape = centered.copy()
        self._tail_cumulative_rotation += rotation_increment
        desired_rotation_increment = np.tanh(
            (-direction * rotation_increment) / np.float32(0.01)
        )
        desired_rotation = -direction * self._tail_cumulative_rotation
        rotation_start_progress = np.clip(
            desired_rotation / np.float32(0.5 * np.pi), 0.0, 1.0
        )
        rotation_turn_progress = np.clip(
            desired_rotation / np.float32(2.0 * np.pi), 0.0, 1.0
        )

        if self._tail_initial_com_x is None:
            self._tail_initial_com_x = com_x.copy()
        forward_displacement = direction * (com_x - self._tail_initial_com_x) / body_length
        forward_displacement_increment = np.tanh(
            direction * np.asarray(self.mean_speed, dtype=np.float32)
            / np.maximum(np.float32(0.002) * body_length, np.float32(1e-5))
        )
        if self._tail_previous_support_x is None:
            contact_migration_increment = np.zeros((self.num_envs, 1), dtype=np.float32)
        else:
            contact_migration_increment = np.tanh(
                direction * (support_x - self._tail_previous_support_x)
                / np.maximum(np.float32(0.002) * body_length, np.float32(1e-5))
            )
        self._tail_previous_support_x = support_x.copy()

        rolling_metrics = self._compute_rolling_metrics()
        rolling_gate = loop_gate * self._smooth_step(
            ((-direction * rolling_metrics["body_omega"]) - np.float32(0.005)) / np.float32(0.005)
        )

        phi0 = (
            np.float32(0.50) * tail_lift_score
            + np.float32(0.35) * tail_forward_score
            + np.float32(0.15) * head_contact_score
            - np.float32(0.20) * head_lift_score
        )
        phi1 = (
            np.float32(0.20) * phi0
            + np.float32(0.45) * curl_prefix_progress
            + np.float32(0.20) * turn_progress
            + np.float32(0.15) * head_contact_score
            - np.float32(0.20) * curl_order_penalty
        )
        phi2 = (
            np.float32(0.10) * phi0
            + np.float32(0.15) * phi1
            + np.float32(0.40) * loop_gate
            + np.float32(0.20) * pivot_gate
            + np.float32(0.15) * rotation_start_progress
        )
        forward_progress = np.clip(forward_displacement / np.float32(0.5), 0.0, 1.0)
        phi3 = (
            np.float32(0.05) * phi0
            + np.float32(0.10) * phi1
            + np.float32(0.15) * phi2
            + np.float32(0.50) * rotation_turn_progress
            + np.float32(0.20) * forward_progress
        )
        tail_stage_potentials = np.concatenate((phi0, phi1, phi2, phi3), axis=1)

        stage_successes = np.concatenate(
            (
                (tail_lift_score > np.float32(0.55))
                & (tail_forward_score > np.float32(0.40))
                & (head_contact_score > np.float32(0.70)),
                (curl_prefix_progress > np.float32(0.55))
                & (closure_ratio < np.float32(0.55))
                & (total_signed_curvature > np.float32(np.pi)),
                (loop_gate > np.float32(0.50))
                & (rotation_start_progress > np.float32(0.50))
                & (support_margin > np.float32(-0.10)),
                (desired_rotation >= np.float32(2.0 * np.pi))
                & (forward_displacement >= np.float32(0.50)),
            ),
            axis=1,
        )
        current_success = stage_successes[:, self.tail_roll_stage : self.tail_roll_stage + 1]

        self._tail_roll_metrics_cache = {
            "tail_lift_score": np.asarray(tail_lift_score, dtype=np.float32),
            "tail_forward_score": np.asarray(tail_forward_score, dtype=np.float32),
            "head_contact_score": np.asarray(head_contact_score, dtype=np.float32),
            "curl_prefix_progress": np.asarray(curl_prefix_progress, dtype=np.float32),
            "curl_order_penalty": np.asarray(curl_order_penalty, dtype=np.float32),
            "total_signed_curvature": np.asarray(total_signed_curvature, dtype=np.float32),
            "closure_ratio": np.asarray(closure_ratio, dtype=np.float32),
            "closure_score": np.asarray(closure_score, dtype=np.float32),
            "support_margin": np.asarray(support_margin, dtype=np.float32),
            "rotation_increment": np.asarray(rotation_increment, dtype=np.float32),
            "desired_rotation_increment": np.asarray(desired_rotation_increment, dtype=np.float32),
            "cumulative_rotation": np.asarray(self._tail_cumulative_rotation, dtype=np.float32),
            "forward_displacement": np.asarray(forward_displacement, dtype=np.float32),
            "forward_displacement_increment": np.asarray(forward_displacement_increment, dtype=np.float32),
            "contact_migration_increment": np.asarray(contact_migration_increment, dtype=np.float32),
            "loop_gate": np.asarray(loop_gate, dtype=np.float32),
            "rolling_gate": np.asarray(rolling_gate, dtype=np.float32),
            "tail_stage_potentials": np.asarray(tail_stage_potentials, dtype=np.float32),
            "tail_stage_success": np.asarray(current_success, dtype=np.float32),
            "tail_roll_stage": np.full((self.num_envs, 1), self.tail_roll_stage, dtype=np.float32),
            "effort_penalty": rolling_metrics["effort_penalty"],
            "action_smoothness_penalty": rolling_metrics["action_smoothness_penalty"],
            "tail_roll_reward": np.zeros((self.num_envs, 1), dtype=np.float32),
        }
        return self._tail_roll_metrics_cache

    def _get_obs(self):
        pos = np.ascontiguousarray(self.pos, dtype=np.complex64)
        thdot = np.ascontiguousarray(self.thdot, dtype=np.float32)
        local_obs = self._obs_func(pos, thdot)
        if self.control_mode == "tail_wave":
            local_obs = local_obs.reshape(self.num_envs, 1, -1)
        if not self.rolling_observation and not self.tail_roll_observation:
            return local_obs

        feature_blocks = [local_obs]
        if self.rolling_observation:
            metrics = self._compute_rolling_metrics()
            speed_feature = np.tanh(
                np.asarray(self.mean_speed, dtype=np.float32) * np.float32(100.0) / self.rolling_speed_ref_x100
            )
            omega_feature = np.tanh(metrics["body_omega"] / self.rolling_omega_ref)
            global_features = np.concatenate(
                (
                    speed_feature,
                    omega_feature,
                    metrics["closure_score"],
                    metrics["circularity_score"],
                ),
                axis=1,
            )
            global_features = np.repeat(global_features[:, None, :], self.num_agents, axis=1)
            phase = np.float32(2.0 * np.pi) * (np.arange(self.num_agents, dtype=np.float32) + np.float32(0.5)) / np.float32(self.num_agents)
            phase_features = np.stack((np.sin(phase), np.cos(phase)), axis=1)
            phase_features = np.repeat(phase_features[None, :, :], self.num_envs, axis=0)
            feature_blocks.extend((global_features, phase_features))

        if self.tail_roll_observation:
            tail_metrics = self._compute_tail_roll_metrics()
            center = np.mean(pos, axis=1, keepdims=True)
            body_length = np.maximum(
                np.sum(np.abs(np.diff(pos, axis=1)), axis=1, keepdims=True),
                np.float32(1e-6),
            )
            tail = pos[:, self.tail_index : self.tail_index + 1]
            if self.head_index == self.num_particles - 1:
                head = pos[:, -1:]
            else:
                head = pos[:, :1]
            rolling_metrics = self._compute_rolling_metrics()
            cumulative_rotation = tail_metrics["cumulative_rotation"]
            stage_one_hot = np.zeros((self.num_envs, 4), dtype=np.float32)
            stage_one_hot[:, self.tail_roll_stage] = np.float32(1.0)
            tail_global = np.concatenate(
                (
                    np.real(tail - center) / body_length,
                    np.imag(tail - center) / body_length,
                    np.real(head - center) / body_length,
                    np.imag(head - center) / body_length,
                    tail_metrics["tail_lift_score"],
                    tail_metrics["tail_forward_score"],
                    tail_metrics["head_contact_score"],
                    tail_metrics["curl_prefix_progress"],
                    tail_metrics["closure_score"],
                    np.tanh(tail_metrics["total_signed_curvature"] / np.float32(2.0 * np.pi)),
                    np.tanh(tail_metrics["support_margin"] / np.float32(0.10)),
                    np.tanh(rolling_metrics["body_omega"] / self.rolling_omega_ref),
                    np.sin(cumulative_rotation),
                    np.cos(cumulative_rotation),
                    stage_one_hot,
                ),
                axis=1,
            )
            tail_global = np.repeat(tail_global[:, None, :], self.num_agents, axis=1)
            if self.tail_side == "left":
                distance_from_tail = np.arange(self.num_agents, dtype=np.float32)
            else:
                distance_from_tail = np.arange(self.num_agents - 1, -1, -1, dtype=np.float32)
            distance_from_tail /= np.float32(max(1, self.num_agents - 1))
            distance_from_tail = np.repeat(
                distance_from_tail[None, :, None], self.num_envs, axis=0
            )
            feature_blocks.extend((tail_global, distance_from_tail))

        return np.ascontiguousarray(np.concatenate(feature_blocks, axis=2), dtype=np.float32)

    def _get_info(self):
        values = {
            "speed": torch.as_tensor(
                np.real(self.mean_speed),
                dtype=torch.float32,
                device=self.device,
            )
        }
        if self.rolling_metrics_enabled:
            metrics = self._compute_rolling_metrics()
            for metric_name in (
                "closure_score",
                "circularity_score",
                "body_omega",
                "slip_penalty",
                "action_smoothness_penalty",
                "curriculum_progress",
                "rolling_reward",
            ):
                values[metric_name] = torch.as_tensor(metrics[metric_name], dtype=torch.float32, device=self.device)
        if self.tail_roll_metrics_enabled:
            tail_metrics = self._compute_tail_roll_metrics()
            for metric_name in (
                "tail_lift_score",
                "tail_forward_score",
                "head_contact_score",
                "curl_prefix_progress",
                "curl_order_penalty",
                "total_signed_curvature",
                "closure_ratio",
                "support_margin",
                "cumulative_rotation",
                "rolling_gate",
                "tail_roll_stage",
                "tail_stage_success",
                "tail_roll_reward",
            ):
                values[metric_name] = torch.as_tensor(
                    tail_metrics[metric_name], dtype=torch.float32, device=self.device
                )
        if self.control_mode == "tail_wave":
            wave_action = np.asarray(self._last_action_for_reward, dtype=np.float32)
            for index, metric_name in enumerate(
                ("wave_amplitude", "wave_center", "wave_width", "wave_hold", "wave_kp", "wave_kd")
            ):
                values[metric_name] = torch.as_tensor(
                    wave_action[:, 0, index:index + 1], dtype=torch.float32, device=self.device
                )
        return TensorDict(
            values,
            batch_size=self.observation_spec['log_info'].shape,
            device=self.device
        )

    def _reset(self, tensordict):
        pos_randomness = self.init_pos_randomness
        if self.material_shape == "ring":
            angles = np.arange(self.num_particles)*2*np.pi/self.num_particles
            radius = (self.edge_length / 2) / np.cos(((self.num_particles - 2)*np.pi)/(self.num_particles*2))
            pos = (radius * (np.cos(angles) + 1j * np.sin(angles))).astype(np.complex64) + (radius + self.particle_radius) * 1j
        else:
            pos = ((np.arange(self.num_particles) - self.num_particles/2) * self.edge_length).astype(np.complex64) + self.particle_radius * 1j
            self.tail_roll_init_assist_fraction = np.float32(0.0)
            if self.tail_roll_stage == 0 and self.tail_roll_init_assist_radians > 0:
                if self.tail_roll_init_assist_episodes > 0:
                    fraction = np.clip(
                        1.0 - self.curriculum_episode / self.tail_roll_init_assist_episodes,
                        0.0,
                        1.0,
                    )
                else:
                    fraction = 1.0
                self.tail_roll_init_assist_fraction = np.float32(fraction)
                assist_angle = self.tail_roll_init_assist_radians * self.tail_roll_init_assist_fraction
                segment_profile = np.zeros(self.num_particles - 1, dtype=np.float32)
                active_segments = self.tail_roll_init_assist_segments
                segment_profile[:active_segments] = np.linspace(
                    assist_angle,
                    np.float32(0.0),
                    active_segments,
                    dtype=np.float32,
                )
                baseline_angle = np.float32(0.0 if self.tail_side == "left" else np.pi)
                segment_angles = baseline_angle - self.tail_curl_sign * segment_profile
                ordered = np.zeros(self.num_particles, dtype=np.complex64)
                ordered[1:] = np.cumsum(
                    self.edge_length * np.exp(1j * segment_angles),
                    dtype=np.complex64,
                )
                assisted = ordered if self.tail_side == "left" else ordered[::-1]
                assisted = assisted.astype(np.complex64, copy=False)
                assisted += np.float32(np.real(np.mean(pos)) - np.real(np.mean(assisted)))
                assisted += 1j * np.float32(self.particle_radius - np.imag(assisted[self.head_index]))
                pos = assisted
        pos = np.tile(pos, (self.num_envs, 1))
        if self.init_angle_range_radians > 0:
            center = np.mean(pos, axis=1, keepdims=True)
            angles = np.random.uniform(-self.init_angle_range_radians, self.init_angle_range_radians, size=(self.num_envs, 1))
            pos = (pos - center) * np.exp(1j * angles) + center
        pos += (np.random.randn(*pos.shape) * pos_randomness + np.random.randn(*pos.shape) * pos_randomness * 1j)
        pos += 2j
        if self.init_height_jitter > 0:
            pos += 1j * np.random.uniform(0.0, self.init_height_jitter, size=(self.num_envs, 1))
        self.pos = np.ascontiguousarray(pos, dtype=np.complex64)
        self.vel = np.zeros_like(self.pos, dtype=np.complex64)
        self.mean_speed = np.zeros((self.num_envs, 1), dtype=np.float32)
        self.thdot = np.zeros(self.pos.shape, dtype=np.float32)
        self.steps = 0
        self._rolling_metrics_cache = None
        self._tail_roll_metrics_cache = None
        self._tail_previous_potential = None
        self._tail_stage_success_latched[...] = False
        center = np.mean(self.pos, axis=1, keepdims=True)
        self._tail_previous_centered_shape = (self.pos - center).copy()
        self._tail_cumulative_rotation[...] = 0.0
        self._tail_previous_support_x = None
        self._tail_initial_com_x = np.real(center).copy()
        tail = self.pos[:, self.tail_index : self.tail_index + 1]
        self._tail_initial_relative_x = np.real(tail) - self._tail_initial_com_x
        self._last_action_for_reward = np.zeros_like(self._last_action_for_reward, dtype=np.float32)
        self._previous_action_for_reward = np.zeros_like(self._previous_action_for_reward, dtype=np.float32)

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": torch.as_tensor(self._get_obs(), dtype=torch.float32, device=self.device)
                    },
                    batch_size=self.observation_spec['agents'].shape,
                    device=self.device
                ),
                "log_info": self._get_info()
            },
            batch_size=self.observation_spec.shape,
            device=self.device
        )

    def _set_seed(self, seed):
        pass

    def _step(self, tensordict):
        action = tensordict["agents", "action"]
        last_pos = np.copy(self.pos)
        if self.control_mode == "formula":
            action_np = np.ascontiguousarray(action.detach().cpu().numpy(), dtype=np.float32)
            if action_np.ndim == 2:
                action_np = np.ascontiguousarray(action_np[:, :, None], dtype=np.float32)
        elif self.action_size > 1:
            action_np = np.ascontiguousarray(action.detach().cpu().numpy(), dtype=np.float32)
        else:
            action_np = np.ascontiguousarray(action.squeeze(-1).detach().cpu().numpy(), dtype=np.float32)
        self._previous_action_for_reward = np.asarray(self._last_action_for_reward, dtype=np.float32).copy()
        self._last_action_for_reward = np.asarray(action_np, dtype=np.float32)
        pos = np.ascontiguousarray(self.pos, dtype=np.complex64)
        vel = np.ascontiguousarray(self.vel, dtype=np.complex64)
        self.pos, self.vel, self.thdot = self._physics_sim(action_np, pos, vel)
        self.pos = np.ascontiguousarray(self.pos, dtype=np.complex64)
        self.vel = np.ascontiguousarray(self.vel, dtype=np.complex64)
        self.thdot = np.ascontiguousarray(self.thdot, dtype=np.float32)
        self.mean_speed = np.asarray(
            np.real(np.mean((self.pos - last_pos), axis=1, keepdims=True)),
            dtype=np.float32
        )
        self._rolling_metrics_cache = None
        self._tail_roll_metrics_cache = None
        self.steps += 1
        done = self.steps >= self.max_steps
        if self._render_flag:
            self.render()

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": torch.as_tensor(self._get_obs(), dtype=torch.float32, device=self.device)
                    },
                    batch_size=self.observation_spec['agents'].shape,
                    device=self.device
                ),
                "reward": self._reward_func(),
                "done": torch.full(self.done_spec.shape, fill_value=done, device=self.device),
                "log_info": self._get_info()
            },
            batch_size=self.observation_spec.shape,
            device=self.device
        )
    
    def _init_renderer(self):
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.render_scale = 15
        if self.render_mode == "human":
            self.render_surface = pygame.display.set_mode((self.window_width, self.window_height))
            self.render_clock = pygame.time.Clock()
        self._renderer_inited = True
        if self._render_env_colors is None:
            self._render_env_colors = [tuple(hsv_to_rgb((hue, 1, 1))*255) for hue in np.linspace(0,1,self.num_envs+1)[:-1]]
    
    def _camera_matrix(self):
        return np.array([
            [self.render_scale, 0, self.render_scale * 10],
            [0, -self.render_scale, self.window_height - self.render_scale],
            [0, 0, 1]
        ])
    
    def _complex_to_screenspace(self, points):
        homogenous_coords = np.stack((np.real(points), np.imag(points), np.ones(points.shape)), axis=1)
        transformed_coords = self._camera_matrix() @ homogenous_coords.T
        transformed_coords = transformed_coords[:2,:] / transformed_coords[-1,:]
        return transformed_coords.T

    def _worldspace_to_screenspace(self, points):
        homogenous_coords = np.stack((points[:,0], points[:,1], np.ones(points[:,0].shape)), axis=1)
        transformed_coords = self._camera_matrix() @ homogenous_coords.T
        transformed_coords = transformed_coords[:2,:] / transformed_coords[-1,:]
        return transformed_coords.T
    
    def _screenspace_to_worldspace(self, points):
        homogenous_coords = np.stack((points[:,0], points[:,1], np.ones(points[:,0].shape)), axis=1)
        transformed_coords = np.linalg.inv(self._camera_matrix()) @ homogenous_coords.T
        transformed_coords = transformed_coords[:2,:] / transformed_coords[-1,:]
        return transformed_coords.T
    
    def render(self):
        # print("doing a lot of work here")
        if not self._renderer_inited:
            self._init_renderer()
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        screenspace_borders = np.array([[0, 0],[self.window_width, self.window_height]])
        worldspace_borders = self._screenspace_to_worldspace(screenspace_borders)
        origin_screenspace = self._worldspace_to_screenspace(np.array([[0,0]]))[0]

        # gridlines
        grid_line_spacing = 5
        grid_line_borders = np.fix(worldspace_borders).astype(int) // grid_line_spacing
        grid_line_colour = (200, 200, 200)
        grid_points_x = np.arange(np.min(grid_line_borders[:,0]),np.max(grid_line_borders[:,0]) + 1) * grid_line_spacing
        grid_points_y = np.arange(np.min(grid_line_borders[:,1]),np.max(grid_line_borders[:,1]) + 1) * grid_line_spacing
        for x, _ in self._worldspace_to_screenspace(np.stack((grid_points_x, np.zeros_like(grid_points_x)), axis=1)):
            pygame.draw.line(canvas, grid_line_colour, (x, screenspace_borders[0,1]), (x, screenspace_borders[1,1]), width=1)
        for _, y in self._worldspace_to_screenspace(np.stack((np.zeros_like(grid_points_y), grid_points_y), axis=1)):
            pygame.draw.line(canvas, grid_line_colour, (screenspace_borders[0,0], y), (screenspace_borders[1,0], y), width=1)
        
        self._render_terrain(**locals())

        for i in range(self.num_envs):
            color = self._render_env_colors[i]
            points = self._complex_to_screenspace(self.pos[i])
            prev = points[-1]
            draw_line = self.material_shape == "ring"
            for x, y in points:
                radius_outline = int(np.asarray(self.particle_radius * self.render_scale + 1).reshape(-1)[0])
                pygame.draw.circle(canvas, (0,0,0), (int(x), int(y)), radius_outline)
                if draw_line:
                    pygame.draw.line(canvas, (0,0,0), (x, y), prev, width=int(np.round(self.particle_radius * self.render_scale)) + 2)
                prev = (x, y)
                draw_line = True
            prev = points[-1]
            draw_line = self.material_shape == "ring"
            for x, y in points:
                radius_fill = int(np.asarray(self.particle_radius * self.render_scale).reshape(-1)[0])
                pygame.draw.circle(canvas, color, (int(x), int(y)), radius_fill)
                if draw_line:
                    pygame.draw.line(canvas, color, (x, y), prev, width=int(np.round(self.particle_radius * self.render_scale)))
                prev = (x, y)
                draw_line = True
        
        self._render_text(canvas, self.render_text_lines + ([f"fps: {int(self.render_clock.get_fps())}"] if self.render_mode == "human" else []))
        # if self.render_mode == "human":
        #     # debug text
        #     canvas.blit(self.font.render(f"fps: {int(self.render_clock.get_fps())}", True, 0), dest=(4,4))


        if self.render_mode == "human":
            self.render_surface.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.render_clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def render_rainbow(self):
        # print("doing a lot of work here")
        if not self._renderer_inited:
            self._init_renderer()
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        screenspace_borders = np.array([[0, 0],[self.window_width, self.window_height]])
        worldspace_borders = self._screenspace_to_worldspace(screenspace_borders)
        origin_screenspace = self._worldspace_to_screenspace(np.array([[0,0]]))[0]

        # gridlines
        grid_line_spacing = 5
        grid_line_borders = np.fix(worldspace_borders).astype(int) // grid_line_spacing
        grid_line_colour = (200, 200, 200)
        grid_points_x = np.arange(np.min(grid_line_borders[:,0]),np.max(grid_line_borders[:,0]) + 1) * grid_line_spacing
        grid_points_y = np.arange(np.min(grid_line_borders[:,1]),np.max(grid_line_borders[:,1]) + 1) * grid_line_spacing
        for x, _ in self._worldspace_to_screenspace(np.stack((grid_points_x, np.zeros_like(grid_points_x)), axis=1)):
            pygame.draw.line(canvas, grid_line_colour, (x, screenspace_borders[0,1]), (x, screenspace_borders[1,1]), width=1)
        for _, y in self._worldspace_to_screenspace(np.stack((np.zeros_like(grid_points_y), grid_points_y), axis=1)):
            pygame.draw.line(canvas, grid_line_colour, (screenspace_borders[0,0], y), (screenspace_borders[1,0], y), width=1)
        
        self._render_terrain(**locals())

        colors = [tuple(hsv_to_rgb((hue, 1, 1))*255) for hue in np.linspace(0,1,len(self.pos[0])+1)[:-1]]
        for i in range(self.num_envs):
            points = self._complex_to_screenspace(self.pos[i])
            prev = points[-1]
            draw_line = self.material_shape == "ring"
            for x, y in points:
                radius_outline = int(np.asarray(self.particle_radius * self.render_scale + 1).reshape(-1)[0])
                pygame.draw.circle(canvas, (0,0,0), (int(x), int(y)), radius_outline)
                if draw_line:
                    pygame.draw.line(canvas, (0,0,0), (x, y), prev, width=int(np.round(self.particle_radius * self.render_scale)) + 2)
                prev = (x, y)
                draw_line = True
            prev = points[-1]
            draw_line = self.material_shape == "ring"
            for c, (x, y) in enumerate(points):
                radius_fill = int(np.asarray(self.particle_radius * self.render_scale).reshape(-1)[0])
                pygame.draw.circle(canvas, colors[c], (int(x), int(y)), radius_fill)
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def _render_text(self, canvas, lines):
        if len(lines) == 0:
            return
        for i in range(len(lines)):
            canvas.blit(self.font.render(lines[i], True, 0), dest=(4,4 + i*18))
    
    def _render_terrain_flat(this, **local_vars):
        pygame.draw.line(local_vars['canvas'], 0, (0, local_vars['origin_screenspace'][1]), (this.window_width, local_vars['origin_screenspace'][1]), width=2)

    def _render_terrain_mesh(this, **local_vars):
        pointsa, pointsb = this._complex_to_screenspace(this.terrain_mesh[:,0]), this._complex_to_screenspace(this.terrain_mesh[:,1])
        for pointa, pointb in zip(pointsa, pointsb):
            pygame.draw.line(local_vars['canvas'], 0, pointa, pointb, width=2)
    
    def _render_terrain_mesh_cycle(this, **local_vars):
        meshes = len(this.terrain_mesh)
        colors = [(255,0,0,255),(0,128,0,255),(0,0,255,255)]
        i = 0
        for mesh in this.terrain_mesh:
            line_surface = pygame.Surface(local_vars['canvas'].get_size(), pygame.SRCALPHA)
            pointsa, pointsb = this._complex_to_screenspace(mesh[:,0]), this._complex_to_screenspace(mesh[:,1])
            for pointa, pointb in zip(pointsa, pointsb):
                pygame.draw.line(line_surface, colors[i%3], pointa, pointb, width=2)
            line_surface.set_alpha(256//meshes)
            local_vars['canvas'].blit(line_surface, (0, 0))
            i += 1




if __name__ == '__main__':
    e = env()
