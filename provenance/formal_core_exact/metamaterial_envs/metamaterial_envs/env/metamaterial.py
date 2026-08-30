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


def _blend_scratch_wr_torque(wave_torque, residual_torque, alpha, max_torque):
    """Blend Scratch-WR branches and clip only at their common actuator layer."""
    alpha = np.float32(alpha)
    max_torque = np.float32(max_torque)
    return np.asarray(
        np.clip(wave_torque + alpha * residual_torque, -max_torque, max_torque),
        dtype=np.float32,
    )


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


TERRAIN_SURFACE_NONE = np.int8(0)
TERRAIN_SURFACE_FLOOR = np.int8(1)
TERRAIN_SURFACE_WALL = np.int8(2)
TERRAIN_SURFACE_CEILING = np.int8(3)
TERRAIN_CONTACT_DIAGNOSTIC_FIELDS = (
    "terrain_floor_contact_strength",
    "terrain_wall_contact_strength",
    "terrain_ceiling_contact_strength",
    "terrain_floor_clearance",
    "terrain_wall_clearance",
    "terrain_ceiling_clearance",
    "terrain_floor_support_index",
    "terrain_floor_contact_count",
    "terrain_floor_normal_x",
    "terrain_floor_normal_y",
    "terrain_nearest_clearance",
    "terrain_nearest_surface_kind",
)


def _prepare_terrain_contact_geometry(segments):
    """Validate terrain segments and attach stable free-space surface normals.

    The mesh presets are open polylines rather than closed polygons, so segment
    orientation alone cannot distinguish a tunnel floor from its ceiling.  For
    every non-vertical segment we compare its midpoint with the other surfaces
    spanning the same x coordinate.  The lower envelope receives an upward
    free-space normal (floor), the upper envelope receives a downward normal
    (ceiling), and near-vertical segments are walls.
    """

    raw = np.asarray(segments, dtype=np.complex64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("Terrain contact geometry must have shape (segments, 2).")

    starts = raw[:, 0]
    ends = raw[:, 1]
    deltas = ends - starts
    lengths = np.abs(deltas).astype(np.float32)
    finite = (
        np.isfinite(np.real(starts))
        & np.isfinite(np.imag(starts))
        & np.isfinite(np.real(ends))
        & np.isfinite(np.imag(ends))
    )
    valid = finite & (lengths > np.float32(1e-7))
    if not np.any(valid):
        raise ValueError("Terrain contact geometry has no finite, non-degenerate segments.")

    starts = np.ascontiguousarray(starts[valid], dtype=np.complex64)
    ends = np.ascontiguousarray(ends[valid], dtype=np.complex64)
    deltas = np.ascontiguousarray(ends - starts, dtype=np.complex64)
    lengths = np.ascontiguousarray(np.abs(deltas), dtype=np.float32)
    length_squared = np.maximum(
        np.real(deltas * np.conj(deltas)).astype(np.float32),
        np.float32(1e-12),
    )
    tangents = np.asarray(deltas / lengths, dtype=np.complex64)
    normals = np.asarray(np.complex64(1j) * tangents, dtype=np.complex64)
    surface_kind = np.full(starts.shape, TERRAIN_SURFACE_FLOOR, dtype=np.int8)

    dx = np.real(deltas).astype(np.float32)
    dy = np.imag(deltas).astype(np.float32)
    wall_mask = np.abs(dx) <= np.float32(1e-5) * lengths
    surface_kind[wall_mask] = TERRAIN_SURFACE_WALL

    nonwall_indices = np.flatnonzero(~wall_mask)
    span_low = np.minimum(np.real(starts), np.real(ends)).astype(np.float32)
    span_high = np.maximum(np.real(starts), np.real(ends)).astype(np.float32)
    midpoint = np.asarray(
        np.float32(0.5) * (starts + ends), dtype=np.complex64
    )
    y_scale = max(
        1.0,
        float(np.max(np.abs(np.imag(raw)))) if raw.size else 1.0,
    )
    envelope_tolerance = np.float32(1e-5 * y_scale + 1e-6)

    for index in nonwall_indices.tolist():
        x_mid = np.float32(np.real(midpoint[index]))
        spans_midpoint = (
            (~wall_mask)
            & (span_low - np.float32(1e-5) <= x_mid)
            & (span_high + np.float32(1e-5) >= x_mid)
        )
        candidates = np.flatnonzero(spans_midpoint)
        if candidates.size <= 1:
            is_ceiling = False
        else:
            candidate_dx = dx[candidates]
            safe_dx = np.where(
                np.abs(candidate_dx) > np.float32(1e-7),
                candidate_dx,
                np.float32(1.0),
            )
            interpolation = np.clip(
                (x_mid - np.real(starts[candidates]).astype(np.float32)) / safe_dx,
                np.float32(0.0),
                np.float32(1.0),
            )
            y_at_midpoint = (
                np.imag(starts[candidates]).astype(np.float32)
                + interpolation * dy[candidates]
            )
            lower = np.min(y_at_midpoint)
            upper = np.max(y_at_midpoint)
            own_y = np.float32(np.imag(midpoint[index]))
            is_ceiling = bool(
                upper - lower > envelope_tolerance
                and own_y > np.float32(0.5) * (lower + upper)
            )

        if is_ceiling:
            surface_kind[index] = TERRAIN_SURFACE_CEILING
            if np.imag(normals[index]) > 0:
                normals[index] = -normals[index]
        else:
            surface_kind[index] = TERRAIN_SURFACE_FLOOR
            if np.imag(normals[index]) < 0:
                normals[index] = -normals[index]

    return {
        "starts": starts,
        "deltas": deltas,
        "length_squared": np.ascontiguousarray(length_squared, dtype=np.float32),
        "normals": np.ascontiguousarray(normals, dtype=np.complex64),
        "surface_kind": np.ascontiguousarray(surface_kind, dtype=np.int8),
    }


def _query_terrain_contact_geometry(positions, geometry, particle_radius):
    """Return vectorized nearest-point/contact data for every particle."""

    pos = np.asarray(positions, dtype=np.complex64)
    if pos.ndim != 2:
        raise ValueError("Terrain contact positions must have shape (envs, particles).")

    starts = geometry["starts"]
    deltas = geometry["deltas"]
    length_squared = geometry["length_squared"]
    points = pos[:, :, None]
    relative = points - starts[None, None, :]
    projection = np.real(
        relative * np.conj(deltas)[None, None, :]
    ).astype(np.float32)
    projection /= length_squared[None, None, :]
    projection = np.clip(projection, np.float32(0.0), np.float32(1.0))
    closest_points = (
        starts[None, None, :]
        + projection.astype(np.complex64) * deltas[None, None, :]
    )
    distances = np.asarray(np.abs(points - closest_points), dtype=np.float32)

    def nearest_for_kind(kind):
        mask = geometry["surface_kind"] == np.int8(kind)
        shape = pos.shape
        if not np.any(mask):
            return {
                "index": np.full(shape, -1, dtype=np.int32),
                "distance": np.full(shape, np.inf, dtype=np.float32),
                "clearance": np.full(shape, np.inf, dtype=np.float32),
                "point": np.zeros(shape, dtype=np.complex64),
                "normal": np.zeros(shape, dtype=np.complex64),
            }
        candidate_indices = np.flatnonzero(mask)
        candidate_distances = distances[:, :, candidate_indices]
        local_index = np.argmin(candidate_distances, axis=2)
        segment_index = candidate_indices[local_index]
        gather_index = local_index[:, :, None]
        distance = np.take_along_axis(
            candidate_distances, gather_index, axis=2
        )[:, :, 0].astype(np.float32)
        candidate_points = closest_points[:, :, candidate_indices]
        point = np.take_along_axis(
            candidate_points, gather_index, axis=2
        )[:, :, 0].astype(np.complex64)
        normal = geometry["normals"][segment_index].astype(np.complex64)
        return {
            "index": segment_index.astype(np.int32),
            "distance": distance,
            "clearance": np.asarray(
                distance - np.float32(particle_radius), dtype=np.float32
            ),
            "point": point,
            "normal": normal,
        }

    nearest_index = np.argmin(distances, axis=2)
    gather_index = nearest_index[:, :, None]
    nearest_distance = np.take_along_axis(
        distances, gather_index, axis=2
    )[:, :, 0].astype(np.float32)
    nearest_point = np.take_along_axis(
        closest_points, gather_index, axis=2
    )[:, :, 0].astype(np.complex64)
    result = {
        "nearest_index": nearest_index.astype(np.int32),
        "nearest_distance": nearest_distance,
        "nearest_clearance": np.asarray(
            nearest_distance - np.float32(particle_radius), dtype=np.float32
        ),
        "nearest_point": nearest_point,
        "nearest_normal": geometry["normals"][nearest_index].astype(np.complex64),
        "nearest_surface_kind": geometry["surface_kind"][nearest_index].astype(np.int8),
        "floor": nearest_for_kind(TERRAIN_SURFACE_FLOOR),
        "wall": nearest_for_kind(TERRAIN_SURFACE_WALL),
        "ceiling": nearest_for_kind(TERRAIN_SURFACE_CEILING),
    }
    return result



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
def reward_func_obs2_roll_repro_v1(self):
    """Reward-only curriculum for reproducing loop rolling with local obs2.

    The actor observation and formula action are deliberately untouched.  The
    reward first pays only high-water progress towards a loop, then blends to
    contact-grounded motion that requires forward translation and desired body
    rotation at the same time.  A valid roll pulse additionally requires the
    existing rotation, translation, material-contact migration, direction and
    ground-contact gates from ``fast_forward_roll_v2``.
    """
    fast = self._compute_fast_forward_roll_v2_metrics()
    roll = self._compute_rolling_metrics()

    alpha = np.asarray(roll["curriculum_progress"], dtype=np.float32)
    closure = np.asarray(roll["closure_score"], dtype=np.float32)
    circularity = np.asarray(roll["circularity_score"], dtype=np.float32)
    # Launch is deliberately strict (0.80/0.65 below), while this maintenance
    # gate is softer because a real rolling loop opens slightly at contact.
    shape_gate = self._smooth_step((closure - np.float32(0.55)) / np.float32(0.08))
    shape_gate *= self._smooth_step((circularity - np.float32(0.50)) / np.float32(0.10))
    # A hard contact gate prevents an airborne closed loop from collecting a
    # dense rotation reward through the smooth detector's non-zero tail.
    contact_gate = (
        fast["fast_forward_ground_contact_strength"] >= np.float32(0.50)
    ).astype(np.float32)

    forward = np.maximum(roll["speed_score"], np.float32(0.0))
    rotation = np.maximum(roll["rotation_score"], np.float32(0.0))
    kinematic_gate = np.sqrt(np.maximum(forward * rotation, np.float32(0.0)))
    no_slip = np.float32(1.0) - np.clip(roll["slip_penalty"], 0.0, 1.0)

    launch_phase = (
        (fast["fast_forward_phase"] < np.float32(0.5))
        | (fast["fast_forward_launch_event"] > np.float32(0.5))
    )
    shape_progress = np.where(
        launch_phase,
        fast["fast_forward_progress_delta"],
        np.float32(0.0),
    ).astype(np.float32)
    shape_reward = (
        np.float32(5.0) * shape_progress
        + np.float32(0.10) * shape_gate * forward
        + np.float32(0.50) * fast["fast_forward_launch_event"]
    )

    rolling_reward = (
        shape_gate
        * (
            contact_gate
            * kinematic_gate
            * (
                np.float32(0.45) * rotation
                + np.float32(0.25) * forward
                + np.float32(0.15) * fast["fast_forward_progress_delta"]
                + np.float32(0.15) * no_slip
            )
            + fast["fast_forward_event_bonus"]
        )
        - np.float32(0.25) * fast["fast_forward_reverse_rotation_penalty"]
        - np.float32(0.15) * fast["fast_forward_backward_penalty"]
        - np.float32(0.002) * fast["effort_penalty"]
        - self.action_smoothness_weight * fast["action_smoothness_penalty"]
        - np.float32(0.002) * fast["fast_forward_stall_penalty"]
    )

    reward = self.rolling_reward_scale * (
        (np.float32(1.0) - alpha) * shape_reward + alpha * rolling_reward
    )
    reward = np.asarray(reward, dtype=np.float32)
    roll["rolling_reward"] = reward
    fast["fast_forward_reward"] = reward
    return torch.as_tensor(reward, dtype=torch.float32, device=self.device)


@reward_func()
def reward_func_obs2_roll_repro_v2(self):
    """Ability-gated, tail-first rolling reward for the unchanged local obs2 controller.

    Unlike v1, preparation is driven by the weakest-link synchrony of tail
    lift, tail-forward motion, ordered curl, and near-ground head contact with
    a real support provenance.  The
    transition to rolling depends on demonstrated ability for eight consecutive
    steps, never on a fixed training-batch schedule or loop closure.  The event
    detector retains the strict grounded rotation/translation/support-migration
    pulse definition used by the frozen evaluator.
    """
    fast = self._compute_fast_forward_roll_v2_metrics()
    roll = self._compute_rolling_metrics()

    launch_score = np.asarray(
        fast["fast_forward_launch_progress"], dtype=np.float32
    )
    ability_x = np.clip(
        (launch_score - np.float32(0.25)) / np.float32(0.50),
        np.float32(0.0),
        np.float32(1.0),
    ).astype(np.float32)
    ability_gate = (
        ability_x
        * ability_x
        * (np.float32(3.0) - np.float32(2.0) * ability_x)
    ).astype(np.float32)

    forward = np.maximum(roll["speed_score"], np.float32(0.0))
    rotation = np.maximum(roll["rotation_score"], np.float32(0.0))
    contact_gate = (
        fast["fast_forward_ground_contact_strength"] >= np.float32(0.50)
    ).astype(np.float32)
    direction_gate = self._smooth_step(
        (
            fast["fast_forward_episode_direction_fraction"]
            - np.float32(0.50)
        )
        / np.float32(0.08)
    )
    no_slip = np.float32(1.0) - np.clip(
        roll["slip_penalty"], np.float32(0.0), np.float32(1.0)
    )
    kinematic_gate = np.sqrt(
        np.maximum(forward * rotation, np.float32(0.0))
    )
    motion_quality = (
        contact_gate
        * kinematic_gate
        * (
            np.float32(0.45) * rotation
            + np.float32(0.25) * forward
            + np.float32(0.15) * direction_gate
            + np.float32(0.15) * no_slip
        )
    ).astype(np.float32)

    progress_age_factor = np.clip(
        np.float32(1.0)
        - fast["fast_forward_progress_age"] / np.float32(100.0),
        np.float32(0.0),
        np.float32(1.0),
    ).astype(np.float32)
    launch_phase = (
        (fast["fast_forward_phase"] < np.float32(0.5))
        | (fast["fast_forward_launch_event"] > np.float32(0.5))
    )

    preparation_reward = (
        np.float32(2.5) * fast["fast_forward_progress_delta"]
        + np.float32(0.03) * launch_score * progress_age_factor
        + np.float32(0.08)
        * launch_score
        * forward
        * contact_gate
        * (np.float32(1.0) - ability_gate)
        * progress_age_factor
        + np.float32(0.08)
        * ability_gate
        * motion_quality
        * progress_age_factor
        + np.float32(1.0) * fast["fast_forward_launch_event"]
        - np.float32(0.0002) * fast["effort_penalty"]
    )

    penalty_scale = (
        np.float32(0.20)
        + np.float32(0.80)
        * np.clip(
            fast["fast_forward_event_count"],
            np.float32(0.0),
            np.float32(1.0),
        )
    ).astype(np.float32)
    rolling_penalty = (
        np.float32(0.25) * fast["fast_forward_reverse_rotation_penalty"]
        + np.float32(0.15) * fast["fast_forward_backward_penalty"]
        + np.float32(0.002) * fast["effort_penalty"]
        + self.action_smoothness_weight * fast["action_smoothness_penalty"]
        + np.float32(0.002) * fast["fast_forward_stall_penalty"]
    )
    rolling_reward = (
        np.float32(1.5) * fast["fast_forward_progress_delta"]
        + fast["fast_forward_event_bonus"]
        + np.float32(0.016) * launch_score * forward * contact_gate
        + np.float32(0.08) * motion_quality
        - penalty_scale * rolling_penalty
    )

    reward = self.rolling_reward_scale * np.where(
        launch_phase, preparation_reward, rolling_reward
    )
    reward = np.asarray(reward, dtype=np.float32)
    if reward.shape != (self.num_envs, 1) or not np.all(np.isfinite(reward)):
        raise RuntimeError("obs2_roll_repro_v2 produced an invalid reward tensor")
    roll["rolling_reward"] = reward
    fast["fast_forward_reward"] = reward
    return torch.as_tensor(reward, dtype=torch.float32, device=self.device)


@reward_func()
def reward_func_obs2_roll_repro_v2_1(self):
    """Ability-gated, tail-first rolling reward for the unchanged local obs2 controller.

    Unlike v1, preparation is driven by the weakest-link synchrony of tail
    lift, tail-forward motion, ordered curl, and near-ground head contact with
    a real support provenance.  The
    transition to rolling depends on demonstrated ability for eight consecutive
    steps, never on a fixed training-batch schedule or loop closure.  The event
    detector retains the strict grounded rotation/translation/support-migration
    pulse definition used by the frozen evaluator.
    """
    fast = self._compute_fast_forward_roll_v2_metrics()
    roll = self._compute_rolling_metrics()

    launch_score = np.asarray(
        fast["fast_forward_launch_progress"], dtype=np.float32
    )
    ability_x = np.clip(
        (launch_score - np.float32(0.25)) / np.float32(0.50),
        np.float32(0.0),
        np.float32(1.0),
    ).astype(np.float32)
    ability_gate = (
        ability_x
        * ability_x
        * (np.float32(3.0) - np.float32(2.0) * ability_x)
    ).astype(np.float32)

    forward = np.maximum(roll["speed_score"], np.float32(0.0))
    rotation = np.maximum(roll["rotation_score"], np.float32(0.0))
    contact_gate = (
        fast["fast_forward_ground_contact_strength"] >= np.float32(0.50)
    ).astype(np.float32)
    direction_gate = self._smooth_step(
        (
            fast["fast_forward_episode_direction_fraction"]
            - np.float32(0.50)
        )
        / np.float32(0.08)
    )
    no_slip = np.float32(1.0) - np.clip(
        roll["slip_penalty"], np.float32(0.0), np.float32(1.0)
    )
    kinematic_gate = np.sqrt(
        np.maximum(forward * rotation, np.float32(0.0))
    )
    motion_quality = (
        contact_gate
        * kinematic_gate
        * (
            np.float32(0.45) * rotation
            + np.float32(0.25) * forward
            + np.float32(0.15) * direction_gate
            + np.float32(0.15) * no_slip
        )
    ).astype(np.float32)

    progress_age_factor = np.clip(
        np.float32(1.0)
        - fast["fast_forward_progress_age"] / np.float32(100.0),
        np.float32(0.0),
        np.float32(1.0),
    ).astype(np.float32)
    launch_phase = (
        (fast["fast_forward_phase"] < np.float32(0.5))
        | (fast["fast_forward_launch_event"] > np.float32(0.5))
    )

    preparation_reward = (
        np.float32(2.5) * fast["fast_forward_progress_delta"]
        + np.float32(0.03) * launch_score * progress_age_factor
        + np.float32(0.08)
        * launch_score
        * forward
        * contact_gate
        * (np.float32(1.0) - ability_gate)
        * progress_age_factor
        + np.float32(0.08)
        * ability_gate
        * motion_quality
        * progress_age_factor
        + np.float32(1.0) * fast["fast_forward_launch_event"]
        - np.float32(0.0002) * fast["effort_penalty"]
    )

    penalty_scale = (
        np.float32(0.20)
        + np.float32(0.80)
        * np.clip(
            fast["fast_forward_event_count"],
            np.float32(0.0),
            np.float32(1.0),
        )
    ).astype(np.float32)
    rolling_penalty = (
        np.float32(0.25) * fast["fast_forward_reverse_rotation_penalty"]
        + np.float32(0.15) * fast["fast_forward_backward_penalty"]
        + np.float32(0.002) * fast["effort_penalty"]
        + self.action_smoothness_weight * fast["action_smoothness_penalty"]
        + np.float32(0.002) * fast["fast_forward_stall_penalty"]
    )
    rolling_reward = (
        np.float32(1.5) * fast["fast_forward_progress_delta"]
        + fast["fast_forward_event_bonus"]
        + np.float32(0.016) * launch_score * forward * contact_gate
        + np.float32(0.16) * motion_quality
        - penalty_scale * rolling_penalty
    )

    reward = self.rolling_reward_scale * np.where(
        launch_phase, preparation_reward, rolling_reward
    )
    reward = np.asarray(reward, dtype=np.float32)
    if reward.shape != (self.num_envs, 1) or not np.all(np.isfinite(reward)):
        raise RuntimeError("obs2_roll_repro_v2_1 produced an invalid reward tensor")
    roll["rolling_reward"] = reward
    fast["fast_forward_reward"] = reward
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


@reward_func()
def reward_func_fast_rollover(self):
    """Reward repeated tail-curl -> forward-flip -> extend cycles.

    This opt-in objective deliberately does not require the crawler's ends to
    meet or a single event to complete a full revolution.  Translation is
    paid only after a tail-first curl has started and while forward rotation
    is present, which blocks the straight-body crawling shortcut.
    """
    metrics = self._compute_fast_rollover_metrics()
    potential = metrics["fast_roll_phase_potential"]
    if self._fast_roll_previous_potential is None:
        progress_reward = np.zeros_like(potential, dtype=np.float32)
    else:
        progress_reward = potential - self._fast_roll_previous_potential
    if np.any(metrics["fast_roll_phase_changed"] > np.float32(0.5)):
        # Potentials from different phases are not comparable.  The explicit
        # transition/event bonuses below reward the boundary instead.
        progress_reward = np.where(
            metrics["fast_roll_phase_changed"] > np.float32(0.5),
            np.float32(0.0),
            progress_reward,
        )
    self._fast_roll_previous_potential = np.asarray(potential, dtype=np.float32).copy()

    desired_increment = metrics["desired_rotation_increment"]
    forward_increment = metrics["forward_displacement_increment"]
    contact_increment = metrics["contact_migration_increment"]
    motion_gate = metrics["fast_roll_motion_gate"]
    reverse_penalty = np.maximum(-desired_increment, np.float32(0.0))
    transition_bonus = np.float32(0.5) * metrics["fast_roll_curl_transition"]
    event_bonus = metrics["fast_roll_event_bonus"]

    reward = (
        np.float32(2.0) * progress_reward
        + np.float32(0.35) * motion_gate * np.maximum(desired_increment, np.float32(0.0))
        + np.float32(0.30) * motion_gate * np.maximum(forward_increment, np.float32(0.0))
        + np.float32(0.20) * motion_gate * np.maximum(contact_increment, np.float32(0.0))
        + transition_bonus
        + event_bonus
        - np.float32(0.10) * reverse_penalty
        - np.float32(0.002) * metrics["effort_penalty"]
        - self.action_smoothness_weight * metrics["action_smoothness_penalty"]
    )
    reward = np.asarray(self.fast_rollover_reward_scale * reward, dtype=np.float32)
    metrics["fast_roll_reward"] = reward
    return torch.as_tensor(reward, dtype=torch.float32, device=self.device)


@reward_func()
def reward_func_fast_forward_roll_v2(self):
    """Tail-launch followed by repeated, closure-free forward-roll pulses.

    The legacy rolling rewards above are deliberately left untouched.  This
    opt-in objective has only two phases: acquire a tail-first launch posture,
    then keep rolling.  A new pulse is emitted whenever rotation, translation,
    and *material* contact migration have all advanced far enough from the
    previous event anchor.  Neither endpoint closure nor a 2*pi revolution is
    part of the objective.

    Progress rewards use per-phase high-water marks.  Returning to an old pose
    therefore cannot create reward by oscillating across the same interval.
    Reverse rotation and backward translation are penalised from raw signed
    increments rather than from an asymmetric clipped surrogate.
    """
    metrics = self._compute_fast_forward_roll_v2_metrics()
    reverse_penalty = metrics["fast_forward_reverse_rotation_penalty"]
    backward_penalty = metrics["fast_forward_backward_penalty"]

    reward = (
        np.float32(2.0) * metrics["fast_forward_progress_delta"]
        + np.float32(0.5) * metrics["fast_forward_launch_event"]
        + metrics["fast_forward_event_bonus"]
        - np.float32(0.25) * reverse_penalty
        - np.float32(0.15) * backward_penalty
        - np.float32(0.002) * metrics["effort_penalty"]
        - self.action_smoothness_weight * metrics["action_smoothness_penalty"]
        - np.float32(0.002) * metrics["fast_forward_stall_penalty"]
    )
    reward = np.asarray(self.fast_forward_reward_scale * reward, dtype=np.float32)
    metrics["fast_forward_reward"] = reward
    return torch.as_tensor(reward, dtype=torch.float32, device=self.device)


@reward_func()
def reward_func_scratch_wr_fast_forward_v2(self):
    """Scratch-WR-v2 shaping without changing the legacy fast-forward reward.

    During the alpha=0 launch-acquisition stage, reward the simultaneous
    weakest-link progress of tail lift, forward curl, and ordered curvature.
    The preparation penalties are linearly restored to their legacy weights,
    so a Z0 policy must eventually work under the original objective.  After
    launch, or as soon as residual authority is enabled, this is exactly the
    legacy closure-free fast-forward objective.
    """
    metrics = self._compute_fast_forward_roll_v2_metrics()
    penalty_scale = np.where(
        metrics["scratch_wr_v2_z0_active"] > np.float32(0.5),
        metrics["scratch_wr_v2_z0_penalty_scale"],
        np.float32(1.0),
    ).astype(np.float32)
    preparation_penalty = (
        np.float32(0.25) * metrics["fast_forward_reverse_rotation_penalty"]
        + np.float32(0.15) * metrics["fast_forward_backward_penalty"]
        + np.float32(0.002) * metrics["effort_penalty"]
        + self.action_smoothness_weight * metrics["action_smoothness_penalty"]
    )
    reward = (
        np.float32(2.0) * metrics["scratch_wr_v2_progress_delta"]
        + np.float32(0.5) * metrics["fast_forward_launch_event"]
        + metrics["fast_forward_event_bonus"]
        + metrics["scratch_wr_v2_z0_dense_reward"]
        - penalty_scale * preparation_penalty
        - np.float32(0.002) * metrics["fast_forward_stall_penalty"]
    )
    reward = np.asarray(self.fast_forward_reward_scale * reward, dtype=np.float32)
    metrics["fast_forward_reward"] = reward
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
                 tail_wave_kp_max=12.0, tail_wave_kd_max=4.0,
                 scratch_wr_alpha=0.0,
                 scratch_wr_v2=False,
                 scratch_wr_v2_sync_dense_weight=0.02,
                 scratch_wr_v2_penalty_start_scale=0.20,
                 scratch_wr_v2_penalty_anneal_batches=200,
                 scratch_wr_v2_wave_ema_beta=0.90,
                 fast_rollover_reward_scale=3.0,
                 fast_rollover_flip_degrees=60.0,
                 fast_rollover_forward_fraction=0.10,
                 fast_rollover_support_fraction=0.02,
                 fast_rollover_reset_open_ratio=0.70,
                 fast_rollover_cycle_target_steps=250,
                 fast_forward_observation=False,
                 fast_forward_reward_scale=1.0,
                 fast_forward_event_degrees=60.0,
                 fast_forward_event_forward_fraction=0.08,
                 fast_forward_event_contact_nodes=1.5,
                 fast_forward_direction_fraction=0.65,
                 fast_forward_event_target_steps=250,
                 fast_forward_launch_lift=0.20,
                 fast_forward_launch_forward=0.10,
                 fast_forward_launch_curl=0.12,
                 fast_forward_launch_head_contact=0.50,
                 fast_forward_launch_hold_steps=8,
                 fast_forward_stall_steps=150,
                 fast_forward_rotation_step_ref_degrees=2.0,
                 fast_forward_translation_step_ref=0.002,
                 particle_mass=0.2,
                 ground_stiffness=1e3,
                 ground_damping=5.0,
                 terrain_contact_mode="legacy_flat"):
        assert material_shape in ["ring", "crawler"]
        super().__init__(device="cpu", batch_size=[num_envs])
        num_agents = num_particles - 2 if material_shape == "crawler" else num_particles
        terrain_contact_mode = str(terrain_contact_mode).strip().lower()
        if terrain_contact_mode not in {"legacy_flat", "mesh_v1"}:
            raise ValueError(
                "terrain_contact_mode must be 'legacy_flat' or 'mesh_v1'."
            )
        self.terrain_contact_mode = terrain_contact_mode
        self._terrain_contact_geometry = None
        self._terrain_contact_cache = None

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
            "tail_wave_residual": "tail_wave_residual",
            "scratch_wr": "tail_wave_residual",
            "wave_residual": "tail_wave_residual",
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
                "Use 'direct', 'formula', 'tail_wave', 'tail_wave_residual', 'nonreciprocity', "
                "'fixed_k1_k2_positive', or 'fixed_k1_k2_negative'."
            )
        control_mode = control_mode_aliases[control_mode_key]
        num_controlled_joints = num_particles - 2 if material_shape == "crawler" else num_particles
        # Wave controllers are one global policy agent. Scratch-WR appends two
        # residual K coefficients for every physical joint to the six wave
        # parameters, while every legacy mode keeps its original agent layout.
        # Legacy modes retain exactly one policy agent per physical joint.
        num_agents = 1 if control_mode in {"tail_wave", "tail_wave_residual"} else num_controlled_joints
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
        particle_mass = np.float32(particle_mass) # kg
        ground_stiffness = np.float32(ground_stiffness)
        ground_damping = np.float32(ground_damping)
        if not np.isfinite(particle_mass) or particle_mass <= 0:
            raise ValueError("particle_mass must be a positive finite value.")
        if not np.isfinite(ground_stiffness) or ground_stiffness <= 0:
            raise ValueError("ground_stiffness must be a positive finite value.")
        if not np.isfinite(ground_damping) or ground_damping < 0:
            raise ValueError("ground_damping must be a non-negative finite value.")

        dt = np.float32(3e-3) #integration time step
        physics_steps_per_timestep = 10 # increase this rather than dt to prevent instabilities
        gravity_constant = np.complex64(-1j*1)
        background_friction = np.float32(0.0)
        angle_eq = np.float32(((np.pi * (num_particles - 2)) / num_particles) if material_shape == "ring" else np.pi)

        angle_stiffness = passive_kappa_value
        angle_damping = np.float32(0.42)
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
        self.particle_mass = float(particle_mass)
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
        if material_shape != "crawler" and control_mode in {"tail_wave", "tail_wave_residual"}:
            raise ValueError("tail-wave controls are defined only for the crawler material shape.")
        if not np.isfinite(float(tail_wave_amplitude_max)) or float(tail_wave_amplitude_max) <= 0:
            raise ValueError("tail_wave_amplitude_max must be positive.")
        if not np.isfinite(float(tail_wave_center_max)) or float(tail_wave_center_max) <= 0:
            raise ValueError("tail_wave_center_max must be positive.")
        if not (0 < float(tail_wave_width_min) < float(tail_wave_width_max)):
            raise ValueError("tail_wave widths must satisfy 0 < min < max.")
        if float(tail_wave_kp_max) <= 0 or float(tail_wave_kd_max) <= 0:
            raise ValueError("tail_wave gain limits must be positive.")
        if not np.isfinite(float(scratch_wr_alpha)) or not (0.0 <= float(scratch_wr_alpha) <= 1.0):
            raise ValueError("scratch_wr_alpha must be a finite value in [0, 1].")
        if not np.isfinite(float(scratch_wr_v2_sync_dense_weight)) or float(scratch_wr_v2_sync_dense_weight) < 0:
            raise ValueError("scratch_wr_v2_sync_dense_weight must be non-negative and finite.")
        if not np.isfinite(float(scratch_wr_v2_penalty_start_scale)) or not (
            0.0 < float(scratch_wr_v2_penalty_start_scale) <= 1.0
        ):
            raise ValueError("scratch_wr_v2_penalty_start_scale must be in (0, 1].")
        if int(scratch_wr_v2_penalty_anneal_batches) < 0:
            raise ValueError("scratch_wr_v2_penalty_anneal_batches must be non-negative.")
        if not np.isfinite(float(scratch_wr_v2_wave_ema_beta)) or not (
            0.0 <= float(scratch_wr_v2_wave_ema_beta) < 1.0
        ):
            raise ValueError("scratch_wr_v2_wave_ema_beta must be in [0, 1).")
        if not np.isfinite(float(fast_rollover_reward_scale)) or float(fast_rollover_reward_scale) <= 0:
            raise ValueError("fast_rollover_reward_scale must be a positive finite value.")
        if not (0.0 < float(fast_rollover_flip_degrees) < 180.0):
            raise ValueError("fast_rollover_flip_degrees must be in (0, 180).")
        if float(fast_rollover_forward_fraction) < 0 or float(fast_rollover_support_fraction) < 0:
            raise ValueError("fast rollover displacement thresholds must be non-negative.")
        if not (0.0 < float(fast_rollover_reset_open_ratio) <= 1.0):
            raise ValueError("fast_rollover_reset_open_ratio must be in (0, 1].")
        if int(fast_rollover_cycle_target_steps) <= 0:
            raise ValueError("fast_rollover_cycle_target_steps must be positive.")
        if not np.isfinite(float(fast_forward_reward_scale)) or float(fast_forward_reward_scale) <= 0:
            raise ValueError("fast_forward_reward_scale must be a positive finite value.")
        if not (0.0 < float(fast_forward_event_degrees) < 180.0):
            raise ValueError("fast_forward_event_degrees must be in (0, 180).")
        if float(fast_forward_event_forward_fraction) <= 0:
            raise ValueError("fast_forward_event_forward_fraction must be positive.")
        if float(fast_forward_event_contact_nodes) <= 0:
            raise ValueError("fast_forward_event_contact_nodes must be positive.")
        if not (0.5 <= float(fast_forward_direction_fraction) <= 1.0):
            raise ValueError("fast_forward_direction_fraction must be in [0.5, 1].")
        if (
            int(fast_forward_event_target_steps) <= 0
            or int(fast_forward_launch_hold_steps) <= 0
            or int(fast_forward_stall_steps) <= 0
        ):
            raise ValueError("fast-forward event/launch-hold/stall step counts must be positive.")
        for threshold_name, threshold_value in (
            ("fast_forward_launch_lift", fast_forward_launch_lift),
            ("fast_forward_launch_forward", fast_forward_launch_forward),
            ("fast_forward_launch_curl", fast_forward_launch_curl),
            ("fast_forward_launch_head_contact", fast_forward_launch_head_contact),
        ):
            if not (0.0 <= float(threshold_value) <= 1.0):
                raise ValueError(f"{threshold_name} must be in [0, 1].")
        if float(fast_forward_rotation_step_ref_degrees) <= 0 or float(fast_forward_translation_step_ref) <= 0:
            raise ValueError("fast-forward step reference values must be positive.")
        if bool(scratch_wr_v2) and control_mode != "tail_wave_residual":
            raise ValueError("scratch_wr_v2 requires tail_wave_residual control.")
        if bool(scratch_wr_v2) and reward_func != "scratch_wr_fast_forward_v2":
            raise ValueError("scratch_wr_v2 requires reward_func='scratch_wr_fast_forward_v2'.")
        if reward_func == "scratch_wr_fast_forward_v2" and not bool(scratch_wr_v2):
            raise ValueError("scratch_wr_fast_forward_v2 requires scratch_wr_v2=True.")
        if material_shape != "crawler" and (
            tail_roll_observation
            or fast_forward_observation
            or reward_func in {
                "tail_roll_curriculum", "fast_rollover", "fast_forward_roll_v2",
                "obs2_roll_repro_v1", "obs2_roll_repro_v2", "obs2_roll_repro_v2_1",
                "scratch_wr_fast_forward_v2",
            }
        ):
            raise ValueError("Tail-first rolling is defined only for the crawler material shape.")

        self.reward_func = reward_func
        self.rolling_observation = bool(rolling_observation)
        self.tail_roll_observation = bool(tail_roll_observation)
        self.fast_rollover_metrics_enabled = reward_func == "fast_rollover"
        self.fast_forward_observation = bool(fast_forward_observation)
        self.fast_forward_metrics_enabled = (
            reward_func in {
                "fast_forward_roll_v2", "obs2_roll_repro_v1", "obs2_roll_repro_v2", "obs2_roll_repro_v2_1",
                "scratch_wr_fast_forward_v2",
            }
            or self.fast_forward_observation
        )
        self.tail_roll_metrics_enabled = (
            self.tail_roll_observation
            or reward_func == "tail_roll_curriculum"
            or self.fast_rollover_metrics_enabled
            or self.fast_forward_metrics_enabled
        )
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
        self.fast_rollover_reward_scale = np.float32(fast_rollover_reward_scale)
        self.fast_rollover_flip_radians = np.float32(np.deg2rad(fast_rollover_flip_degrees))
        self.fast_rollover_forward_fraction = np.float32(fast_rollover_forward_fraction)
        self.fast_rollover_support_fraction = np.float32(fast_rollover_support_fraction)
        self.fast_rollover_reset_open_ratio = np.float32(fast_rollover_reset_open_ratio)
        self.fast_rollover_cycle_target_steps = int(fast_rollover_cycle_target_steps)
        self.fast_forward_reward_scale = np.float32(fast_forward_reward_scale)
        self.fast_forward_event_radians = np.float32(np.deg2rad(fast_forward_event_degrees))
        self.fast_forward_event_forward_fraction = np.float32(fast_forward_event_forward_fraction)
        self.fast_forward_event_contact_nodes = np.float32(fast_forward_event_contact_nodes)
        self.fast_forward_direction_fraction_threshold = np.float32(fast_forward_direction_fraction)
        self.fast_forward_event_target_steps = int(fast_forward_event_target_steps)
        self.fast_forward_launch_lift = np.float32(fast_forward_launch_lift)
        self.fast_forward_launch_forward = np.float32(fast_forward_launch_forward)
        self.fast_forward_launch_curl = np.float32(fast_forward_launch_curl)
        self.fast_forward_launch_head_contact = np.float32(fast_forward_launch_head_contact)
        self.fast_forward_launch_hold_steps = int(fast_forward_launch_hold_steps)
        self.fast_forward_stall_steps = int(fast_forward_stall_steps)
        self.fast_forward_rotation_step_ref = np.float32(
            np.deg2rad(fast_forward_rotation_step_ref_degrees)
        )
        self.fast_forward_translation_step_ref = np.float32(fast_forward_translation_step_ref)
        self.tail_wave_action_names = ("amplitude", "center", "width", "hold", "kp", "kd")
        self.tail_wave_action_low = np.asarray(
            [0.0, 0.0, tail_wave_width_min, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.tail_wave_action_high = np.asarray(
            [tail_wave_amplitude_max, tail_wave_center_max, tail_wave_width_max, 1.0, tail_wave_kp_max, tail_wave_kd_max],
            dtype=np.float32,
        )
        self.scratch_wr_alpha = np.float32(scratch_wr_alpha)
        self.scratch_wr_v2 = bool(scratch_wr_v2)
        self.scratch_wr_v2_sync_dense_weight = np.float32(scratch_wr_v2_sync_dense_weight)
        self.scratch_wr_v2_penalty_start_scale = np.float32(scratch_wr_v2_penalty_start_scale)
        self.scratch_wr_v2_penalty_anneal_batches = int(scratch_wr_v2_penalty_anneal_batches)
        self.scratch_wr_v2_wave_ema_beta = np.float32(scratch_wr_v2_wave_ema_beta)
        scratch_wr_k1_min = k1_min_value if k1_min_set else -max_control_gain
        scratch_wr_k1_max = k1_max_value if k1_max_set else max_control_gain
        scratch_wr_k2_min = k2_min_value if k2_min_set else -max_control_gain
        scratch_wr_k2_max = k2_max_value if k2_max_set else max_control_gain
        self.scratch_wr_k1_min = float(scratch_wr_k1_min)
        self.scratch_wr_k1_max = float(scratch_wr_k1_max)
        self.scratch_wr_k2_min = float(scratch_wr_k2_min)
        self.scratch_wr_k2_max = float(scratch_wr_k2_max)
        residual_low = np.tile(
            np.asarray([scratch_wr_k1_min, scratch_wr_k2_min], dtype=np.float32),
            num_controlled_joints,
        ) / k_action_scale_value
        residual_high = np.tile(
            np.asarray([scratch_wr_k1_max, scratch_wr_k2_max], dtype=np.float32),
            num_controlled_joints,
        ) / k_action_scale_value
        residual_names = tuple(
            name
            for joint_index in range(num_controlled_joints)
            for name in (f"residual_k1_joint_{joint_index}", f"residual_k2_joint_{joint_index}")
        )
        self.scratch_wr_action_names = self.tail_wave_action_names + residual_names
        self.scratch_wr_action_low = np.concatenate((self.tail_wave_action_low, residual_low)).astype(np.float32)
        self.scratch_wr_action_high = np.concatenate((self.tail_wave_action_high, residual_high)).astype(np.float32)
        self._scratch_wr_metrics_cache = None
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
        self._fast_rollover_metrics_cache = None
        self._fast_roll_phase = np.zeros((num_envs, 1), dtype=np.int32)
        self._fast_roll_phase_steps = np.zeros((num_envs, 1), dtype=np.int32)
        self._fast_roll_flip_count = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_roll_cycle_count = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_roll_cycle_start_rotation = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_roll_cycle_start_com_x = None
        self._fast_roll_cycle_start_support_x = None
        self._fast_roll_previous_potential = None
        self._fast_roll_positive_rotation = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_roll_absolute_rotation = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_forward_metrics_cache = None
        self._fast_forward_phase = np.zeros((num_envs, 1), dtype=np.int32)
        self._fast_forward_phase_steps = np.zeros((num_envs, 1), dtype=np.int32)
        self._fast_forward_launch_high_water = np.zeros((num_envs, 1), dtype=np.float32)
        self._scratch_wr_v2_sync_high_water = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_forward_launch_ready_steps = np.zeros((num_envs, 1), dtype=np.int32)
        self._fast_forward_roll_high_water = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_forward_event_count = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_forward_event_anchor_rotation = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_forward_event_anchor_com_x = None
        self._fast_forward_event_anchor_support_index = None
        self._fast_forward_event_anchor_support_valid = np.zeros((num_envs, 1), dtype=bool)
        self._fast_forward_last_valid_support_index = None
        self._fast_forward_event_positive_rotation = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_forward_event_absolute_rotation = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_forward_episode_positive_rotation = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_forward_episode_absolute_rotation = np.zeros((num_envs, 1), dtype=np.float32)
        self._fast_forward_progress_age = np.zeros((num_envs, 1), dtype=np.int32)
        wave_midpoint = np.float32(0.5) * (
            self.tail_wave_action_low + self.tail_wave_action_high
        )
        self._scratch_wr_v2_filtered_wave_action = np.repeat(
            wave_midpoint[None, None, :], num_envs, axis=0
        ).astype(np.float32)
        self._scratch_wr_v2_filter_valid = np.zeros((num_envs, 1), dtype=bool)
        if control_mode == "tail_wave":
            initial_action_size = 6
        elif control_mode == "tail_wave_residual":
            initial_action_size = 6 + 2 * num_controlled_joints
        else:
            initial_action_size = max(1, len(formula_action_names))
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
        self.terrain_type = terrain_type
        self.terrain_settings = terrain_settings

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
            if self.terrain_contact_mode == "mesh_v1":
                self._terrain_contact_geometry = _prepare_terrain_contact_geometry(
                    np.array([[-1000, 1000]], dtype=np.complex64)
                )
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
            if self.terrain_contact_mode == "mesh_v1":
                self._terrain_contact_geometry = _prepare_terrain_contact_geometry(
                    terrain_mesh
                )
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
            if self.terrain_contact_mode == "mesh_v1":
                raise ValueError(
                    "terrain_contact_mode='mesh_v1' supports flat, stairs, and "
                    "tunnel terrains, but not mesh_cycle."
                )
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
        elif control_mode == "tail_wave_residual":
            action_size = 6 + 2 * num_controlled_joints
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
        elif control_mode == "tail_wave_residual":
            # Scratch-WR is deliberately a new opt-in controller.  The first
            # six actions define the existing global tail wave.  The remaining
            # interleaved K1/K2 pairs are joint-specific residual coefficients.
            # Neither branch is actuator-clipped on its own: they meet first,
            # and only their sum is clipped to the physical torque limit.
            if tail_side_key == "left":
                joint_coordinate = np.linspace(0.0, 1.0, num_controlled_joints, dtype=np.float32)[None, :]
            else:
                joint_coordinate = np.linspace(1.0, 0.0, num_controlled_joints, dtype=np.float32)[None, :]

            def func(action_params: np.ndarray, pos: np.ndarray, vel: np.ndarray, alpha: np.float32):
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
                wave_tau = kp * (target_curvature - current_curvature) - kd * current_thdot

                residual_flat = np.ascontiguousarray(action_params[:, 0, 6:])
                residual = residual_flat.reshape((num_envs, num_controlled_joints, 2))
                k1 = np.minimum(
                    np.maximum(residual[:, :, 0] * k_action_scale_value, scratch_wr_k1_min),
                    scratch_wr_k1_max,
                )
                k2 = np.minimum(
                    np.maximum(residual[:, :, 1] * k_action_scale_value, scratch_wr_k2_min),
                    scratch_wr_k2_max,
                )
                dth_for_action = dth.copy()
                dth_for_action[:, 0] = 0.0
                dth_for_action[:, -1] = 0.0
                dth_tot = roll_neg1_real(dth_for_action) - roll_pos1_real(dth_for_action)
                residual_tau = (
                    k1 * dth_tot[:, 1:-1]
                    + k2 * feedback_gain_value * current_thdot
                )
                combined_unclipped = wave_tau + alpha * residual_tau
                combined_tau = np.clip(combined_unclipped, -max_torque, max_torque)
                tauO = np.zeros_like(dth)
                tauO[:, 1:-1] = combined_tau
                fO = convert_torque_to_force(tauO, o, oM)
                fs = edge_force(pos, r, dv)
                ff = slide_force(vel)
                fw = wall_force(pos, vel)
                fg = grav_force(pos)
                return (
                    fO + fs + ff + fw + fg + fE + fD,
                    thdot,
                    wave_tau,
                    residual_tau,
                    combined_tau,
                    combined_unclipped,
                )

            get_forces = nb.njit(func)

            def func(action_params: np.ndarray, pos: np.ndarray, vel: np.ndarray, alpha: np.float32):
                for _ in range(physics_steps_per_timestep):
                    f, _, _, _, _, _ = get_forces(action_params, pos, vel, alpha)
                    vel += f * 0.5 * dt / particle_mass
                    pos += vel * dt
                    f, thdot, wave_tau, residual_tau, combined_tau, combined_unclipped = get_forces(
                        action_params, pos, vel, alpha
                    )
                    vel += f * 0.5 * dt / particle_mass
                return pos, vel, thdot, wave_tau, residual_tau, combined_tau, combined_unclipped

            velocity_verlet = nb.njit(func)
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
        if control_mode in {"tail_wave", "tail_wave_residual"}:
            # The sole global agent observes every physical joint.
            base_observation_size *= num_controlled_joints
        self.base_observation_size = int(base_observation_size)
        self.rolling_observation_size = 6 if self.rolling_observation else 0
        # 14 kinematic/shape features + 4-stage one-hot + 1 open-chain
        # joint coordinate measured from the configured tail.
        self.tail_roll_observation_size = 19 if self.tail_roll_observation else 0
        # Two-state one-hot, phase time, four event-progress diagnostics, and
        # the event-local direction-consistency fraction.
        # Kept as a separate opt-in block so every legacy observation remains
        # bit-for-bit shape compatible when the flag is disabled.
        self.fast_forward_observation_size = 8 if self.fast_forward_observation else 0
        # Six normalized applied wave parameters plus one filter-valid bit.
        # This makes the opt-in EMA controller state fully observable.
        self.scratch_wr_v2_observation_size = 7 if self.scratch_wr_v2 else 0
        observation_size = (
            base_observation_size
            + self.rolling_observation_size
            + self.tail_roll_observation_size
            + self.fast_forward_observation_size
            + self.scratch_wr_v2_observation_size
        )
        observation_mag = observation_funcs[observation_func]['mag']
        log_info_specs = {
            "speed": Unbounded(
                shape=torch.Size([num_envs, 1]),
                dtype=torch.float32,
            )
        }
        if self.terrain_contact_mode == "mesh_v1":
            for metric_name in TERRAIN_CONTACT_DIAGNOSTIC_FIELDS:
                log_info_specs[metric_name] = Unbounded(
                    shape=torch.Size([num_envs, 1]),
                    dtype=torch.float32,
                )
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
        if self.fast_rollover_metrics_enabled:
            for metric_name in (
                "fast_roll_phase",
                "fast_roll_phase_steps",
                "fast_roll_phase_progress",
                "fast_roll_flip_event",
                "fast_roll_flip_count",
                "fast_roll_cycle_count",
                "fast_roll_cycle_rotation",
                "fast_roll_cycle_forward",
                "fast_roll_support_migration",
                "fast_roll_direction_fraction",
                "fast_roll_motion_gate",
                "fast_roll_reward",
            ):
                log_info_specs[metric_name] = Unbounded(
                    shape=torch.Size([num_envs, 1]),
                    dtype=torch.float32,
                )
        if self.fast_forward_metrics_enabled:
            for metric_name in (
                "fast_forward_phase",
                "fast_forward_phase_steps",
                "fast_forward_launch_progress",
                "fast_forward_launch_ready_steps",
                "fast_forward_roll_progress",
                "fast_forward_progress_delta",
                "fast_forward_launch_event",
                "fast_forward_event_pulse",
                "fast_forward_event_bonus",
                "fast_forward_event_count",
                "fast_forward_event_rotation",
                "fast_forward_event_forward",
                "fast_forward_support_index",
                "fast_forward_support_migration_nodes",
                "fast_forward_ground_contact_strength",
                "fast_forward_event_direction_fraction",
                "fast_forward_episode_direction_fraction",
                "fast_forward_event_steps",
                "fast_forward_progress_age",
                "fast_forward_stall_penalty",
                "fast_forward_reward",
            ):
                log_info_specs[metric_name] = Unbounded(
                    shape=torch.Size([num_envs, 1]),
                    dtype=torch.float32,
                )
            if self.scratch_wr_v2 or self.reward_func in {"obs2_roll_repro_v2", "obs2_roll_repro_v2_1"}:
                for metric_name in (
                    "fast_forward_reverse_rotation_penalty",
                    "fast_forward_backward_penalty",
                    "effort_penalty",
                ):
                    log_info_specs[metric_name] = Unbounded(
                        shape=torch.Size([num_envs, 1]),
                        dtype=torch.float32,
                    )
            if self.scratch_wr_v2:
                for metric_name in (
                    "scratch_wr_v2_z0_lift_ratio",
                    "scratch_wr_v2_z0_forward_ratio",
                    "scratch_wr_v2_z0_curl_ratio",
                    "scratch_wr_v2_z0_sync_score",
                    "scratch_wr_v2_z0_candidate_mask",
                    "scratch_wr_v2_z0_active",
                    "scratch_wr_v2_z0_penalty_scale",
                    "scratch_wr_v2_z0_dense_reward",
                    "scratch_wr_v2_progress_delta",
                ):
                    log_info_specs[metric_name] = Unbounded(
                        shape=torch.Size([num_envs, 1]),
                        dtype=torch.float32,
                    )
        if control_mode in {"tail_wave", "tail_wave_residual"}:
            for metric_name in (
                "wave_amplitude", "wave_center", "wave_width",
                "wave_hold", "wave_kp", "wave_kd",
            ):
                log_info_specs[metric_name] = Unbounded(
                    shape=torch.Size([num_envs, 1]), dtype=torch.float32
                )
        if control_mode == "tail_wave_residual":
            for metric_name in (
                "scratch_wr_alpha",
                "scratch_wr_wave_torque_rms",
                "scratch_wr_residual_torque_rms",
                "scratch_wr_applied_residual_torque_rms",
                "scratch_wr_total_torque_rms",
                "scratch_wr_torque_clip_fraction",
                "scratch_wr_residual_saturation_fraction",
            ):
                log_info_specs[metric_name] = Unbounded(
                    shape=torch.Size([num_envs, 1]), dtype=torch.float32
                )
            if self.scratch_wr_v2:
                for metric_name in (
                    "scratch_wr_v2_wave_ema_beta",
                    "scratch_wr_v2_wave_filter_delta_rms",
                    "scratch_wr_v2_applied_wave_amplitude",
                    "scratch_wr_v2_applied_wave_center",
                    "scratch_wr_v2_applied_wave_width",
                    "scratch_wr_v2_applied_wave_hold",
                    "scratch_wr_v2_applied_wave_kp",
                    "scratch_wr_v2_applied_wave_kd",
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
        elif control_mode == "tail_wave_residual":
            action_low = torch.as_tensor(self.scratch_wr_action_low, dtype=torch.float32).view(1, 1, action_size).expand(num_envs, num_agents, action_size).clone()
            action_high = torch.as_tensor(self.scratch_wr_action_high, dtype=torch.float32).view(1, 1, action_size).expand(num_envs, num_agents, action_size).clone()
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

    def set_scratch_wr_alpha(self, alpha):
        """Set the residual authority used by the next Scratch-WR environment step."""
        alpha = float(alpha)
        if not np.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
            raise ValueError("Scratch-WR alpha must be a finite value in [0, 1].")
        self.scratch_wr_alpha = np.float32(alpha)
        self._scratch_wr_metrics_cache = None

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
        if self.control_mode == "tail_wave_residual":
            wave_low = self.tail_wave_action_low.reshape(1, 1, -1)
            wave_span = np.maximum(
                (self.tail_wave_action_high - self.tail_wave_action_low).reshape(1, 1, -1),
                np.float32(1e-6),
            )
            wave_normalized = np.float32(2.0) * (action[:, :, :6] - wave_low) / wave_span - np.float32(1.0)
            residual_low = self.scratch_wr_action_low[6:].reshape(1, 1, -1)
            residual_span = np.maximum(
                (self.scratch_wr_action_high[6:] - self.scratch_wr_action_low[6:]).reshape(1, 1, -1),
                np.float32(1e-6),
            )
            residual_normalized = (
                np.float32(2.0) * (action[:, :, 6:] - residual_low) / residual_span - np.float32(1.0)
            )
            wave_effort = np.mean(np.clip(wave_normalized ** 2, 0.0, 1.0), axis=(1, 2))[:, None]
            residual_effort = np.mean(np.clip(residual_normalized ** 2, 0.0, 1.0), axis=(1, 2))[:, None]
            effort_penalty = wave_effort + np.float32(self.scratch_wr_alpha ** 2) * residual_effort
        elif self.control_mode == "tail_wave":
            low = self.tail_wave_action_low.reshape(1, 1, -1)
            span = np.maximum(
                (self.tail_wave_action_high - self.tail_wave_action_low).reshape(1, 1, -1),
                np.float32(1e-6),
            )
            normalized_action = np.float32(2.0) * (action - low) / span - np.float32(1.0)
            effort_penalty = np.mean(np.clip(normalized_action ** 2, 0.0, 1.0), axis=(1, 2))[:, None]
        else:
            effort_penalty = np.mean(np.tanh(action ** 2), axis=tuple(range(1, action.ndim)), keepdims=False)[:, None]
        previous_action = np.asarray(self._previous_action_for_reward, dtype=np.float32)
        if previous_action.shape != action.shape:
            previous_action = np.reshape(previous_action, action.shape)
        action_delta = action - previous_action
        if self.control_mode == "tail_wave_residual":
            wave_span = np.maximum(
                (self.tail_wave_action_high - self.tail_wave_action_low).reshape(1, 1, -1),
                np.float32(1e-6),
            )
            residual_span = np.maximum(
                (self.scratch_wr_action_high[6:] - self.scratch_wr_action_low[6:]).reshape(1, 1, -1),
                np.float32(1e-6),
            )
            wave_delta = action_delta[:, :, :6] / wave_span
            residual_delta = action_delta[:, :, 6:] / residual_span
            wave_smoothness = np.mean(np.clip(wave_delta ** 2, 0.0, 1.0), axis=(1, 2))[:, None]
            residual_smoothness = np.mean(np.clip(residual_delta ** 2, 0.0, 1.0), axis=(1, 2))[:, None]
            action_smoothness_penalty = (
                wave_smoothness + np.float32(self.scratch_wr_alpha ** 2) * residual_smoothness
            )
            action_delta = None
        elif self.control_mode == "tail_wave":
            span = np.maximum(
                (self.tail_wave_action_high - self.tail_wave_action_low).reshape(1, 1, -1),
                np.float32(1e-6),
            )
            action_delta = action_delta / span
        elif self.control_mode == "direct":
            action_delta = action_delta / np.float32(max(self.max_torque, 1e-6))
        else:
            action_delta = np.tanh(action_delta)
        if action_delta is not None:
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

    def _get_terrain_contact_query(self):
        if self.terrain_contact_mode != "mesh_v1":
            raise RuntimeError(
                "Terrain-aware contact query requires terrain_contact_mode='mesh_v1'."
            )
        if self._terrain_contact_geometry is None:
            raise RuntimeError("Terrain-aware contact geometry is not initialized.")
        if self._terrain_contact_cache is None:
            self._terrain_contact_cache = _query_terrain_contact_geometry(
                np.asarray(self.pos, dtype=np.complex64),
                self._terrain_contact_geometry,
                self.particle_radius,
            )
        return self._terrain_contact_cache

    def _terrain_surface_contact_weights(
        self,
        surface_name,
        body_length,
        *,
        contact_band_fraction,
        softness_fraction,
    ):
        query = self._get_terrain_contact_query()
        distance = query[surface_name]["distance"]
        softness = np.maximum(
            np.float32(softness_fraction) * body_length,
            np.float32(1e-4),
        )
        band = np.float32(contact_band_fraction) * body_length
        weights = self._smooth_step(
            (
                self.particle_radius
                + band
                - distance
            )
            / softness
        )
        return np.asarray(weights, dtype=np.float32)

    def _compute_terrain_contact_diagnostics(self):
        query = self._get_terrain_contact_query()
        pos = np.asarray(self.pos, dtype=np.complex64)
        body_length = np.maximum(
            np.sum(np.abs(np.diff(pos, axis=1)), axis=1, keepdims=True),
            np.float32(1e-6),
        )
        surface_weights = {
            name: self._terrain_surface_contact_weights(
                name,
                body_length,
                contact_band_fraction=0.015,
                softness_fraction=0.01,
            )
            for name in ("floor", "wall", "ceiling")
        }
        diagnostics = {}
        for name in ("floor", "wall", "ceiling"):
            weights = surface_weights[name]
            diagnostics[f"terrain_{name}_contact_strength"] = np.max(
                weights, axis=1, keepdims=True
            ).astype(np.float32)
            clearance = query[name]["clearance"]
            minimum_clearance = np.min(clearance, axis=1, keepdims=True)
            diagnostics[f"terrain_{name}_clearance"] = np.where(
                np.isfinite(minimum_clearance),
                minimum_clearance,
                np.float32(1e6),
            ).astype(np.float32)

        floor_weights = surface_weights["floor"]
        material_index = np.arange(
            self.num_particles, dtype=np.float32
        )[None, :]
        floor_weight_sum = np.sum(floor_weights, axis=1, keepdims=True)
        floor_support_index = np.sum(
            floor_weights * material_index, axis=1, keepdims=True
        ) / np.maximum(floor_weight_sum, np.float32(1e-6))
        diagnostics["terrain_floor_support_index"] = np.where(
            floor_weight_sum > np.float32(1e-6),
            floor_support_index,
            np.float32(-1.0),
        ).astype(np.float32)
        diagnostics["terrain_floor_contact_count"] = np.sum(
            floor_weights >= np.float32(0.5), axis=1, keepdims=True
        ).astype(np.float32)

        floor_normals = query["floor"]["normal"]
        weighted_normal = np.sum(
            floor_weights * floor_normals, axis=1, keepdims=True
        )
        weighted_normal_magnitude = np.abs(weighted_normal)
        closest_floor_particle = np.argmin(
            query["floor"]["distance"], axis=1
        )[:, None]
        fallback_floor_normal = np.take_along_axis(
            floor_normals, closest_floor_particle, axis=1
        )
        support_normal = np.where(
            weighted_normal_magnitude > np.float32(1e-6),
            weighted_normal
            / np.maximum(weighted_normal_magnitude, np.float32(1e-6)),
            fallback_floor_normal,
        ).astype(np.complex64)
        diagnostics["terrain_floor_normal_x"] = np.real(
            support_normal
        ).astype(np.float32)
        diagnostics["terrain_floor_normal_y"] = np.imag(
            support_normal
        ).astype(np.float32)

        closest_particle = np.argmin(
            query["nearest_clearance"], axis=1
        )[:, None]
        diagnostics["terrain_nearest_clearance"] = np.take_along_axis(
            query["nearest_clearance"], closest_particle, axis=1
        ).astype(np.float32)
        diagnostics["terrain_nearest_surface_kind"] = np.take_along_axis(
            query["nearest_surface_kind"], closest_particle, axis=1
        ).astype(np.float32)
        return diagnostics

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
        if self.terrain_contact_mode == "mesh_v1":
            floor_distance = self._get_terrain_contact_query()["floor"]["distance"]
            head_floor_distance = floor_distance[
                :, self.head_index : self.head_index + 1
            ]
            head_height = np.maximum(
                head_floor_distance - self.particle_radius,
                np.float32(0.0),
            )
        else:
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
        if self.terrain_contact_mode == "mesh_v1":
            ordered_floor_distance = (
                floor_distance
                if self.tail_side == "left"
                else floor_distance[:, ::-1]
            )
            contact_weights = self._smooth_step(
                (
                    self.particle_radius
                    + self.tail_roll_contact_margin * body_length
                    - ordered_floor_distance
                )
                / contact_softness
            )
        else:
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
            "body_length": np.asarray(body_length, dtype=np.float32),
            "com_x": np.asarray(com_x, dtype=np.float32),
            "support_x": np.asarray(support_x, dtype=np.float32),
        }
        return self._tail_roll_metrics_cache

    def _compute_fast_rollover_metrics(self):
        if self._fast_rollover_metrics_cache is not None:
            return self._fast_rollover_metrics_cache

        tail = self._compute_tail_roll_metrics()
        body_length = np.maximum(tail["body_length"], np.float32(1e-6))
        com_x = tail["com_x"]
        support_x = tail["support_x"]
        direction = self.rolling_direction_sign
        desired_rotation = -direction * tail["cumulative_rotation"]

        if self._fast_roll_cycle_start_com_x is None:
            self._fast_roll_cycle_start_com_x = com_x.copy()
        if self._fast_roll_cycle_start_support_x is None:
            self._fast_roll_cycle_start_support_x = support_x.copy()

        cycle_rotation = desired_rotation - self._fast_roll_cycle_start_rotation
        cycle_forward = direction * (com_x - self._fast_roll_cycle_start_com_x) / body_length
        support_migration = direction * (support_x - self._fast_roll_cycle_start_support_x) / body_length
        raw_rotation_increment = -direction * tail["rotation_increment"]
        self._fast_roll_positive_rotation += np.maximum(raw_rotation_increment, np.float32(0.0))
        self._fast_roll_absolute_rotation += np.abs(raw_rotation_increment)
        direction_fraction = self._fast_roll_positive_rotation / np.maximum(
            self._fast_roll_absolute_rotation, np.float32(1e-6)
        )

        phase_before = self._fast_roll_phase.copy()
        phase_steps_before = self._fast_roll_phase_steps.copy()
        self._fast_roll_phase_steps += 1

        curl_ready = (
            (tail["tail_lift_score"] >= np.float32(0.20))
            & (tail["tail_forward_score"] >= np.float32(0.10))
            & (tail["curl_prefix_progress"] >= np.float32(0.12))
        )
        flip_ready = (
            (cycle_rotation >= self.fast_rollover_flip_radians)
            & (cycle_forward >= self.fast_rollover_forward_fraction)
            & (support_migration >= self.fast_rollover_support_fraction)
        )
        reset_ready = (
            (tail["closure_ratio"] >= self.fast_rollover_reset_open_ratio)
            & (tail["curl_prefix_progress"] <= np.float32(0.30))
        )

        curl_transition = (phase_before == 0) & curl_ready
        flip_event = (phase_before == 1) & flip_ready
        reset_event = (phase_before == 2) & reset_ready

        self._fast_roll_phase = np.where(curl_transition, 1, self._fast_roll_phase)
        self._fast_roll_phase = np.where(flip_event, 2, self._fast_roll_phase)
        self._fast_roll_phase = np.where(reset_event, 0, self._fast_roll_phase)
        phase_changed = self._fast_roll_phase != phase_before
        self._fast_roll_phase_steps = np.where(
            phase_changed, np.int32(0), self._fast_roll_phase_steps
        ).astype(np.int32)
        self._fast_roll_flip_count += flip_event.astype(np.float32)
        self._fast_roll_cycle_count += reset_event.astype(np.float32)

        event_steps = np.maximum(phase_steps_before.astype(np.float32), np.float32(1.0))
        event_speed = np.clip(
            np.float32(self.fast_rollover_cycle_target_steps) / event_steps,
            np.float32(0.5),
            np.float32(2.0),
        )
        event_bonus = np.float32(4.0) * event_speed * flip_event.astype(np.float32)

        if np.any(reset_event):
            self._fast_roll_cycle_start_rotation = np.where(
                reset_event, desired_rotation, self._fast_roll_cycle_start_rotation
            )
            self._fast_roll_cycle_start_com_x = np.where(
                reset_event, com_x, self._fast_roll_cycle_start_com_x
            )
            self._fast_roll_cycle_start_support_x = np.where(
                reset_event, support_x, self._fast_roll_cycle_start_support_x
            )

        curl_potential = (
            np.float32(0.35) * tail["tail_lift_score"]
            + np.float32(0.25) * tail["tail_forward_score"]
            + np.float32(0.40) * tail["curl_prefix_progress"]
        )
        flip_potential = (
            np.float32(0.55) * np.clip(cycle_rotation / self.fast_rollover_flip_radians, 0.0, 1.0)
            + np.float32(0.25) * np.clip(
                cycle_forward / max(float(self.fast_rollover_forward_fraction), 1e-6), 0.0, 1.0
            )
            + np.float32(0.20) * np.clip(
                support_migration / max(float(self.fast_rollover_support_fraction), 1e-6), 0.0, 1.0
            )
        )
        extend_potential = (
            np.float32(0.55) * np.clip(
                (tail["closure_ratio"] - np.float32(0.35)) / np.float32(0.50), 0.0, 1.0
            )
            + np.float32(0.45) * (np.float32(1.0) - tail["curl_prefix_progress"])
        )
        phase = self._fast_roll_phase
        phase_potential = np.where(
            phase == 0,
            curl_potential,
            np.where(phase == 1, flip_potential, extend_potential),
        ).astype(np.float32)
        phase_progress = np.where(
            phase == 0,
            curl_potential,
            np.where(
                phase == 1,
                np.clip(cycle_rotation / self.fast_rollover_flip_radians, 0.0, 1.0),
                extend_potential,
            ),
        ).astype(np.float32)
        # Forward/contact rewards are active only after a real tail curl and
        # some desired rotation; this prevents a straight crawling policy.
        motion_gate = (phase >= 1).astype(np.float32) * self._smooth_step(
            (cycle_rotation - np.float32(np.deg2rad(10.0))) / np.float32(np.deg2rad(5.0))
        )

        self._fast_rollover_metrics_cache = {
            "fast_roll_phase": phase.astype(np.float32),
            "fast_roll_phase_steps": self._fast_roll_phase_steps.astype(np.float32),
            "fast_roll_phase_progress": phase_progress,
            "fast_roll_phase_potential": phase_potential,
            "fast_roll_phase_changed": phase_changed.astype(np.float32),
            "fast_roll_curl_transition": curl_transition.astype(np.float32),
            "fast_roll_flip_event": flip_event.astype(np.float32),
            "fast_roll_flip_count": self._fast_roll_flip_count.copy(),
            "fast_roll_cycle_count": self._fast_roll_cycle_count.copy(),
            "fast_roll_cycle_rotation": np.asarray(cycle_rotation, dtype=np.float32),
            "fast_roll_cycle_forward": np.asarray(cycle_forward, dtype=np.float32),
            "fast_roll_support_migration": np.asarray(support_migration, dtype=np.float32),
            "fast_roll_direction_fraction": np.asarray(direction_fraction, dtype=np.float32),
            "fast_roll_motion_gate": np.asarray(motion_gate, dtype=np.float32),
            "fast_roll_event_bonus": np.asarray(event_bonus, dtype=np.float32),
            "desired_rotation_increment": tail["desired_rotation_increment"],
            "forward_displacement_increment": tail["forward_displacement_increment"],
            "contact_migration_increment": tail["contact_migration_increment"],
            "effort_penalty": tail["effort_penalty"],
            "action_smoothness_penalty": tail["action_smoothness_penalty"],
            "fast_roll_reward": np.zeros((self.num_envs, 1), dtype=np.float32),
        }
        return self._fast_rollover_metrics_cache

    def _compute_fast_forward_roll_v2_metrics(self):
        """Compute and advance the closure-free fast-forward event detector.

        This method is cached once per environment step because it advances
        event anchors and high-water marks.  ``_reset`` sets ``steps`` to zero,
        so constructing the initial observation never advances the state
        machine.
        """
        if self._fast_forward_metrics_cache is not None:
            return self._fast_forward_metrics_cache

        tail = self._compute_tail_roll_metrics()
        body_length = np.maximum(tail["body_length"], np.float32(1e-6))
        com_x = tail["com_x"]
        direction = self.rolling_direction_sign
        desired_rotation = -direction * tail["cumulative_rotation"]
        raw_rotation_increment = -direction * tail["rotation_increment"]
        forward_step = (
            direction * np.asarray(self.mean_speed, dtype=np.float32) / body_length
        ).astype(np.float32)

        # Material-coordinate ground support.  Unlike absolute support_x, this
        # does not increase merely because a straight crawler translates.  A
        # pulse also requires a currently valid ground contact, so an airborne
        # shape change cannot masquerade as contact migration.  legacy_flat
        # retains the original absolute-y detector bit-for-bit; mesh_v1 uses
        # the cached nearest floor segment for flat, stairs, and tunnel alike.
        pos = np.asarray(self.pos, dtype=np.complex64)
        ordered = pos if self.tail_side == "left" else pos[:, ::-1]
        y = np.imag(ordered)
        contact_softness = np.maximum(np.float32(0.01) * body_length, np.float32(1e-4))
        contact_band = np.float32(0.015) * body_length
        if self.terrain_contact_mode == "mesh_v1":
            floor_distance = self._get_terrain_contact_query()["floor"]["distance"]
            ordered_floor_distance = (
                floor_distance
                if self.tail_side == "left"
                else floor_distance[:, ::-1]
            )
            contact_weights = self._smooth_step(
                (
                    self.particle_radius
                    + contact_band
                    - ordered_floor_distance
                )
                / contact_softness
            )
        else:
            contact_weights = self._smooth_step(
                (self.particle_radius + contact_band - y) / contact_softness
            )
        contact_strength = np.max(contact_weights, axis=1, keepdims=True)
        contact_valid = contact_strength >= np.float32(0.50)
        material_index = np.arange(self.num_particles, dtype=np.float32)[None, :]
        support_index = np.sum(contact_weights * material_index, axis=1, keepdims=True) / np.maximum(
            np.sum(contact_weights, axis=1, keepdims=True), np.float32(1e-6)
        )

        if self._fast_forward_last_valid_support_index is None:
            self._fast_forward_last_valid_support_index = support_index.copy()
        effective_support_index = np.where(
            contact_valid, support_index, self._fast_forward_last_valid_support_index
        ).astype(np.float32)
        if self._fast_forward_event_anchor_com_x is None:
            self._fast_forward_event_anchor_com_x = com_x.copy()
        if self._fast_forward_event_anchor_support_index is None:
            self._fast_forward_event_anchor_support_index = effective_support_index.copy()

        phase_before = self._fast_forward_phase.copy()
        phase_steps_before = self._fast_forward_phase_steps.copy()
        advance_state = bool(self.steps > 0)
        if advance_state:
            self._fast_forward_phase_steps += 1
            self._fast_forward_episode_positive_rotation += np.maximum(
                raw_rotation_increment, np.float32(0.0)
            )
            self._fast_forward_episode_absolute_rotation += np.abs(raw_rotation_increment)
            active = phase_before == 1
            self._fast_forward_event_positive_rotation += np.where(
                active, np.maximum(raw_rotation_increment, np.float32(0.0)), np.float32(0.0)
            )
            self._fast_forward_event_absolute_rotation += np.where(
                active, np.abs(raw_rotation_increment), np.float32(0.0)
            )
            self._fast_forward_last_valid_support_index = np.where(
                contact_valid, support_index, self._fast_forward_last_valid_support_index
            ).astype(np.float32)

            # Cache the most recent *real* material support while waiting for
            # launch.  Tail launch commonly happens after the body has already
            # left the floor; in that case the rolling event must start from
            # the pre-launch contact instead of discarding it and acquiring a
            # fresh anchor on landing (which would zero the migration signal).
            prelaunch_support_cache = (phase_before == 0) & contact_valid
            self._fast_forward_event_anchor_support_index = np.where(
                prelaunch_support_cache,
                support_index,
                self._fast_forward_event_anchor_support_index,
            ).astype(np.float32)
            self._fast_forward_event_anchor_support_valid = np.where(
                prelaunch_support_cache,
                True,
                self._fast_forward_event_anchor_support_valid,
            ).astype(bool)

        rolling = self._compute_rolling_metrics()
        if self.reward_func == "obs2_roll_repro_v1":
            launch_potential = (
                np.float32(0.65) * rolling["closure_score"]
                + np.float32(0.35) * rolling["circularity_score"]
            )
        else:
            launch_potential = (
                np.float32(0.40) * tail["tail_lift_score"]
                + np.float32(0.35) * tail["curl_prefix_progress"]
                + np.float32(0.15) * tail["tail_forward_score"]
                + np.float32(0.10) * tail["head_contact_score"]
                - np.float32(0.15) * tail["curl_order_penalty"]
            )
        launch_potential = np.clip(launch_potential, 0.0, 1.0).astype(np.float32)
        if self.reward_func == "obs2_roll_repro_v1" and not advance_state:
            # The initial straight pose is the potential baseline, not a free
            # first-step reward.  This state is reward-only and never observed.
            self._fast_forward_launch_high_water = launch_potential.copy()
        ratio_epsilon = np.float32(1e-4)
        lift_ratio = np.clip(
            tail["tail_lift_score"] / np.maximum(self.fast_forward_launch_lift, ratio_epsilon),
            0.0,
            1.0,
        ).astype(np.float32)
        forward_ratio = np.clip(
            tail["tail_forward_score"] / np.maximum(self.fast_forward_launch_forward, ratio_epsilon),
            0.0,
            1.0,
        ).astype(np.float32)
        curl_ratio = np.clip(
            tail["curl_prefix_progress"] / np.maximum(self.fast_forward_launch_curl, ratio_epsilon),
            0.0,
            1.0,
        ).astype(np.float32)
        head_gate = self._smooth_step(
            (tail["head_contact_score"] - self.fast_forward_launch_head_contact)
            / np.float32(0.10)
        )
        weakest_ratio = np.minimum(np.minimum(lift_ratio, forward_ratio), curl_ratio)
        harmonic_ratio = np.float32(3.0) / (
            np.float32(1.0) / (lift_ratio + ratio_epsilon)
            + np.float32(1.0) / (forward_ratio + ratio_epsilon)
            + np.float32(1.0) / (curl_ratio + ratio_epsilon)
        )
        sync_raw = head_gate * (
            np.float32(0.70) * weakest_ratio + np.float32(0.30) * harmonic_ratio
        )
        sync_score = np.clip(
            (sync_raw - np.float32(0.02)) / np.float32(0.98), 0.0, 1.0
        ).astype(np.float32)
        if self.reward_func in {"obs2_roll_repro_v2", "obs2_roll_repro_v2_1"}:
            # v2's preparation potential is the weakest-link tail synchrony;
            # closure and circularity remain diagnostics only.  A pose that
            # has never supplied real floor support cannot collect preparation
            # progress merely by hovering near the head-contact surrogate.
            launch_potential = (
                sync_score
                * self._fast_forward_event_anchor_support_valid.astype(np.float32)
            ).astype(np.float32)
        if self.reward_func in {"obs2_roll_repro_v1", "obs2_roll_repro_v2", "obs2_roll_repro_v2_1"} and not advance_state:
            # The initial pose is the potential baseline, not a free first-step
            # reward.  This state is reward-only and is never observed.
            self._fast_forward_launch_high_water = launch_potential.copy()
        if self.reward_func == "obs2_roll_repro_v1":
            launch_ready = (
                (rolling["closure_score"] >= np.float32(0.80))
                & (rolling["circularity_score"] >= np.float32(0.65))
            )
        elif self.reward_func in {"obs2_roll_repro_v2", "obs2_roll_repro_v2_1"}:
            launch_ready = (
                (tail["tail_lift_score"] >= self.fast_forward_launch_lift)
                & (tail["tail_forward_score"] >= self.fast_forward_launch_forward)
                & (tail["curl_prefix_progress"] >= self.fast_forward_launch_curl)
                & (tail["head_contact_score"] >= self.fast_forward_launch_head_contact)
                & self._fast_forward_event_anchor_support_valid
            )
        else:
            launch_ready = (
                (tail["tail_lift_score"] >= self.fast_forward_launch_lift)
                & (tail["tail_forward_score"] >= self.fast_forward_launch_forward)
                & (tail["curl_prefix_progress"] >= self.fast_forward_launch_curl)
                & (tail["head_contact_score"] >= self.fast_forward_launch_head_contact)
            )

        event_rotation = desired_rotation - self._fast_forward_event_anchor_rotation
        event_forward = (
            direction * (com_x - self._fast_forward_event_anchor_com_x) / body_length
        )
        support_migration_nodes = np.abs(
            effective_support_index - self._fast_forward_event_anchor_support_index
        )
        event_direction_fraction = self._fast_forward_event_positive_rotation / np.maximum(
            self._fast_forward_event_absolute_rotation, np.float32(1e-6)
        )
        episode_direction_fraction = self._fast_forward_episode_positive_rotation / np.maximum(
            self._fast_forward_episode_absolute_rotation, np.float32(1e-6)
        )
        rotation_progress = np.clip(
            event_rotation / self.fast_forward_event_radians, 0.0, 1.0
        )
        forward_progress = np.clip(
            event_forward / self.fast_forward_event_forward_fraction, 0.0, 1.0
        )
        contact_progress = np.clip(
            support_migration_nodes / self.fast_forward_event_contact_nodes, 0.0, 1.0
        )
        roll_potential = (
            np.float32(0.55) * rotation_progress
            + np.float32(0.30) * forward_progress
            + np.float32(0.15) * contact_progress
        ).astype(np.float32)

        launch_event = np.zeros((self.num_envs, 1), dtype=bool)
        event_pulse = np.zeros((self.num_envs, 1), dtype=bool)
        progress_delta = np.zeros((self.num_envs, 1), dtype=np.float32)
        event_bonus = np.zeros((self.num_envs, 1), dtype=np.float32)
        sync_progress_delta = np.zeros((self.num_envs, 1), dtype=np.float32)

        if advance_state:
            launch_ready_now = (phase_before == 0) & launch_ready
            self._fast_forward_launch_ready_steps = np.where(
                launch_ready_now,
                self._fast_forward_launch_ready_steps + np.int32(1),
                np.int32(0),
            ).astype(np.int32)
            launch_high_water = np.maximum(
                self._fast_forward_launch_high_water, launch_potential
            )
            launch_delta = launch_high_water - self._fast_forward_launch_high_water
            sync_high_water = np.maximum(
                self._scratch_wr_v2_sync_high_water, sync_score
            )
            sync_progress_delta = (
                sync_high_water - self._scratch_wr_v2_sync_high_water
            ).astype(np.float32)
            roll_high_water = np.maximum(self._fast_forward_roll_high_water, roll_potential)
            roll_delta = roll_high_water - self._fast_forward_roll_high_water
            progress_delta = np.where(
                phase_before == 0, launch_delta, roll_delta
            ).astype(np.float32)
            self._fast_forward_launch_high_water = np.where(
                phase_before == 0, launch_high_water, self._fast_forward_launch_high_water
            ).astype(np.float32)
            self._scratch_wr_v2_sync_high_water = np.where(
                phase_before == 0,
                sync_high_water,
                self._scratch_wr_v2_sync_high_water,
            ).astype(np.float32)
            self._fast_forward_roll_high_water = np.where(
                phase_before == 1, roll_high_water, self._fast_forward_roll_high_water
            ).astype(np.float32)

            launch_event = (
                (phase_before == 0)
                & launch_ready
                & (
                    self._fast_forward_launch_ready_steps
                    >= self.fast_forward_launch_hold_steps
                )
            )
            support_anchor_acquire = (
                (phase_before == 1)
                & (~self._fast_forward_event_anchor_support_valid)
                & contact_valid
            )
            event_pulse = (
                (phase_before == 1)
                & (event_rotation >= self.fast_forward_event_radians)
                & (event_forward >= self.fast_forward_event_forward_fraction)
                & (support_migration_nodes >= self.fast_forward_event_contact_nodes)
                & (event_direction_fraction >= self.fast_forward_direction_fraction_threshold)
                & contact_valid
                & self._fast_forward_event_anchor_support_valid
            )

            event_steps = np.maximum(
                self._fast_forward_phase_steps.astype(np.float32), np.float32(1.0)
            )
            speed_factor = np.clip(
                np.float32(self.fast_forward_event_target_steps) / event_steps,
                np.float32(0.5),
                np.float32(1.5),
            )
            event_bonus = (
                np.float32(3.0) * speed_factor * event_pulse.astype(np.float32)
            )

            # The launch transition establishes the first rolling anchor.  A
            # pulse immediately installs the next anchor; no open/reset phase
            # is required, so continuous forward tumbling can be counted.
            self._fast_forward_phase = np.where(
                launch_event, np.int32(1), self._fast_forward_phase
            ).astype(np.int32)
            self._scratch_wr_v2_sync_high_water = np.where(
                self._fast_forward_phase == 0,
                self._scratch_wr_v2_sync_high_water,
                np.float32(0.0),
            ).astype(np.float32)
            anchor_update = launch_event | event_pulse
            self._fast_forward_event_anchor_rotation = np.where(
                anchor_update, desired_rotation, self._fast_forward_event_anchor_rotation
            ).astype(np.float32)
            self._fast_forward_event_anchor_com_x = np.where(
                anchor_update, com_x, self._fast_forward_event_anchor_com_x
            ).astype(np.float32)
            valid_anchor_update = anchor_update & contact_valid
            support_anchor_update = valid_anchor_update | support_anchor_acquire
            self._fast_forward_event_anchor_support_index = np.where(
                support_anchor_update,
                support_index,
                self._fast_forward_event_anchor_support_index,
            ).astype(np.float32)
            # An airborne launch inherits the cached pre-launch material
            # support.  Only a real current contact may install a new support
            # anchor; lack of contact never invalidates a previously valid one.
            self._fast_forward_event_anchor_support_valid = np.where(
                support_anchor_update,
                True,
                self._fast_forward_event_anchor_support_valid,
            ).astype(bool)
            self._fast_forward_event_positive_rotation = np.where(
                anchor_update, np.float32(0.0), self._fast_forward_event_positive_rotation
            ).astype(np.float32)
            self._fast_forward_event_absolute_rotation = np.where(
                anchor_update, np.float32(0.0), self._fast_forward_event_absolute_rotation
            ).astype(np.float32)
            self._fast_forward_roll_high_water = np.where(
                anchor_update, np.float32(0.0), self._fast_forward_roll_high_water
            ).astype(np.float32)
            self._fast_forward_phase_steps = np.where(
                anchor_update, np.int32(0), self._fast_forward_phase_steps
            ).astype(np.int32)
            self._fast_forward_event_count += event_pulse.astype(np.float32)
            self._fast_forward_launch_ready_steps = np.where(
                self._fast_forward_phase == 0,
                self._fast_forward_launch_ready_steps,
                np.int32(0),
            ).astype(np.int32)

            made_progress = (
                (progress_delta > np.float32(1e-6)) | launch_event | event_pulse
            )
            self._fast_forward_progress_age = np.where(
                made_progress,
                np.int32(0),
                self._fast_forward_progress_age + np.int32(1),
            ).astype(np.int32)

        stall_penalty = (
            (self._fast_forward_phase == 1)
            & (self._fast_forward_progress_age >= self.fast_forward_stall_steps)
        ).astype(np.float32)
        reverse_rotation_penalty = np.clip(
            np.maximum(-raw_rotation_increment, np.float32(0.0))
            / self.fast_forward_rotation_step_ref,
            0.0,
            1.0,
        ).astype(np.float32)
        backward_penalty = np.clip(
            np.maximum(-forward_step, np.float32(0.0))
            / self.fast_forward_translation_step_ref,
            0.0,
            1.0,
        ).astype(np.float32)

        z0_active = (
            self.scratch_wr_v2
            and float(self.scratch_wr_alpha) <= 1e-8
        )
        z0_active_mask = (
            (phase_before == 0) if z0_active else np.zeros_like(phase_before, dtype=bool)
        )
        if self.scratch_wr_v2_penalty_anneal_batches > 0:
            penalty_fraction = np.float32(
                np.clip(
                    self.curriculum_episode / self.scratch_wr_v2_penalty_anneal_batches,
                    0.0,
                    1.0,
                )
            )
        else:
            penalty_fraction = np.float32(1.0)
        scheduled_penalty_scale = np.full(
            (self.num_envs, 1),
            self.scratch_wr_v2_penalty_start_scale
            + (np.float32(1.0) - self.scratch_wr_v2_penalty_start_scale) * penalty_fraction,
            dtype=np.float32,
        )
        z0_penalty_scale = np.where(
            z0_active_mask, scheduled_penalty_scale, np.float32(1.0)
        ).astype(np.float32)
        scratch_wr_v2_progress_delta = np.where(
            z0_active_mask, sync_progress_delta, progress_delta
        ).astype(np.float32)
        z0_dense_reward = (
            self.scratch_wr_v2_sync_dense_weight
            * sync_score
            * z0_active_mask.astype(np.float32)
        ).astype(np.float32)

        self._fast_forward_metrics_cache = {
            "fast_forward_phase": self._fast_forward_phase.astype(np.float32),
            "fast_forward_phase_steps": self._fast_forward_phase_steps.astype(np.float32),
            "fast_forward_launch_progress": launch_potential.astype(np.float32),
            "fast_forward_launch_ready_steps": self._fast_forward_launch_ready_steps.astype(np.float32),
            "fast_forward_roll_progress": roll_potential.astype(np.float32),
            "fast_forward_progress_delta": progress_delta,
            "fast_forward_launch_event": launch_event.astype(np.float32),
            "fast_forward_event_pulse": event_pulse.astype(np.float32),
            "fast_forward_event_bonus": event_bonus.astype(np.float32),
            "fast_forward_event_count": self._fast_forward_event_count.copy(),
            "fast_forward_event_rotation": np.asarray(event_rotation, dtype=np.float32),
            "fast_forward_event_forward": np.asarray(event_forward, dtype=np.float32),
            "fast_forward_support_index": np.asarray(support_index, dtype=np.float32),
            "fast_forward_support_migration_nodes": np.asarray(
                support_migration_nodes, dtype=np.float32
            ),
            "fast_forward_ground_contact_strength": np.asarray(
                contact_strength, dtype=np.float32
            ),
            "fast_forward_event_direction_fraction": np.asarray(
                event_direction_fraction, dtype=np.float32
            ),
            "fast_forward_episode_direction_fraction": np.asarray(
                episode_direction_fraction, dtype=np.float32
            ),
            "fast_forward_event_steps": self._fast_forward_phase_steps.astype(np.float32),
            "fast_forward_progress_age": self._fast_forward_progress_age.astype(np.float32),
            "fast_forward_stall_penalty": stall_penalty,
            "fast_forward_reverse_rotation_penalty": reverse_rotation_penalty,
            "fast_forward_backward_penalty": backward_penalty,
            "effort_penalty": tail["effort_penalty"],
            "action_smoothness_penalty": tail["action_smoothness_penalty"],
            "scratch_wr_v2_z0_lift_ratio": lift_ratio,
            "scratch_wr_v2_z0_forward_ratio": forward_ratio,
            "scratch_wr_v2_z0_curl_ratio": curl_ratio,
            "scratch_wr_v2_z0_sync_score": sync_score,
            "scratch_wr_v2_z0_candidate_mask": launch_ready.astype(np.float32),
            "scratch_wr_v2_z0_active": z0_active_mask.astype(np.float32),
            "scratch_wr_v2_z0_penalty_scale": z0_penalty_scale,
            "scratch_wr_v2_z0_dense_reward": z0_dense_reward,
            "scratch_wr_v2_progress_delta": scratch_wr_v2_progress_delta,
            "fast_forward_reward": np.zeros((self.num_envs, 1), dtype=np.float32),
        }
        return self._fast_forward_metrics_cache

    def _get_obs(self):
        pos = np.ascontiguousarray(self.pos, dtype=np.complex64)
        thdot = np.ascontiguousarray(self.thdot, dtype=np.float32)
        local_obs = self._obs_func(pos, thdot)
        if self.control_mode in {"tail_wave", "tail_wave_residual"}:
            local_obs = local_obs.reshape(self.num_envs, 1, -1)
        if (
            not self.rolling_observation
            and not self.tail_roll_observation
            and not self.fast_forward_observation
            and not self.scratch_wr_v2
        ):
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

        if self.fast_forward_observation:
            fast_metrics = self._compute_fast_forward_roll_v2_metrics()
            phase_index = np.clip(
                fast_metrics["fast_forward_phase"].astype(np.int32), 0, 1
            )[:, 0]
            phase_one_hot = np.zeros((self.num_envs, 2), dtype=np.float32)
            phase_one_hot[np.arange(self.num_envs), phase_index] = np.float32(1.0)
            phase_time = np.clip(
                fast_metrics["fast_forward_phase_steps"]
                / np.float32(max(1, self.fast_forward_event_target_steps)),
                0.0,
                1.0,
            )
            rotation_progress = np.clip(
                fast_metrics["fast_forward_event_rotation"]
                / self.fast_forward_event_radians,
                0.0,
                1.0,
            )
            forward_progress = np.clip(
                fast_metrics["fast_forward_event_forward"]
                / self.fast_forward_event_forward_fraction,
                0.0,
                1.0,
            )
            contact_progress = np.clip(
                fast_metrics["fast_forward_support_migration_nodes"]
                / self.fast_forward_event_contact_nodes,
                0.0,
                1.0,
            )
            fast_global = np.concatenate(
                (
                    phase_one_hot,
                    phase_time,
                    fast_metrics["fast_forward_launch_progress"],
                    rotation_progress,
                    forward_progress,
                    contact_progress,
                    fast_metrics["fast_forward_event_direction_fraction"],
                ),
                axis=1,
            )
            fast_global = np.repeat(fast_global[:, None, :], self.num_agents, axis=1)
            feature_blocks.append(fast_global)

        if self.scratch_wr_v2:
            wave_span = np.maximum(
                self.tail_wave_action_high - self.tail_wave_action_low,
                np.float32(1e-6),
            )
            filtered_wave = np.asarray(
                self._scratch_wr_v2_filtered_wave_action, dtype=np.float32
            )[:, 0, :]
            filtered_wave_normalized = (
                np.float32(2.0)
                * (filtered_wave - self.tail_wave_action_low[None, :])
                / wave_span[None, :]
                - np.float32(1.0)
            )
            filtered_wave_normalized = np.where(
                self._scratch_wr_v2_filter_valid,
                filtered_wave_normalized,
                np.float32(0.0),
            ).astype(np.float32)
            filter_state = np.concatenate(
                (
                    filtered_wave_normalized,
                    self._scratch_wr_v2_filter_valid.astype(np.float32),
                ),
                axis=1,
            )
            feature_blocks.append(
                np.repeat(filter_state[:, None, :], self.num_agents, axis=1)
            )

        return np.ascontiguousarray(np.concatenate(feature_blocks, axis=2), dtype=np.float32)

    def _get_info(self):
        values = {
            "speed": torch.as_tensor(
                np.real(self.mean_speed),
                dtype=torch.float32,
                device=self.device,
            )
        }
        if self.terrain_contact_mode == "mesh_v1":
            terrain_contact_diagnostics = (
                self._compute_terrain_contact_diagnostics()
            )
            for metric_name in TERRAIN_CONTACT_DIAGNOSTIC_FIELDS:
                values[metric_name] = torch.as_tensor(
                    terrain_contact_diagnostics[metric_name],
                    dtype=torch.float32,
                    device=self.device,
                )
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
        if self.fast_rollover_metrics_enabled:
            fast_metrics = self._compute_fast_rollover_metrics()
            for metric_name in (
                "fast_roll_phase",
                "fast_roll_phase_steps",
                "fast_roll_phase_progress",
                "fast_roll_flip_event",
                "fast_roll_flip_count",
                "fast_roll_cycle_count",
                "fast_roll_cycle_rotation",
                "fast_roll_cycle_forward",
                "fast_roll_support_migration",
                "fast_roll_direction_fraction",
                "fast_roll_motion_gate",
                "fast_roll_reward",
            ):
                values[metric_name] = torch.as_tensor(
                    fast_metrics[metric_name], dtype=torch.float32, device=self.device
                )
        if self.fast_forward_metrics_enabled:
            fast_forward_metrics = self._compute_fast_forward_roll_v2_metrics()
            for metric_name in (
                "fast_forward_phase",
                "fast_forward_phase_steps",
                "fast_forward_launch_progress",
                "fast_forward_launch_ready_steps",
                "fast_forward_roll_progress",
                "fast_forward_progress_delta",
                "fast_forward_launch_event",
                "fast_forward_event_pulse",
                "fast_forward_event_bonus",
                "fast_forward_event_count",
                "fast_forward_event_rotation",
                "fast_forward_event_forward",
                "fast_forward_support_index",
                "fast_forward_support_migration_nodes",
                "fast_forward_ground_contact_strength",
                "fast_forward_event_direction_fraction",
                "fast_forward_episode_direction_fraction",
                "fast_forward_event_steps",
                "fast_forward_progress_age",
                "fast_forward_stall_penalty",
                "fast_forward_reward",
            ):
                values[metric_name] = torch.as_tensor(
                    fast_forward_metrics[metric_name],
                    dtype=torch.float32,
                    device=self.device,
                )
            if self.scratch_wr_v2 or self.reward_func in {"obs2_roll_repro_v2", "obs2_roll_repro_v2_1"}:
                for metric_name in (
                    "fast_forward_reverse_rotation_penalty",
                    "fast_forward_backward_penalty",
                    "effort_penalty",
                ):
                    values[metric_name] = torch.as_tensor(
                        fast_forward_metrics[metric_name],
                        dtype=torch.float32,
                        device=self.device,
                    )
            if self.scratch_wr_v2:
                for metric_name in (
                    "scratch_wr_v2_z0_lift_ratio",
                    "scratch_wr_v2_z0_forward_ratio",
                    "scratch_wr_v2_z0_curl_ratio",
                    "scratch_wr_v2_z0_sync_score",
                    "scratch_wr_v2_z0_candidate_mask",
                    "scratch_wr_v2_z0_active",
                    "scratch_wr_v2_z0_penalty_scale",
                    "scratch_wr_v2_z0_dense_reward",
                    "scratch_wr_v2_progress_delta",
                ):
                    values[metric_name] = torch.as_tensor(
                        fast_forward_metrics[metric_name],
                        dtype=torch.float32,
                        device=self.device,
                    )
        if self.control_mode in {"tail_wave", "tail_wave_residual"}:
            wave_action = np.asarray(self._last_action_for_reward, dtype=np.float32)
            for index, metric_name in enumerate(
                ("wave_amplitude", "wave_center", "wave_width", "wave_hold", "wave_kp", "wave_kd")
            ):
                values[metric_name] = torch.as_tensor(
                    wave_action[:, 0, index:index + 1], dtype=torch.float32, device=self.device
                )
        if self.control_mode == "tail_wave_residual":
            metrics = self._scratch_wr_metrics_cache
            if metrics is None:
                metrics = {
                    metric_name: np.zeros((self.num_envs, 1), dtype=np.float32)
                    for metric_name in (
                        "scratch_wr_wave_torque_rms",
                        "scratch_wr_residual_torque_rms",
                        "scratch_wr_applied_residual_torque_rms",
                        "scratch_wr_total_torque_rms",
                        "scratch_wr_torque_clip_fraction",
                        "scratch_wr_residual_saturation_fraction",
                    )
                }
                if self.scratch_wr_v2:
                    for metric_name in (
                        "scratch_wr_v2_wave_ema_beta",
                        "scratch_wr_v2_wave_filter_delta_rms",
                        "scratch_wr_v2_applied_wave_amplitude",
                        "scratch_wr_v2_applied_wave_center",
                        "scratch_wr_v2_applied_wave_width",
                        "scratch_wr_v2_applied_wave_hold",
                        "scratch_wr_v2_applied_wave_kp",
                        "scratch_wr_v2_applied_wave_kd",
                    ):
                        metrics[metric_name] = np.zeros(
                            (self.num_envs, 1), dtype=np.float32
                        )
            values["scratch_wr_alpha"] = torch.full(
                (self.num_envs, 1),
                float(self.scratch_wr_alpha),
                dtype=torch.float32,
                device=self.device,
            )
            for metric_name, metric_value in metrics.items():
                values[metric_name] = torch.as_tensor(
                    metric_value, dtype=torch.float32, device=self.device
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
        self._fast_rollover_metrics_cache = None
        self._fast_forward_metrics_cache = None
        self._terrain_contact_cache = None
        self._tail_previous_potential = None
        self._tail_stage_success_latched[...] = False
        center = np.mean(self.pos, axis=1, keepdims=True)
        self._tail_previous_centered_shape = (self.pos - center).copy()
        self._tail_cumulative_rotation[...] = 0.0
        self._tail_previous_support_x = None
        self._tail_initial_com_x = np.real(center).copy()
        tail = self.pos[:, self.tail_index : self.tail_index + 1]
        self._tail_initial_relative_x = np.real(tail) - self._tail_initial_com_x
        self._fast_roll_phase[...] = 0
        self._fast_roll_phase_steps[...] = 0
        self._fast_roll_flip_count[...] = 0.0
        self._fast_roll_cycle_count[...] = 0.0
        self._fast_roll_cycle_start_rotation[...] = 0.0
        self._fast_roll_cycle_start_com_x = self._tail_initial_com_x.copy()
        self._fast_roll_cycle_start_support_x = None
        self._fast_roll_previous_potential = None
        self._fast_roll_positive_rotation[...] = 0.0
        self._fast_roll_absolute_rotation[...] = 0.0
        self._fast_forward_phase[...] = 0
        self._fast_forward_phase_steps[...] = 0
        self._fast_forward_launch_high_water[...] = 0.0
        self._scratch_wr_v2_sync_high_water[...] = 0.0
        self._fast_forward_launch_ready_steps[...] = 0
        self._fast_forward_roll_high_water[...] = 0.0
        self._fast_forward_event_count[...] = 0.0
        self._fast_forward_event_anchor_rotation[...] = 0.0
        self._fast_forward_event_anchor_com_x = self._tail_initial_com_x.copy()
        self._fast_forward_event_anchor_support_index = None
        self._fast_forward_event_anchor_support_valid[...] = False
        self._fast_forward_last_valid_support_index = None
        self._fast_forward_event_positive_rotation[...] = 0.0
        self._fast_forward_event_absolute_rotation[...] = 0.0
        self._fast_forward_episode_positive_rotation[...] = 0.0
        self._fast_forward_episode_absolute_rotation[...] = 0.0
        self._fast_forward_progress_age[...] = 0
        self._last_action_for_reward = np.zeros_like(self._last_action_for_reward, dtype=np.float32)
        self._previous_action_for_reward = np.zeros_like(self._previous_action_for_reward, dtype=np.float32)
        wave_midpoint = np.float32(0.5) * (
            self.tail_wave_action_low + self.tail_wave_action_high
        )
        self._scratch_wr_v2_filtered_wave_action[...] = wave_midpoint[None, None, :]
        self._scratch_wr_v2_filter_valid[...] = False
        self._scratch_wr_metrics_cache = None

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
        if self.control_mode == "tail_wave_residual":
            physics_action_np = action_np
            wave_filter_delta_rms = np.zeros((self.num_envs, 1), dtype=np.float32)
            if self.scratch_wr_v2:
                raw_wave = np.asarray(action_np[:, :, :6], dtype=np.float32)
                filtered_wave = np.where(
                    self._scratch_wr_v2_filter_valid[:, :, None],
                    self.scratch_wr_v2_wave_ema_beta
                    * self._scratch_wr_v2_filtered_wave_action
                    + (np.float32(1.0) - self.scratch_wr_v2_wave_ema_beta) * raw_wave,
                    raw_wave,
                ).astype(np.float32)
                wave_filter_delta_rms = np.sqrt(
                    np.mean((raw_wave - filtered_wave) ** 2, axis=(1, 2), keepdims=False)
                )[:, None].astype(np.float32)
                self._scratch_wr_v2_filtered_wave_action = filtered_wave
                self._scratch_wr_v2_filter_valid[...] = True
                physics_action_np = np.asarray(action_np, dtype=np.float32).copy()
                physics_action_np[:, :, :6] = filtered_wave
            (
                self.pos,
                self.vel,
                self.thdot,
                wave_tau,
                residual_tau,
                combined_tau,
                combined_unclipped,
            ) = self._physics_sim(physics_action_np, pos, vel, np.float32(self.scratch_wr_alpha))
            residual_action = action_np[:, 0, 6:]
            residual_low = self.scratch_wr_action_low[6:][None, :]
            residual_high = self.scratch_wr_action_high[6:][None, :]
            residual_span = np.maximum(residual_high - residual_low, np.float32(1e-6))
            normalized_residual = np.float32(2.0) * (residual_action - residual_low) / residual_span - np.float32(1.0)
            residual_rms = np.sqrt(np.mean(residual_tau ** 2, axis=1, keepdims=True))
            self._scratch_wr_metrics_cache = {
                "scratch_wr_wave_torque_rms": np.sqrt(np.mean(wave_tau ** 2, axis=1, keepdims=True)).astype(np.float32),
                "scratch_wr_residual_torque_rms": residual_rms.astype(np.float32),
                "scratch_wr_applied_residual_torque_rms": (
                    np.float32(self.scratch_wr_alpha) * residual_rms
                ).astype(np.float32),
                "scratch_wr_total_torque_rms": np.sqrt(np.mean(combined_tau ** 2, axis=1, keepdims=True)).astype(np.float32),
                "scratch_wr_torque_clip_fraction": np.mean(
                    np.abs(combined_unclipped) > np.float32(self.max_torque), axis=1, keepdims=True
                ).astype(np.float32),
                "scratch_wr_residual_saturation_fraction": np.mean(
                    np.abs(normalized_residual) >= np.float32(0.98), axis=1, keepdims=True
                ).astype(np.float32),
            }
            if self.scratch_wr_v2:
                applied_wave = np.asarray(
                    self._scratch_wr_v2_filtered_wave_action[:, 0, :], dtype=np.float32
                )
                self._scratch_wr_metrics_cache.update(
                    {
                        "scratch_wr_v2_wave_ema_beta": np.full(
                            (self.num_envs, 1),
                            self.scratch_wr_v2_wave_ema_beta,
                            dtype=np.float32,
                        ),
                        "scratch_wr_v2_wave_filter_delta_rms": wave_filter_delta_rms,
                        "scratch_wr_v2_applied_wave_amplitude": applied_wave[:, 0:1],
                        "scratch_wr_v2_applied_wave_center": applied_wave[:, 1:2],
                        "scratch_wr_v2_applied_wave_width": applied_wave[:, 2:3],
                        "scratch_wr_v2_applied_wave_hold": applied_wave[:, 3:4],
                        "scratch_wr_v2_applied_wave_kp": applied_wave[:, 4:5],
                        "scratch_wr_v2_applied_wave_kd": applied_wave[:, 5:6],
                    }
                )
        else:
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
        self._fast_rollover_metrics_cache = None
        self._fast_forward_metrics_cache = None
        self._terrain_contact_cache = None
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
