"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import os
import math
import numpy as np
import torch

# parse arguments
args = gymutil.parse_arguments(description = "DVRK Import Test",
                               custom_parameters=[
                                   {"name": "--num_envs",
                                    "type": int,
                                    "default": 1,
                                    "help": "Number of environments to create"},
                               ])

# Initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Load dvrk asset
asset_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../assets/urdf")
dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.armature = 0.01
asset_options.disable_gravity = True
asset_options.override_com = True
asset_options.override_inertia = True

print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
dvrk_asset = gym.load_asset(
    sim, asset_root, dvrk_asset_file, asset_options)

# get joint limits and ranges for dvrk
dvrk_dof_props = gym.get_asset_dof_properties(dvrk_asset)
dvrk_lower_limits = dvrk_dof_props['lower']
dvrk_upper_limits = dvrk_dof_props['upper']
print(dvrk_upper_limits)
print(dvrk_lower_limits)
dvrk_ranges = dvrk_upper_limits - dvrk_lower_limits
dvrk_mids = 0.5 * (dvrk_upper_limits + dvrk_lower_limits)
dvrk_num_dofs = len(dvrk_dof_props)

# set default DOF states
default_dof_state = np.zeros(dvrk_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"][:7] = dvrk_mids[:7]

# set DOF control properties (except grippers)
dvrk_dof_props["driveMode"][:8].fill(gymapi.DOF_MODE_EFFORT)
dvrk_dof_props["stiffness"][:8].fill(0.0)
dvrk_dof_props["damping"][:8].fill(0.0)

# set DOF control properties for grippers
dvrk_dof_props["driveMode"][8:].fill(gymapi.DOF_MODE_POS)
dvrk_dof_props["stiffness"][8:].fill(800.0)
dvrk_dof_props["damping"][8:].fill(40.0)

# Set up the env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default dvrk pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
pose.r = gymapi.Quat(0, 0, 0, 1)

print("Creating %d environments" % num_envs)

envs = []
hand_idxs = []

for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Add dvrk
    dvrk_handle = gym.create_actor(env, dvrk_asset, pose, "dvrk", i, 1)

    # Set initial DOF states
    gym.set_actor_dof_states(env, dvrk_handle, default_dof_state, gymapi.STATE_ALL)

    # Set DOF control properties
    gym.set_actor_dof_properties(env, dvrk_handle, dvrk_dof_props)

# Point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API to access and control the physics simulation
gym.prepare_sim(sim)

_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# DOF state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

_pos_control = torch.zeros((num_envs, dvrk_num_dofs), dtype=torch.float, device=args.graphics_device_id)
_effort_control = torch.zeros_like(_pos_control)

while not gym.query_viewer_has_closed(viewer):

    _effort_control = torch.rand((num_envs, dvrk_num_dofs), device=args.graphics_device_id) - 0.5
    print(_effort_control)

    gym.set_dof_position_target_tensor(
        sim,
        gymtorch.unwrap_tensor(_pos_control)
    )
    gym.set_dof_actuation_force_tensor(
        sim,
        gymtorch.unwrap_tensor(_effort_control)
    )

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
