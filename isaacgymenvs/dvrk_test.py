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
args.physics_engine = gymapi.SIM_FLEX

# Initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 10
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.75
sim_params.flex.warm_start = 0.4
sim_params.flex.shape_collision_margin = 0.1
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

# enable Von-Mises stress visualization
sim_params.stress_visualization = True
sim_params.stress_visualization_min = 0.0
sim_params.stress_visualization_max = 1.e+5

sim_params.use_gpu_pipeline = False
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

asset_root = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../assets/")
dvrk_asset_file = "urdf/dvrk_description/psm/psm_for_issacgym.urdf"
# dvrk_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
soft_asset_file = "urdf/box.urdf"

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

soft_thickness = 0.1
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.thickness = soft_thickness
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
soft_asset = gym.load_asset(
    sim, asset_root, soft_asset_file, asset_options)

asset_soft_body_count = gym.get_asset_soft_body_count(soft_asset)
asset_soft_materials = gym.get_asset_soft_materials(soft_asset)
print('Soft Material Properties:')
for i in range(asset_soft_body_count):
    mat = asset_soft_materials[i]
    print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')

# get joint limits and ranges for dvrk
dvrk_dof_props = gym.get_asset_dof_properties(dvrk_asset)
dvrk_lower_limits = dvrk_dof_props['lower']
dvrk_upper_limits = dvrk_dof_props['upper']
for i in (zip(dvrk_lower_limits, dvrk_upper_limits)):
    print(i)
dvrk_mids = 0.5 * (dvrk_upper_limits + dvrk_lower_limits)
dvrk_num_dofs = len(dvrk_dof_props)

# set default DOF states
default_dof_state = np.zeros(dvrk_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"][:] = dvrk_mids[:]

# set DOF control properties (except grippers)
dvrk_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
dvrk_dof_props["stiffness"][:].fill(100.0)
dvrk_dof_props["damping"][:].fill(100.0)

# set DOF control properties for grippers
dvrk_dof_props["driveMode"][8:].fill(gymapi.DOF_MODE_POS)
dvrk_dof_props["stiffness"][8:].fill(800.0)
dvrk_dof_props["damping"][8:].fill(40.0)

# Set up the env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 10.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default dvrk pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
pose.r = gymapi.Quat(0, 0, 0, 1)

soft_pose = gymapi.Transform()
soft_pose.p = gymapi.Vec3(0, 0, 0)
soft_pose.r = gymapi.Quat(0, 0, 0, 1)

print("Creating %d environments" % num_envs)

envs = []
handles = []

for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Add dvrk
    dvrk_handle = gym.create_actor(env, dvrk_asset, pose, "dvrk", i, 1)
    handles.append(dvrk_handle)
    gym.set_actor_dof_states(env, dvrk_handle, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_properties(env, dvrk_handle, dvrk_dof_props)

    # add soft object
    # soft_handle = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 2)

# Point camera at middle env
cam_pos = gymapi.Vec3(1, 1, 1)
cam_target = gymapi.Vec3(0, 0, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

gym.prepare_sim(sim)

while not gym.query_viewer_has_closed(viewer):

    arr = np.array([5] * dvrk_num_dofs, dtype=np.float32)
    gym.apply_actor_dof_efforts(envs[0], handles[0], arr)

    print(gym.get_env_rigid_contact_forces(envs[0]))

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)

    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
