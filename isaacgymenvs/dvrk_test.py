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

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0
    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

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
sim_params.substeps = 3
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5
sim_params.flex.contact_regularization = 0.001
sim_params.flex.shape_collision_distance = 0.001
sim_params.flex.shape_collision_margin = 0.001
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

# enable Von-Mises stress visualization
# sim_params.stress_visualization = True
# sim_params.stress_visualization_min = 0.0
# sim_params.stress_visualization_max = 1.e+5

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
soft_asset_file = "urdf/organs/liver_gall.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.flip_visual_attachments = False
asset_options.armature = 0.01
asset_options.disable_gravity = True
asset_options.override_com = True
asset_options.override_inertia = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
dvrk_asset = gym.load_asset(
    sim, asset_root, dvrk_asset_file, asset_options)

soft_thickness = 0.001
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.thickness = soft_thickness
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
soft_asset = gym.load_asset(
    sim, asset_root, soft_asset_file, asset_options)

# Set up the env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 10.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default dvrk pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.3, 0, 0.02)
pose.r = gymapi.Quat(0, 0, 0, 1.0)

soft_pose = gymapi.Transform()
soft_pose.p = gymapi.Vec3(0.0, 0.7, 0.2)
rot = axisangle2quat(torch.tensor([np.pi/3, 0, 0], dtype=torch.float32))
soft_pose.r = gymapi.Quat(*rot.numpy().tolist())

print("Creating %d environments" % num_envs)

envs = []

for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add soft object
    soft_handle = gym.create_actor(env, soft_asset, soft_pose, "soft", i, 2)

# Point camera at middle env
cam_pos = gymapi.Vec3(1, 1, 1)
cam_target = gymapi.Vec3(0, 0, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

gym.prepare_sim(sim)

while not gym.query_viewer_has_closed(viewer):

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
