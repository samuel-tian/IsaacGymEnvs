import numpy as np
import os, sys, pickle
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from .base.vec_task import VecTask

class SoftBody(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.cfg = cfg
        self.max_episode_length = self.cfg['env']['episodeLength']

        self.action_scale = self.cfg['env']['actionScale']
        self.dvrk_dof_noise = self.cfg['env']['dvrkDofNoise']

        self.cfg['env']['numObservations'] = 17
        self.cfg['env']['numActions'] = 9

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

        # setup viewer settings
        cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        
    def create_sim(self):
        
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        self._create_envs(self.num_envs, self.cfg['env']['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../assets")
        dvrk_asset_file = "urdf/dvrk_description/psm/psm_for_issacgym.urdf"
        soft_asset_file = "urdf/box.urdf"

        # configure and load dvrk asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        asset_options.override_com = True
        asset_options.override_inertia = True
        print("Loading asset '%s' from '%s'" % (dvrk_asset_file, asset_root))
        dvrk_asset = self.gym.load_asset(self.sim, asset_root, dvrk_asset_file, asset_options)

        self.num_dvrk_bodies = self.gym.get_asset_rigid_body_count(dvrk_asset)
        self.num_dvrk_dofs = self.gym.get_asset_dof_count(dvrk_asset)
        print ("num dvrk bodies: ", self.num_dvrk_bodies)
        print ("num dvrk dofs: ", self.num_dvrk_dofs)

        # set dvrk dof properties
        dvrk_dof_props = self.gym.get_asset_dof_properties(dvrk_asset)
        self.dvrk_dof_lower_limits = dvrk_dof_props['lower']
        self.dvrk_dof_upper_limits = dvrk_dof_props['upper']
        self.dvrk_effort_limits = dvrk_dof_props['effort']
        self.num_dofs = len(dvrk_dof_props)

        self.dvrk_dof_lower_limits = to_torch(self.dvrk_dof_lower_limits, device=self.device)
        self.dvrk_dof_upper_limits = to_torch(self.dvrk_dof_upper_limits, device=self.device)

        # set dvrk pose
        dvrk_pose = gymapi.Transform()
        dvrk_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        dvrk_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # configure and load deformable asset
        soft_thickness = 0.1
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.thickness = soft_thickness
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        print("Loading asset '%s' from '%s'" % (soft_asset_file, asset_root))
        soft_asset = self.gym.load_asset(self.sim, asset_root, soft_asset_file, asset_options)

        self.asset_soft_body_count = self.gym.get_asset_soft_body_count(soft_asset)
        self.asset_soft_materials = self.gym.get_asset_soft_materials(soft_asset)
        print('Soft Material Properties:')
        for i in range(self.asset_soft_body_count):
            mat = self.asset_soft_materials[i]
            print(f'(Body {i}) youngs: {mat.youngs} poissons: {mat.poissons} damping: {mat.damping}')

        # set soft object pose
        soft_pose = gymapi.Transform()
        soft_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        
        # setup env grid
        self.dvrk_handles = []
        self.soft_handles = []
        self.envs = []
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            dvrk_handle = self.gym.create_actor(env_ptr, dvrk_asset, dvrk_pose, "dvrk", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, dvrk_handle, dvrk_dof_props)

            soft_handle = self.gym.create_actor(env_ptr, soft_asset, soft_pose, "soft", i, 2, 0)

            self.envs.append(env_ptr)
            self.dvrk_handles.append(dvrk_handle)
            # self.soft_handles.append(soft_handle)

        self.init_data()

    def init_data(self):
        pass

    def compute_observations(self, env_ids = None):
        pass

    def reset_idx(self, env_ids):
        pass

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        pass

    def compute_reward(self, actions):
        pass
