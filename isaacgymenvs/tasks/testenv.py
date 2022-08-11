import numpy as np
import os, sys, pickle
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from .base.vec_task import VecTask

ROBOT_Z_OFFSET = 0.25

class TestEnv(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.cfg['sim']['dt'] = 1.0 / 60.0
        self.max_episode_length = self.cfg['env']['episodeLength']

        self.action_scale = self.cfg['env']['actionScale']
        self.dvrk_dof_noise = self.cfg['env']['dvrkDofNoise']
        
        # observations include: eef_pose (7) + q (10)
        self.cfg['env']['numObservations'] = 17
        # actions include: joint torques (8) + bool gripper (1)
        self.cfg['env']['numActions'] = 9

        # values to be filled in at runtime
        self.states = {}
        self.handles = {}
        self.actions = None
        self.num_dofs = None
        self.actions = None
        self.num_dvrk_bodies = None
        self.num_dvrk_dofs = None
        self.dvrk_default_dof_pos = None

        # tensor placeholders
        self._root_state = None # state of root body (n_envs, 13)
        self._rigid_body_state = None # state of all rigid bodies (n_envs, n_bodies, 13)
        self._dof_state = None # state of all joints (n_envs, n_dof)
        self._q = None # joint positions (n_envs, n_dof)
        self._qd = None # joint velocities (n_envs, n_dof)
        self._eef_state = None
        self._arm_control = None # tensor buffer for controlling arm
        self._gipper_control = None # tensor buffer for controlling gipper
        self._pos_control = None # position actions
        self._effort_control = None # torque actions
        self._dvrk_effort_limits = None # actuator effort limits for dvrk
        self._global_indices = None
        
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        # dvrk defaults
        # self.dvrk_default_dof_pos = to_torch(
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0.035, 0.035], device=self.device
        # )

        # set control limits
        self.cmd_limit = self._dvrk_effort_limits[:8].unsqueeze(0)

        # setup viewer settings
        cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # reset tensors
        self._refresh()

    def create_sim(self):
        # implement sim set up and environment creation here
        #    - set up-axis
        #    - call super().create_sim with device args (see docstring)
        #    - create ground plane
        #    - set up environments

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg['env']['envSpacing'], int(np.sqrt(self.num_envs)))
        
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # setup env grid
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../assets/urdf")
        dvrk_asset_file = "dvrk_description/psm/psm_for_issacgym.urdf"
        soft_asset_file = "box.urdf"

        # load dvrk asset
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
        # asset_options.max_angular_velocity = 4000.0
        dvrk_asset = self.gym.load_asset(self.sim, asset_root, dvrk_asset_file, asset_options)

        dvrk_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 0, 5000, 5000], dtype=torch.float, device=self.device)
        dvrk_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # load soft object asset
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        soft_asset = self.gym.load_asset(self.sim, asset_root, soft_asset_file, asset_options)

        self.num_dvrk_bodies = self.gym.get_asset_rigid_body_count(dvrk_asset)
        self.num_dvrk_dofs = self.gym.get_asset_dof_count(dvrk_asset)
        print ("num dvrk bodies: ", self.num_dvrk_bodies)
        print ("num dvrk dofs: ", self.num_dvrk_dofs)

        # set dvrk dof properties
        dvrk_dof_props = self.gym.get_asset_dof_properties(dvrk_asset)
        self.dvrk_dof_lower_limits = dvrk_dof_props['lower']
        self.dvrk_dof_upper_limits = dvrk_dof_props['upper']
        self._dvrk_effort_limits = dvrk_dof_props['effort']
        self.dvrk_mids = 0.5 * (self.dvrk_dof_lower_limits + self.dvrk_dof_upper_limits)
        self.dvrk_mids = to_torch(self.dvrk_mids, device=self.device)
        self.num_dofs = len(dvrk_dof_props)

        self.dvrk_default_dof_pos = torch.zeros((self.num_dofs), device=self.device)
        self.dvrk_default_dof_pos[:-2] = self.dvrk_mids[:-2]

        for i in range(self.num_dvrk_dofs):
            dvrk_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i >= 8 else gymapi.DOF_MODE_EFFORT
            dvrk_dof_props['stiffness'][i] = dvrk_dof_stiffness[i]
            dvrk_dof_props['damping'][i] = dvrk_dof_damping[i]

        self.dvrk_dof_lower_limits = to_torch(self.dvrk_dof_lower_limits, device=self.device)
        self.dvrk_dof_upper_limits = to_torch(self.dvrk_dof_upper_limits, device=self.device)
        self._dvrk_effort_limits = to_torch(self._dvrk_effort_limits, device=self.device)
        self.dvrk_dof_speed_scales = torch.ones_like(self.dvrk_dof_lower_limits)
        self.dvrk_dof_speed_scales[[-2, -1]] = 0.1
        dvrk_dof_props['effort'][-2] = 200
        dvrk_dof_props['effort'][-1] = 200

        # set soft object dof properties
        soft_dof_props = self.gym.get_asset_dof_properties(soft_asset)
        
        # asset orientation
        dvrk_pose = gymapi.Transform()
        dvrk_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        dvrk_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        soft_start_pose = gymapi.Transform()
        soft_start_pose.p = gymapi.Vec3(1.0, 1.0, 1.0)

        # set up the env grid
        self.dvrk_handles = []
        self.soft_objs = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            dvrk_handle = self.gym.create_actor(env_ptr, dvrk_asset, dvrk_pose, "dvrk", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, dvrk_handle, dvrk_dof_props)

            soft_handle = self.gym.create_actor(env_ptr, soft_asset, soft_start_pose, "soft", i, 2, 0)
            self.gym.set_actor_dof_properties(env_ptr, soft_handle, soft_dof_props)

            self.envs.append(env_ptr)
            self.dvrk_handles.append(dvrk_handle)
            self.soft_objs.append(soft_handle)

        self.init_data()

    def init_data(self):
        # setup sim handles
        env_ptr = self.envs[0]
        dvrk_handle = 0
        self.handles = {
            "yaw_link" : self.gym.find_actor_rigid_body_handle(env_ptr, dvrk_handle, "psm_tool_yaw_link")
        }

        # setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self.dvrk_dof_state = self._dof_state.view(self.num_envs, -1, 2)[:, :self.num_dvrk_dofs]
        self.dvrk_dof_pos = self.dvrk_dof_state[..., 0]
        self.dvrk_dof_vel = self.dvrk_dof_state[..., 1]
        
        self._eef_state = self._rigid_body_state[:, self.handles["yaw_link"], :]
        
        # initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # initialize control
        self._arm_control = self._effort_control[:, :8]
        self._gripper_control = self._pos_control[:, 8:10]

        # initialize indices
        self._global_indices = torch.arange(self.num_envs * 1,
                                            dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)


    def _update_states(self):
        self.states.update({
            "q": self._q[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self._update_states()

    def compute_observations(self, env_ids = None):
        self._refresh()
        obs = ["eef_pos", "eef_quat", "q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf
        
    def reset_idx(self, env_ids):
        print("============RESETING=========")
        reset_noise = torch.rand((len(env_ids), self.num_dofs), device=self.device)
        pos = tensor_clamp(
            self.dvrk_default_dof_pos.unsqueeze(0) +
            self.dvrk_dof_noise * 2.0 * (reset_noise - 0.5),
            self.dvrk_dof_lower_limits.unsqueeze(0),
            self.dvrk_dof_upper_limits)
        
        pos[:, -2:] = self.dvrk_default_dof_pos[-2:]

        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32)
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32)
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        self._refresh()

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions

        self.actions = actions.clone().to(self.device)

        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # control arm
        u_arm = u_arm * self.cmd_limit / self.action_scale
        self._arm_control[:, :] = u_arm

        # control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0,
                                      self.dvrk_dof_upper_limits[-2].item(),
                                      self.dvrk_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0,
                                      self.dvrk_dof_upper_limits[-1].item(),
                                      self.dvrk_dof_lower_limits[-1].item())
        self._gripper_control[:, :] = u_fingers

        # deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
        
    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations

        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        target_pos = torch.tensor([0.2, 0.2, 0.2], device=self.device)
        # target_vel = torch.tensor([0, 0, 0, 0, 0, 0], device=self.device)
        d_pos = torch.norm(self.states["eef_pos"] - target_pos, dim=-1)
        # d_vel = torch.norm(self.states["eef_vel"] - target_vel, dim=-1)
        d = d_pos # + d_vel
        
        rewards = -d

        reset_buf = torch.where((self.progress_buf >= self.max_episode_length - 1) | (d < 0.05),
                                torch.ones_like(self.reset_buf), self.reset_buf)

        self.rew_buf[:], self.reset_buf[:] = rewards, reset_buf
