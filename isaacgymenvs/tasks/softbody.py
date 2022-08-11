import numpy as np
import os, sys, pickle
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask

class SoftBody(VecTask):

    ROBOT_Z_OFFSET = 0.25

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        
        self.cfg = cfg
        self.max_episode_length = self.cfg['env']['episodeLength']

        self.action_scale = self.cfg['env']['actionScale']
        self.dvrk_dof_noise = self.cfg['env']['dvrkDofNoise']

        self.cfg['env']['numObservations'] = 20
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
        cam_pos = gymapi.Vec3(1.0, 1.0, 1.0 + self.ROBOT_Z_OFFSET)
        cam_target = gymapi.Vec3(0.0, 0.0, self.ROBOT_Z_OFFSET)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.refresh(torch.arange(self.num_envs, device=self.device))
        
        
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
        self.dvrk_dof_lower_limits = np.array(dvrk_dof_props['lower'])
        self.dvrk_dof_upper_limits = np.array(dvrk_dof_props['upper'])
        self.dvrk_effort_limits = np.array(dvrk_dof_props['effort'])
        self.cmd_limit = self.dvrk_effort_limits[:8]
        self.num_dofs = len(dvrk_dof_props)

        dvrk_dof_props["driveMode"][:].fill(gymapi.DOF_MODE_EFFORT)
        dvrk_dof_props["stiffness"][:].fill(100.0)
        dvrk_dof_props["damping"][:].fill(100.0)
        dvrk_dof_props["driveMode"][8:].fill(gymapi.DOF_MODE_POS)
        dvrk_dof_props["stiffness"][8:].fill(800.0)
        dvrk_dof_props["damping"][8:].fill(40.0)

        self.dvrk_default_dof_pos = np.zeros((self.num_dofs), dtype=np.float32)
        self.dvrk_default_dof_pos[:-2] = (0.5 * (self.dvrk_dof_lower_limits + self.dvrk_dof_upper_limits))[:-2]

        # set dvrk pose
        dvrk_pose = gymapi.Transform()
        dvrk_pose.p = gymapi.Vec3(0.0, 0.0, self.ROBOT_Z_OFFSET)
        dvrk_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # configure and load deformable asset
        soft_thickness = 0.001
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.thickness = soft_thickness
        asset_options.disable_gravity = True
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
        soft_pose.p = gymapi.Vec3(0.0, 0.5, 0.01)
        
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
            self.soft_handles.append(soft_handle)

        self.init_data()
        

    def init_data(self):
        # initialize states
        self.states = {}
        self.q = np.zeros((self.num_envs, self.num_dofs), dtype=np.float32)
        self.eef_state = np.zeros((self.num_envs), dtype=gymapi.RigidBodyState)
        self.eef_pose = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.eef_quat = np.zeros((self.num_envs, 4), dtype=np.float32)
        self.eef_vel = np.zeros((self.num_envs, 6), dtype=np.float32)

        _particle_state = self.gym.acquire_particle_state_tensor(self.sim)
        self.particle_state = gymtorch.wrap_tensor(_particle_state).view(self.num_envs, -1, 11)

        self.refresh(torch.arange(self.num_envs, device=self.device))

        self.init_soft_state = torch.clone(self.particle_state)
        
        # initialize actions
        self.pos_control = np.zeros((self.num_envs, self.num_dofs), dtype=np.float32)
        self.effort_control = np.zeros_like(self.pos_control)

        # initialize control
        self.arm_control = self.effort_control[:, :8]
        self.gripper_control = self.pos_control[:, 8:10]

        self.global_indices = torch.arange(self.num_envs * 2,
                                           dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)


    def refresh(self, env_ids):
        self.gym.refresh_particle_state_tensor(self.sim)
        
        self.rigid_body_state = self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL)
        self.dvrk_dof_state = self.gym.get_vec_actor_dof_states(self.envs, self.dvrk_handles, gymapi.STATE_ALL)

        for i in env_ids:
            # reset q
            for j in range(self.num_dofs):
                self.q[i][j] = self.dvrk_dof_state[i][j]['pos']

            # reset eef_state
            handle = self.gym.find_actor_rigid_body_index(self.envs[i], self.dvrk_handles[i], "psm_tool_yaw_link", gymapi.DOMAIN_SIM)
            self.eef_state[i] = self.rigid_body_state[handle]
            self.eef_pose[i][:] = np.array([self.eef_state[i]['pose']['p']['x'],
                                            self.eef_state[i]['pose']['p']['y'],
                                            self.eef_state[i]['pose']['p']['z']])
            self.eef_quat[i][:] = np.array([self.eef_state[i]['pose']['r']['x'],
                                            self.eef_state[i]['pose']['r']['y'],
                                            self.eef_state[i]['pose']['r']['z'],
                                            self.eef_state[i]['pose']['r']['w']])
            self.eef_vel[i][:] = np.array([self.eef_state[i]['vel']['linear']['x'],
                                           self.eef_state[i]['vel']['linear']['y'],
                                           self.eef_state[i]['vel']['linear']['z'],
                                           self.eef_state[i]['vel']['angular']['x'],
                                           self.eef_state[i]['vel']['angular']['y'],
                                           self.eef_state[i]['vel']['angular']['z']])

        # update non tensors
        self.states.update({
            'q': torch.tensor(self.q[:, :]),
            'eef_pos': torch.tensor(self.eef_pose[:]),
            'eef_quat': torch.tensor(self.eef_quat[:]),
            'eef_vel': torch.tensor(self.eef_vel[:]),

            'soft_pos': self.particle_state[:, -1, :3],
        })


    def compute_observations(self):
        self.refresh(torch.arange(self.num_envs, device=self.device))
        obs = ["eef_pos", "eef_quat", "q", "soft_pos"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        
        return self.obs_buf
    

    def reset_idx(self, env_ids):
        print("RESETING ENVIRONMENTS", env_ids)

        success_envs = []
        for env_id in env_ids:
            if self.progress_buf[env_id] < self.max_episode_length - 1:
                success_envs.append(env_id)
        if success_envs:
            print("SUCCESSFUL", success_envs)

        reset_noise = np.random.rand(len(env_ids), self.num_dofs)
        pos = np.clip(
            self.dvrk_default_dof_pos + self.dvrk_dof_noise * 2.0 * (reset_noise - 0.5),
            self.dvrk_dof_lower_limits,
            self.dvrk_dof_upper_limits
        )
        pos[:, -2:] = self.dvrk_default_dof_pos[-2:]

        self.pos_control[env_ids, :] = pos
        self.effort_control[env_ids, :] = np.zeros_like(pos)

        zero_velocity = np.zeros((self.num_envs, self.num_dofs), dtype=np.float32)

        # deploy updates
        for i in range(len(env_ids)):
            env = env_ids[i]
            for j in range(self.num_dofs):
                self.dvrk_dof_state[env][j]['pos'] = pos[i][j]
                self.dvrk_dof_state[env][j]['vel'] = 0

            self.gym.set_actor_dof_position_targets(self.envs[env], self.dvrk_handles[env], self.pos_control[env])
            self.gym.set_actor_dof_velocity_targets(self.envs[env], self.dvrk_handles[env], zero_velocity[env])
            self.gym.apply_actor_dof_efforts(self.envs[env], self.dvrk_handles[env], self.effort_control[env])
            self.gym.set_actor_dof_states(self.envs[env], self.dvrk_handles[env], self.dvrk_dof_state[env], gymapi.STATE_ALL)

        # reset soft body
        soft_ids = self.global_indices[env_ids, 1].flatten()
        # print("actor indices")
        # print(self.gym.find_actor_index(self.envs[0], "soft", gymapi.DOMAIN_SIM))
        # print(self.gym.find_actor_index(self.envs[1], "soft", gymapi.DOMAIN_SIM))
        self.gym.set_particle_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.init_soft_state),
            gymtorch.unwrap_tensor(soft_ids),
            len(soft_ids)
        )
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # control arm
        u_arm = u_arm * self.cmd_limit / self.action_scale
        self.arm_control[:, :] = u_arm

        # control gripper
        u_fingers = np.zeros_like(self.gripper_control)
        u_fingers[:, 0] = np.where(u_gripper >= 0.0,
                                   self.dvrk_dof_upper_limits[-2].item(),
                                   self.dvrk_dof_lower_limits[-2].item())
        u_fingers[:, 1] = np.where(u_gripper >= 0.0,
                                   self.dvrk_dof_upper_limits[-1].item(),
                                   self.dvrk_dof_lower_limits[-1].item())
        self.gripper_control[:, :] = u_fingers

        for i in range(self.num_envs):
            self.gym.set_actor_dof_position_targets(self.envs[i], self.dvrk_handles[i], self.pos_control[i])
            self.gym.apply_actor_dof_efforts(self.envs[i], self.dvrk_handles[i], self.effort_control[i])
        
        
    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
    

    def compute_reward(self, actions):
        target = self.init_soft_state[0, -1, :3]
        target = torch.clone(target)
        target[2] = target[2] + 0.5
        self.rew_buf[:], self.reset_buf[:] = compute_dvrk_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.states,
            target,
            self.max_episode_length
        )

@torch.jit.script
def compute_dvrk_reward(reset_buf, progress_buf, actions, states, target, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, float) -> Tuple[Tensor, Tensor]

    dvrk_pos = states['eef_pos']
    dvrk_vel = states['eef_vel']
    current_soft_pos = states['soft_pos']
    target_soft_pos = target

    d_to_soft_reward = torch.norm(dvrk_pos - current_soft_pos, dim=-1)
    vel_reward = torch.norm(dvrk_vel, dim=-1)
    soft_to_target_reward = torch.norm(current_soft_pos - target_soft_pos, dim=-1)

    dvrk_to_soft_scale = -10
    zero_vel_scale = -1
    soft_to_target_scale = -10

    rewards = (dvrk_to_soft_scale * d_to_soft_reward +
               zero_vel_scale * vel_reward +
               soft_to_target_scale * soft_to_target_reward)

    finished_epsilon = 0.1
    reset = torch.where((progress_buf >= max_episode_length - 1) | (soft_to_target_reward < finished_epsilon),
                        torch.ones_like(reset_buf), reset_buf)

    return rewards, reset
