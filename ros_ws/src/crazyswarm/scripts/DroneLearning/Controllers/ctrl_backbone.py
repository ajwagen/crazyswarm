import numpy as np
import yaml
import torch
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR, TEST_POLICY_DIR
from quadsim.learning.utils.adaptation_network import AdaptationNetwork
from stable_baselines3.common.env_util import make_vec_env

# importing MPPI
from quadrotor_torch import Quadrotor_torch
from param_torch import Param, Timer
import controller_torch
from pathlib import Path
import os

class ControllerBackbone():
    def __init__(self, isSim, policy_config, isPPO = False, adaptive=False):
        self.isSim = isSim
        self.mass = 0.034
        self.g = 9.8
        self.adaptive = adaptive
        # self.I = np.array([[3.144988, 4.753588, 4.640540],
        #                    [4.753588, 3.151127, 4.541223],
        #                    [4.640540, 4.541223, 7.058874]])*1e-5
        self.I = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])
        
        self.prev_t = None
        self.dt = None
        self.offset_pos = np.zeros(3)

        self.time_horizon = 10
        self.policy_config = policy_config

    def updateDt(self,t):

        if self.prev_t is None:
            self.dt = 0
        else:
            self.dt = t - self.prev_t
        
        if self.dt < 0:
            self.dt = 0

    def select_policy_configs(self,):
        self.log_scale = False
    # Naive Setpoint Tracking
        # No Adaptive Module
        if self.policy_config=="hover":
            self.task: DroneTask = DroneTask.HOVER
            self.policy_name = "hover_04k"
            self.config_filename = "default_hover.py"

        if self.policy_config == "yawflip":
            self.task: DroneTask = DroneTask.YAWFLIP_90
            self.policy_name = "yawflip_90"
            self.config_filename = "yawflip_latency.py"

        # With Adaptive module
        if self.policy_config=="hover_adaptive":
            self.task: DroneTask = DroneTask.HOVER
            self.policy_name = "hover_latency_adaptive"
            self.config_filename = "hover_latency_adaptive.py"
    
    # Trajectory tracking with feedforward/feedback
        if self.policy_config == "trajectory":
            self.task: DroneTask = DroneTask.TRAJFBFF
            # self.policy_name = "traj_fbff_h10_p1_3i"
            # self.policy_name = "traj_random_zigzag_curriculum"
            # self.policy_name = "ppo_base"
            # self.policy_name = "traj_mixed_reset_thresh_5000_body_frame_relative_fixed3_7500000_steps.zip"
            self.policy_name = "ppo-rl-zigzag_x_bf_rel.zip" # traj_mixed_reset_thresh_5000_body_frame_relative_fixed3_7500000_steps.zip
            self.config_filename = "trajectory_latency.py"
            self.body_frame = True
            self.relative = True
        
        if self.policy_config == "trajectory_2d":
            self.task: DroneTask = DroneTask.TRAJFBFF
            self.policy_name = "ppo-rl-zigzag_xy_bf_rel.zip" # traj_mixed2D_reset_thresh_5000_body_frame_relative_fixed3.zip
            self.config_filename = "trajectory_latency.py"
            self.body_frame = True
            self.relative = True
        
               
        if self.policy_config == "trajectory_2d_mass_adaptive":
            self.task: DroneTask = DroneTask.TRAJFBFF
            self.policy_name = "traj_mixed2D_mass_adaptive_symlog.zip"
            self.adaptive_policy_name = 'RMA_mass_nowind.pth'
            self.config_filename = "trajectory_mass_adaptive_low.py"
            self.body_frame = True
            self.relative = True
            self.log_scale = True

    def set_policy(self,):

        self.select_policy_configs()

        self.algo = RLAlgo.PPO
        # self.config = None

        self.algo_class = self.algo.algo_class()

        config = import_config(self.config_filename)
        # self.config = config
        self.env = self.task.env()(
            config=config,
            log_scale = self.log_scale,
            body_frame = self.body_frame,
            relative = self.relative
        )

        self.policy = self.algo_class.load(TEST_POLICY_DIR / f'{self.policy_name}', self.env)
        if self.adaptive == True:
            print("rwik")
            self.adaptive_policy = AdaptationNetwork(10, 1)
            self.adaptive_policy.load_state_dict(torch.load(TEST_POLICY_DIR / f'{self.adaptive_policy_name}', map_location='cuda:0'))
        self.prev_pos = 0.
    
    def set_MPPI_controller(self,):
        config_dir = config_dir = os.path.dirname(os.path.abspath(__file__)) + "/mppi_config"

        # config_dir = "/mnt/hdd/drones/crazyswarm/ros_ws/src/crazyswarm/scripts/DroneLearning/Controllers/mppi_config"
        # config_dir = "/home/guan/ya/rwik/drones/crazyswarm/ros_ws/src/crazyswarm/scripts/DroneLearning/Controllers/mppi_config"
        with open(config_dir + "/hover.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.param_MPPI = Param(config, MPPI=True)
        env_MPPI = Quadrotor_torch(self.param_MPPI, config)
        controller = controller_torch.MPPI_thrust_omega(env_MPPI, config)
        
        return controller

    def set_BC_policy(self, ):
        from imitation.algorithms import bc
        # bc_policy_name = 'ppo_mppi_bc'
        # bc_policy_name = 'ppo-mppi_zigzag_bf_rel_bc'
        # bc_policy_name = 'ppo-mppi_zigzag_bc'
        bc_policy_name = 'ppo-mppi_zigzag_xy_bc'
        self.body_frame = False
        self.relative = False
        self.bc_policy = bc.reconstruct_policy(TEST_POLICY_DIR / f'{bc_policy_name}')