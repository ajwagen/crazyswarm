import numpy as np
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from stable_baselines3.common.env_util import make_vec_env

class ControllerBackbone():
    def __init__(self, isSim, policy_config, isPPO = False):
        self.isSim = isSim
        self.mass = 0.032
        self.g = 9.8
        self.I = np.array([[3.144988,4.753588,4.640540],
                    [4.753588,3.151127,4.541223],
                    [4.640540,4.541223,7.058874]])*1e-5
        
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
            self.policy_name = "traj_random_zigzag_curriculum"
            self.config_filename = "trajectory_latency.py"

    def set_policy(self,):

        self.select_policy_configs()

        self.algo = RLAlgo.PPO
        self.eval_steps = 1000
        self.train_config = None

        self.algo_class = self.algo.algo_class()

        config = import_config(self.config_filename)
        if self.train_config is not None:
            self.train_config = import_config(self.train_config)
        else:
            self.train_config = config
        self.env = make_vec_env(self.task.env(), n_envs=8,
            env_kwargs={
                'config': self.train_config
            }
        )
        self.evalenv = self.task.env()(config=config)
        self.policy = self.algo_class.load(SAVED_POLICY_DIR / f'{self.policy_name}', self.env)
        self.prev_pos = 0.