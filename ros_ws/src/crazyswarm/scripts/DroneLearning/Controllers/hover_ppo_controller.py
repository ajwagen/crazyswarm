import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env
from quadsim.learning.utils.adaptation_network import AdaptationNetwork

# from quadsim.control import Controller
def add2npQueue(array, new):
    array[0:-1] = array[1:]
    array[-1] = new
    return array

class PPOController():
  def __init__(self,isSim, policy_config="hover"):
    super().__init__()
    # self.model = model

    self.isSim = isSim
    self.policy_config = policy_config
    self.mass = 0.027
    self.g = 9.8
    self.offset_pos = np.zeros(3)
    self.trajectories = None

    self.prev_t = None
    self.set_policy()
  
  def select_policy_configs(self,):
    if self.policy_config=="hover":
      self.task: DroneTask = DroneTask.HOVER
      self.policy_name = "hover_04k"
      self.config_filename = "default_hover.py"

    if self.policy_config == "yawflip":
      self.task: DroneTask = DroneTask.YAWFLIP_90
      self.policy_name = "yawflip_90"
      self.config_filename = "yawflip_latency.py"

  def set_policy(self,):

    # self.task: DroneTask = DroneTask.YAWFLIP
    # self.policy_name = "yawflip_latency_ucost"
    # self.config_filename = "yawflip.py"
    self.select_policy_configs()

    self.algo = RLAlgo.PPO
    self.eval_steps = 1000
    # viz = True
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

    self.policy = self.algo_class.load(SAVED_POLICY_DIR / f'{self.policy_name}.zip', self.env)
    self.prev_pos = 0.


  def response(self, t, state, ref , ref_func, fl=1):
    """
        Given a time t and state state, return body z force and torque (body-frame).

        State is defined in rigid_body.py and includes pos, vel, rot, and ang.

        The reference is available using self.ref (defined in flatref.py)
        and contains pos, vel, acc, jerk, snap, yaw, yawvel, and yawacc,
        which are all functions of time.

        self.model contains quadrotor model parameters such as mass, inertia, and gravity.
        See models.py.

        TODO Implement a basic quadrotor controller.
        E.g. you may want to compute position error using something like state.pos - self.ref.pos(t).

        You can test your controller by running main.py.
    """
    if self.prev_t is None:
      dt = 0
    else:
      dt = t - self.prev_t
    
    if fl:
      self.prev_t = t

    # the drone always tries to orient to (0,0,0)
    pos = state.pos - ref.pos
    vel = state.vel
    rot = state.rot

    quat = rot.as_quat() 

    obs = np.hstack((pos,vel,quat))
    action, _states = self.policy.predict(obs, deterministic=True)


    ################################
    # # Gradient (gain) calculation for hovering
    # th_obs,_ = self.policy.policy.obs_to_tensor(obs)
    # j = jacobian(self.policy.policy._predict,(th_obs))
    # j = j[0,:,0,:].detach().cpu().numpy()
    # print(j)
    # exit()
    ################################
# 

# [[  3.67305589   1.38716006  -9.53558922   4.38564968   1.68557179 -10.64670753  -7.13586092  15.51110363 -12.54623222   0.24622881]
#  [  0.90983611  10.75347614   4.6693573   -0.83937329  10.08096027   2.01146364 -28.32143784  -2.30262518  -6.37237597  -0.67694801]
#  [ -9.83611393  -4.55986023   3.62569857 -13.52033424  -1.33899236   2.54128242   5.33164978 -43.40154266   1.18937016   0.59184641]
#  [ -0.34359568  -0.66659546   1.27601814   0.58130348  -1.72741628   1.22041595   4.60577106   1.54117    -16.57507133   0.13079186]]
# 
    action[0]+=self.g
    self.prev_pos = pos.copy()
    return action[0], action[1:]

class PPOController_adaptive():
  def __init__(self,isSim, policy_config="hover"):
    super().__init__()
    # self.model = model

    self.isSim = isSim
    self.policy_config = policy_config
    self.mass = 0.027
    self.g = 9.8
    self.offset_pos = np.zeros(3)
    self.trajectories = None

    self.prev_t = None
    self.set_policy()
    self.init_adaptive_network()

  def init_adaptive_network(self,):
    self.e_dims = 3
    self.history = None
    self.time_horizon = 50
    self.e_pred = np.zeros(self.e_dims)[None, :]

    adaptation_network_policy_name = "hover_wind_obs_noise_adapt"
    self.adaptation_network = AdaptationNetwork(input_dims=14, e_dims=self.e_dims)
    adaptation_network_state_dict = torch.load(Path().absolute() / 'saved_policies'/ adaptation_network_policy_name)
    self.adaptation_network.load_state_dict(adaptation_network_state_dict)

  def select_policy_configs(self,):
    if self.policy_config=="hover":
      self.task: DroneTask = DroneTask.HOVER
      self.policy_name = "hover_04k"
      self.config_filename = "default_hover.py"

    if self.policy_config == "yawflip":
      self.task: DroneTask = DroneTask.YAWFLIP_90
      self.policy_name = "yawflip_90"
      self.config_filename = "yawflip_latency.py"

  def set_policy(self,):

    # self.task: DroneTask = DroneTask.YAWFLIP
    # self.policy_name = "yawflip_latency_ucost"
    # self.config_filename = "yawflip.py"
    self.select_policy_configs()

    self.algo = RLAlgo.PPO
    self.eval_steps = 1000
    # viz = True
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

    self.policy = self.algo_class.load(SAVED_POLICY_DIR / f'{self.policy_name}.zip', self.env)
    self.prev_pos = 0.


  def response(self, t, state, ref , ref_func, fl=1):
    """
        Given a time t and state state, return body z force and torque (body-frame).

        State is defined in rigid_body.py and includes pos, vel, rot, and ang.

        The reference is available using self.ref (defined in flatref.py)
        and contains pos, vel, acc, jerk, snap, yaw, yawvel, and yawacc,
        which are all functions of time.

        self.model contains quadrotor model parameters such as mass, inertia, and gravity.
        See models.py.

        TODO Implement a basic quadrotor controller.
        E.g. you may want to compute position error using something like state.pos - self.ref.pos(t).

        You can test your controller by running main.py.
    """
    # if self.hi
    
    
    if self.prev_t is None:
      dt = 0
    else:
      dt = t - self.prev_t
    
    if fl:
      self.prev_t = t

    # the drone always tries to orient to (0,0,0)
    pos = state.pos - ref.pos
    vel = state.vel
    rot = state.rot

    quat = rot.as_quat() 

    obs = np.hstack((pos,vel,quat))

    # print
    action, _states = self.policy.predict(obs, deterministic=True)
    obs = np.r_[obs, self.e_pred[0]]

    if self.history is None:
      self.history = np.r_[obs[:-self.e_dims],action][None, :].T
    else:
      self.history = np.c_[self.history, np.r_[obs[:-self.e_dims], action][None, :].T]
    
    if self.history.shape[1] >= self.time_horizon:
      self.e_pred = self.adaptation_network(torch.from_numpy(self.history[None, ..., -self.time_horizon :]).type(torch.FloatTensor)).detach().numpy()
    
    
    ################################
    # # Gradient (gain) calculation for hovering
    # th_obs,_ = self.policy.policy.obs_to_tensor(obs)
    # j = jacobian(self.policy.policy._predict,(th_obs))
    # j = j[0,:,0,:].detach().cpu().numpy()
    # print(j)
    # exit()
    ################################
# 

# [[  3.67305589   1.38716006  -9.53558922   4.38564968   1.68557179 -10.64670753  -7.13586092  15.51110363 -12.54623222   0.24622881]
#  [  0.90983611  10.75347614   4.6693573   -0.83937329  10.08096027   2.01146364 -28.32143784  -2.30262518  -6.37237597  -0.67694801]
#  [ -9.83611393  -4.55986023   3.62569857 -13.52033424  -1.33899236   2.54128242   5.33164978 -43.40154266   1.18937016   0.59184641]
#  [ -0.34359568  -0.66659546   1.27601814   0.58130348  -1.72741628   1.22041595   4.60577106   1.54117    -16.57507133   0.13079186]]
# 
    action[0]+=self.g
    self.prev_pos = pos.copy()
    return action[0], action[1:]