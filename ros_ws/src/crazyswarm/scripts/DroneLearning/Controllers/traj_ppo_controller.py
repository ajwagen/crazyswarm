import numpy as np

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env

# from quadsim.control import Controller
def add2npQueue(array, new):
    array[0:-1] = array[1:]
    array[-1] = new
    return array

class PPOController_trajectory():
  def __init__(self,isSim, policy_config="trajectory"):
    super().__init__()

    self.isSim = isSim
    self.policy_config = policy_config
    self.mass = 0.027
    self.g = 9.8

    self.prev_t = None
    self.offset_pos = np.zeros(3)
    self.set_policy()
    self.time_horizon = 10

    self.trajectories = None
  
  def select_policy_configs(self,):

    if self.policy_config == "trajectory":
      self.task: DroneTask = DroneTask.TRAJFBFF
      self.policy_name = "traj_fbff_h10_p1_3i"
      self.config_filename = "trajectory_latency.py"

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
    
    pos = state.pos - self.offset_pos
    vel = state.vel
    rot = state.rot

    quat = rot.as_quat() 

    obs = np.hstack((pos,vel,quat))

    if fl==0:
       obs_=np.zeros((self.time_horizon+1)*3+10)
    else:
        ff_terms = [ref_func(t + 3 * i * dt)[0].pos for i in range(self.time_horizon)]
        obs_ = np.hstack([obs, obs[0:3] - ref_func(t)[0].pos] + ff_terms)


    action, _states = self.policy.predict(obs_, deterministic=True)

    action[0]+=self.g
    self.prev_pos = pos.copy()
    return action[0], action[1:]
