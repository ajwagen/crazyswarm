import numpy as np

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from stable_baselines3.common.env_util import make_vec_env

# from quadsim.control import Controller
def add2npQueue(array, new):
    array[0:-1] = array[1:]
    array[-1] = new
    return array

class PPOController():
  def __init__(self,isSim):
    super().__init__()
    # self.model = model

    self.isSim = isSim

    self.mass = 0.027
    self.g = 9.8

    self.prev_t = None

    self.task: DroneTask = DroneTask.HOVER
    self.policy_name = "hover_basic"
    self.algo = RLAlgo.PPO
    self.eval_steps = 1000
    self.config_filename = "default_hover.py"
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




  def response(self, t, state, ref ):
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
    self.prev_t = t
    pos = state.pos - ref.pos
    vel = state.vel
    rot = state.rot

    quat = rot.as_quat() 

    # print(pos,vel,rot)
    obs = np.hstack((pos,vel,quat))
    action, _states = self.policy.predict(obs, deterministic=True)

    th_obs,_ = self.policy.policy.obs_to_tensor(obs)
    _action = self.policy.policy._predict(th_obs, deterministic=True)
    # print()
    _action[0].backward()
    # print(th_obs.grad)
    torch.autograd.grad(_action[0],th_obs,retain_graph=True,allow_unused=True)

    action[0]+=self.g
    self.prev_pos = pos.copy()
    # print(type(action))
    return action[0], action[1:]
