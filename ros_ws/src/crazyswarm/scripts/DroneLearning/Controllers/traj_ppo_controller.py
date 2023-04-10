import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env

class PPOController_trajectory(ControllerBackbone):
  def __init__(self,isSim, policy_config="trajectory",adaptive=False):
    super().__init__(isSim, policy_config, isPPO=True)

    self.set_policy()

  def response(self, t, state, ref , ref_func, fl=1, adaptive=False):

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

    obs = np.hstack((pos, vel, quat))

    if fl==0:
       obs_ = np.zeros((self.time_horizon+1) * 3 + 10)
    else:
        ff_terms = [ref_func(t + 3 * i * dt)[0].pos for i in range(self.time_horizon)]
        obs_ = np.hstack([obs, obs[0:3] - ref_func(t)[0].pos] + ff_terms)


    action, _states = self.policy.predict(obs_, deterministic=True)

    action[0] += self.g
    self.prev_pos = pos.copy()
    return action[0], action[1:]
