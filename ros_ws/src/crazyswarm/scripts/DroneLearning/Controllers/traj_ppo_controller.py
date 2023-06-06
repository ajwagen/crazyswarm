import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env
import time

class PPOController_trajectory(ControllerBackbone):
  def __init__(self,isSim, policy_config="trajectory",adaptive=False):
    super().__init__(isSim, policy_config, isPPO=True)

    self.set_policy()

  def response(self, t, state, ref , ref_func, ref_func_obj, fl=1, adaptive=False, adaptation_mean_value=np.zeros(4)):

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


    if self.body_frame:
      pos = rot.inv().apply(pos)
      vel = rot.inv().apply(vel)

    # obs = np.hstack((pos, vel, quat))


    if fl==0:
        obs_ = np.zeros((self.time_horizon+1) * 3 + 10)
    else:
        obs_ = np.hstack((pos, vel, quat))
        if self.relative:
          obs_ = np.hstack([obs_, obs_[0:3] - rot.inv().apply(ref_func(t)[0].pos)] + [obs_[0:3] - rot.inv().apply(ref_func(t + 3 * i * dt)[0].pos) for i in range(self.time_horizon)])

        else:
          ff_terms = [ref_func(t + 3 * i * dt)[0].pos for i in range(self.time_horizon)]
          obs_ = np.hstack([obs_, obs_[0:3] - ref_func(t)[0].pos] + ff_terms)


    action, _states = self.policy.predict(obs_, deterministic=True)

    if self.log_scale:
      action[0] = np.sinh(action[0])
    else:
      action[0] += self.g

    self.prev_pos = pos.copy()
    return action[0], action[1:]
