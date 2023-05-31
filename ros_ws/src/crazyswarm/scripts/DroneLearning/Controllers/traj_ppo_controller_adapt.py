import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env

class PPOController_trajectory_adaptive(ControllerBackbone):
  def __init__(self,isSim, policy_config="trajectory", adaptive=True, e_dims = 1):
    super().__init__(isSim, policy_config, isPPO=True, adaptive=adaptive)
    self.e_dims = e_dims
    self.set_policy()
    self.obs_history = np.zeros((100, 14))

  def response(self, t, state, ref , ref_func, ref_func_obj, fl=1, adaptive=1):

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

    obs = np.hstack((pos, vel, quat))
    
    adaptation_term = self.adaptive_policy(torch.tensor(self.obs_history.transpose(1, 0)[None, : ]).float()).flatten()
    adaptation_term = (adaptation_term * 0.8 + 0.6)
    obs_ = np.hstack((obs, adaptation_term.detach().cpu().numpy()))

    if fl==0:
        obs_ = np.zeros((self.time_horizon+1) * 3 + 10 + self.e_dims)
    else:

        if self.relative:
          obs_ = np.hstack([obs_, obs_[0:3] - rot.inv().apply(ref_func(t)[0].pos)] + [obs_[0:3] - rot.inv().apply(ref_func(t + 3 * i * dt)[0].pos) for i in range(self.time_horizon)])

        else:
          ff_terms = [ref_func(t + 3 * i * dt)[0].pos for i in range(self.time_horizon)]
          obs_ = np.hstack([obs_, obs_[0:3] - ref_func(t)[0].pos] + ff_terms)


    action, _states = self.policy.predict(obs_, deterministic=True)
    action[0] = np.sinh(action[0])

    new_obs = np.hstack((obs, action))
    self.obs_history[0:-1] = self.obs_history[1:]
    self.obs_history[-1] = new_obs

    return action[0], action[1:]
