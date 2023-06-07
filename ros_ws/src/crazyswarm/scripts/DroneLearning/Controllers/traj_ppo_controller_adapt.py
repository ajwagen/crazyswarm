import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env

# def smoothing(arr, scale=1.0):
#     smooth_horizon = 5
#     smooth_arr = []
#     horizon = np.zeros(smooth_horizon)
#     for i in range(len(arr)):
        
#         horizon[i % smooth_horizon] = arr[i]
#         if i < smooth_horizon:
#             smooth_arr.append(arr[i])
#         else:
#             smooth_arr.append(np.mean(horizon))
    
#     return np.array(smooth_arr) / scale

class PPOController_trajectory_adaptive(ControllerBackbone):
  def __init__(self,isSim, policy_config="trajectory", adaptive=True, pseudo_adapt=False, adapt_smooth=False):
    super().__init__(isSim, policy_config, isPPO=True, adaptive=adaptive, pseudo_adapt = pseudo_adapt)
    # self.e_dims = e_dims
    self.set_policy()
    # self.obs_history = np.zeros((50, 14))
    # self.history = torch.zeros((14, 50))./to("cuda:0")
    self.history = torch.zeros((1, 14, 50)).to("cuda:0")

    self.adapt_smooth = adapt_smooth
    self.adaptation_history = np.zeros((1, 4))
    self.adaptation_history_len = 100

  def response(self, t, state, ref , ref_func, ref_func_obj, fl=1, adaptation_mean_value=np.zeros(4)):

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
    
    if self.pseudo_adapt==False and fl!=0.0:

      # if self.adaptation_warmup:
      adaptation_term = self.adaptive_policy(self.history).detach().cpu().numpy()[0]
      self.adaptation_mean = np.vstack((self.adaptation_mean, self.adaptation_terms[None, :]))
      
      if self.adapt_smooth:
        if len(self.adaptation_history) >= self.adaptation_history_len: 
          self.adaptation_history = np.vstack((adaptation_term[None, :], self.adaptation_history[:-1]))
        else:
          self.adaptation_history = np.vstack((adaptation_term[None, :], self.adaptation_history))

        adaptation_term[0] = np.clip(np.mean(self.adaptation_history[:, 0]), 0.0, 10.0)
        adaptation_term[1] = np.mean(self.adaptation_history[:, 1]) / 3
        adaptation_term[2] = np.mean(self.adaptation_history[:, 2]) / 3
        adaptation_term[3] = np.mean(self.adaptation_history[:, 3]) / 3

      self.adaptation_terms = adaptation_term

      obs_ = np.hstack((obs, adaptation_term))

    else:
      pseudo_adapt_term =  np.ones(self.e_dims) * 1.0
      pseudo_adapt_term[1:] *= 0 # mass -> 1, wind-> 0
      obs_ = np.hstack((obs, pseudo_adapt_term))
    # obs_ = np.hstack((obs, 1.0))

    if fl==0:
        obs_ = np.zeros((self.time_horizon+1) * 3 + 10 + self.e_dims)
    else:

        if self.relative:
          obs_ = np.hstack([obs_, obs_[0:3] - rot.inv().apply(ref_func(t)[0].pos)] + [obs_[0:3] - rot.inv().apply(ref_func(t + 3 * i * dt)[0].pos) for i in range(self.time_horizon)])

        else:
          ff_terms = [ref_func(t + 3 * i * dt)[0].pos for i in range(self.time_horizon)]
          obs_ = np.hstack([obs_, obs_[0:3] - ref_func(t)[0].pos] + ff_terms)

    # import pdb;pdb.set_trace()

    action, _states = self.policy.predict(obs_, deterministic=True)

    adaptation_input = np.concatenate((obs, action), axis=0)
    adaptation_input = torch.from_numpy(adaptation_input).to("cuda:0").float()

    if fl!=0.0:
      self.history = torch.cat((adaptation_input[None, :, None], self.history[:, :, :-1].clone()), dim=2)
      # import pdb;pdb.set_trace()
    if self.log_scale:
      action[0] = np.sinh(action[0])
    else:
      action[0] += self.g

    # new_obs = np.hstack((obs, action))
    # self.obs_history[1:] = self.obs_history[0:-1]
    # self.obs_history[0] = new_obs
    # import pdb;pdb.set_trace()
    return action[0], action[1:]
