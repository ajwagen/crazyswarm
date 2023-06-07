import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env

class PPOController_trajectory_L1_adaptive(ControllerBackbone):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.set_policy()

    self.history = np.zeros((1, 14, 50))

    self.adaptation_history = np.zeros((1, 4))
    self.adaptation_history_len = 100
    self.lamb = 0.2
    self.v_prev = 0
    self.adapt_term = np.zeros(3)
    self.count = 0

  def _response(self, fl = 1, **response_inputs):

    t = response_inputs.get('t')
    state = response_inputs.get('state')
    ref = response_inputs.get('ref')
    ref_func = response_inputs.get('ref_func')
    ref_func_obj = response_inputs.get('ref_func_obj')

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

    obs_bf = np.hstack((pos, vel, quat))

    if self.body_frame:
      pos = rot.inv().apply(pos)
      vel = rot.inv().apply(vel)

    obs = np.hstack((pos, vel, quat))
    
    if fl!=0.0:
    # if self.pseudo_adapt==False and fl!=0.0:

      # if self.adaptation_warmup:
      # adaptation_term = self.adaptive_policy(self.history).detach().cpu().numpy()[0]
      # self.adaptation_mean = np.vstack((self.adaptation_mean, self.adaptation_terms[None, :]))
      adaptation_term = np.ones(4)
      adaptation_term[1:] *= 0

      v_t = 0
      v_t_prev = 0
      if self.count > 2:
        v_t = state.vel
        v_t_prev = self.v_prev
        a_t = (v_t - v_t_prev) / dt
      else:
        a_t = np.array([0, 0, 0])
        

      mass = 1
      cf_mass = 0.04

      f_t = rot.apply(np.array([0, 0, self.history[0, 10, 0]])) * mass
      z_w = np.array([0, 0, -1])
      adapt_term = mass * a_t - mass * z_w * 9.8 - f_t
      self.adapt_term = (1 - self.lamb) * self.adapt_term + self.lamb * adapt_term
            
      self.adaptation_terms = self.adapt_term
      obs_ = np.hstack((obs, self.adapt_term))

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

    # adaptation_input = torch.from_numpy(adaptation_input).to("cuda:0").float()

      # import pdb;pdb.set_trace()
    if self.log_scale:
      action[0] = np.sinh(action[0])
    else:
      action[0] += self.g
    
    adaptation_input = np.concatenate((obs_bf, action), axis=0)
    if fl!=0.0:
      self.history = np.concatenate((adaptation_input[None, :, None], self.history[:, :, :-1]), axis=2)

    self.dt_prev = dt
    self.count += 1
    self.v_prev = state.vel
    # new_obs = np.hstack((obs, action))
    # self.obs_history[1:] = self.obs_history[0:-1]
    # self.obs_history[0] = new_obs
    # import pdb;pdb.set_trace()
    return action[0], action[1:]
