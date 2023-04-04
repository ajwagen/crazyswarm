import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env

class MPPIController(ControllerBackbone):
  def __init__(self,isSim, policy_config="trajectory",adaptive=False):
    super().__init__(isSim, policy_config, isPPO=True)

    self.mppi_controller = self.set_MPPI_cnotroller()

  def response(self, t, state, ref, ref_func, fl=1, adaptive=False):
    self.updateDt(t)
    if fl:
      self.prev_t = t
    pos = state.pos - self.offset_pos
    vel = state.vel
    rot = state.rot
    ang = state.ang

    quat = rot.as_quat()
    # (x,y,z,w) -> (w,x,y,z)
    quat = np.roll(quat, 1)

    obs = np.hstack((pos, vel, quat, ang))
    noise = np.random.normal(scale=self.param_MPPI.noise_measurement_std)
    noisystate = obs + noise
    noisystate[6:10] /= np.linalg.norm(noisystate[6:10])

    state_torch = torch.as_tensor(noisystate, dtype=torch.float32)
    
    action = self.mppi_controller.policy_cf(state=state_torch, time=t).cpu().numpy()
    
    # MPPI controller designed for output in world frame
    # World Frame -> Body Frame
    
    action[1:] = (rot.as_matrix().T).dot(action[1:])
    return action[0], action[1:]
