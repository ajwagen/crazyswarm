import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env
import time

class MPPIController(ControllerBackbone):
  def __init__(self,isSim, policy_config="trajectory",adaptive=False):
    super().__init__(isSim, policy_config, isPPO=True)

    self.mppi_controller = self.set_MPPI_controller()

  def ref_func_t(self, t):
    # import pdb;pdb.set_trace()
    ref_pos = self.ref_func_obj.pos(t).T
    ref_vel = self.ref_func_obj.vel(t).T
    ref_quat = self.ref_func_obj.quat(t).T
    ref_angvel = self.ref_func_obj.angvel(t).T

    ref = np.hstack((ref_pos, ref_vel, ref_quat, ref_angvel))

    return ref
  
  def response(self, t, state, ref, ref_func, ref_func_obj, fl=1, adaptive=False):
    self.updateDt(t)
    if fl:
      self.prev_t = t
    pos = state.pos - self.offset_pos
    vel = state.vel
    rot = state.rot
    ang = state.ang

    self.ref_func_obj = ref_func_obj

    quat = rot.as_quat()
    # (x,y,z,w) -> (w,x,y,z)
    quat = np.roll(quat, 1)

    obs = np.hstack((pos, vel, quat))
    noise = np.random.normal(scale=self.param_MPPI.noise_measurement_std)
    noisystate = obs + noise
    noisystate[6:10] /= np.linalg.norm(noisystate[6:10])

    state_torch = torch.as_tensor(noisystate, dtype=torch.float32)
    
    # action = self.mppi_controller.policy_cf(state=state_torch, time=t).cpu().numpy()
    action = self.mppi_controller.policy_with_ref_func(state=state_torch, time=t, new_ref_func=self.ref_func_t).cpu().numpy()
    # MPPI controller designed for output in world frame
    # World Frame -> Body Frame
    
    # st = time.time()
    action[1:] = (rot.as_matrix().T).dot(action[1:])
    # print(time.time() - st)
    # print("-------")
    return action[0], action[1:]
