import numpy as np

from gym import spaces

from quadsim.learning.base_env import BaseQuadsimEnv
from quadsim.learning.configuration.configuration import AllConfig

class YawflipEnv(BaseQuadsimEnv):
  """
     state is pos, vel, quat
     action is u, angvel
  """
  def __init__(self, config: AllConfig, save_data: bool=False, data_file=None):
    super().__init__(config, save_data, data_file)

    self.action_space = spaces.Box(low=np.array([-9.8, -20, -20, -1.5]), high=np.array([30, 20, 20, 1.5]))
    self.t_end = 5.0
    self.pos_weight = 1.0

  def reward(self, state, action):
    yaw = state.rot.as_euler('ZYX')[0]

    yawcost = 0.5 * min(abs(np.pi/2 - yaw), abs(-np.pi/2 - yaw))
    poscost = self.pos_weight * min(np.linalg.norm(state.pos), 1.0)

    cost = yawcost + poscost

    return -cost
