import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone
from scipy.spatial.transform import Rotation as R
from cf_utils.rigid_body import State_struct
import torch
import time

class PIDController(ControllerBackbone):
  def __init__(self,isSim, policy_config=None, adaptive = False):
    super().__init__(isSim, policy_config)

    self.kp_pos = 6.0
    self.kd_pos = 4.0
    self.ki_pos = 1.2 # 0 for sim
    self.kp_rot =   150.0/16
    self.yaw_gain = 220.0/16
    self.kp_ang =   16

    self.pos_err_int = np.zeros(3)

    # self.mppi_controller = self.set_MPPI_cnotroller()

  def response(self, t, state, ref, ref_func, ref_func_obj, fl=1, adaptation_mean_value=np.zeros(4)):

    self.updateDt(t)
    if fl:
      self.prev_t = t

    # PID
    pos = state.pos
    vel = state.vel
    rot = state.rot
    p_err = pos - ref.pos
    v_err = vel - ref.vel
    # Updating error for integral term.
    self.pos_err_int += p_err * self.dt

    acc_des = (np.array([0, 0, self.g]) 
              - self.kp_pos * (p_err) 
              - self.kd_pos * (v_err) 
              - self.ki_pos * self.pos_err_int 
              + 0.5 * ref.acc)

    u_des = rot.as_matrix().T.dot(acc_des)

    acc_des = np.linalg.norm(u_des)

    rot_err = np.cross(u_des / acc_des, np.array([0, 0, 1]))

    eulers = rot.as_euler("ZYX")
    yaw = eulers[0]
    omega_des = - self.kp_rot * rot_err
    omega_des[2] += - self.yaw_gain * (yaw - 0.0)

    # ref_orient = ref.rot.as_euler("ZYX")
    # yaw, _, _ = rot.as_euler('ZYX')
    # yaw_des = ref_orient[0]  # self.ref.yaw(t)

    # omega_des = - self.kp_rot * rot_err
    # omega_des[2] = - self.yaw_gain * (yaw - yaw_des)
    # print("PID a : ", acc_des, " w : ", omega_des, "MPPI a: ", action[0], " w : ", omega_d)
    return acc_des, omega_des