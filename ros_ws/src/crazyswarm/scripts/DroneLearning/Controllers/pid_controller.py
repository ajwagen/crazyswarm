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
    self.count = 0
    self.lamb = 0.2
    self.v_prev = np.zeros(3)
    self.adapt_term = np.zeros(3)

    self.history = np.zeros((1, 14, 50))
    

    # self.mppi_controller = self.set_MPPI_cnotroller()

  def response(self, t, state, ref, ref_func, ref_func_obj, fl=1, adaptation_mean_value=np.zeros(4)):

    self.updateDt(t)
    # if fl:
    if self.prev_t != None:
      dt = t - self.prev_t
    else:
      dt = 0.02
    #   self.prev_t = t

    # PID
    pos = state.pos
    vel = state.vel
    rot = state.rot
    p_err = pos - ref.pos
    v_err = vel - ref.vel
    quat = rot.as_quat() 

    obs = np.hstack((pos, vel, quat))
    # Updating error for integral term.
    self.pos_err_int += p_err * self.dt

    acc_des = (np.array([0, 0, self.g]) 
              - self.kp_pos * (p_err) 
              - self.kd_pos * (v_err) 
              - self.ki_pos * self.pos_err_int 
              + 0.5 * ref.acc
              - 0 * self.adapt_term)

    u_des = rot.as_matrix().T.dot(acc_des)

    acc_des = np.linalg.norm(u_des)

    rot_err = np.cross(u_des / acc_des, np.array([0, 0, 1]))

    eulers = rot.as_euler("ZYX")
    yaw = eulers[0]
    omega_des = - self.kp_rot * rot_err
    omega_des[2] += - self.yaw_gain * (yaw - 0.0)


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
    adaptation_term = np.ones(4)
    adaptation_term[1:] *= 0
    f_t = rot.apply(np.array([0, 0, self.history[0, 10, 0]])) * mass
    z_w = np.array([0, 0, -1])
    adapt_term = mass * a_t - mass * z_w * 9.8 - f_t
    self.adapt_term = (1 - self.lamb) * self.adapt_term + self.lamb * adapt_term
    
    self.adaptation_terms[1:] = self.adapt_term * cf_mass

    adaptation_input = np.r_[obs, acc_des, omega_des]
    if fl!=0.0:
      self.history = np.concatenate((adaptation_input[None, :, None], self.history[:, :, :-1]), axis=2)

    self.count += 1
    self.v_prev = state.vel
    return acc_des, omega_des