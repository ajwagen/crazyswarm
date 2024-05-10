import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone
from scipy.spatial.transform import Rotation as R
from cf_utils.rigid_body import State_struct
import torch
import time

class PIDController(ControllerBackbone):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

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

    self.history = np.zeros((1, 14, 50))
    

    # self.mppi_controller = self.set_MPPI_cnotroller()
    self.runL1 = True

  def _response(self, fl=1, **response_inputs ):
    
    t = response_inputs.get('t')
    state = response_inputs.get('state')
    ref = response_inputs.get('ref')


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


    v_t = state.vel
    if self.count > 2:
      v_t = state.vel
      a_t = (v_t - self.v_prev) / dt
    else:
      a_t = np.array([0, 0, 0])

    unity_mass = 1
    f_t = rot.apply(np.array([0, 0, self.history[0, 10, 0]])) * unity_mass

    obs = np.hstack((pos, vel, quat))
    state_torch = torch.as_tensor(obs, dtype=torch.float32)
    adaptation_type = 'L1_learned'
    # Make sure to remember that on the May 8th runs we had to 0 out the
    # z terms of the A_hat matrix used in L1_learned after we had already
    # ran trials with L1 Basic
    self.L1_adaptation(self.dt, v_t, f_t)
    self.L1_learned(state_torch[3:9])
    if not self.pseudo_adapt:
      if adaptation_type == 'L1_basic':
          # L1 adaptation update
        self.final_adapt_term = self.wind_adapt_term
        print('L_1 here')
      elif adaptation_type == 'L1_learned':
          self.final_adapt_term = self.L1_adapt_term.cpu().detach().numpy()
          print('L1_learned here')
      else:
        self.naive_adaptation(a_t, f_t)
        self.final_adapt_term = np.zeros_like(self.wind_adapt_term)
        print('naive here')
    print(self.final_adapt_term)
    #obs = np.hstack((pos, vel, quat))
    # Updating error for integral term.
    self.pos_err_int += p_err * self.dt
    acc_des = (np.array([0, 0, self.g]) 
              - self.kp_pos * (p_err) 
              - self.kd_pos * (v_err) 
              - self.ki_pos * self.pos_err_int 
              + 0.5 * ref.acc
              + 1 * self.final_adapt_term)
    #          + 1 * self.wind_adapt_term)
    

    u_des = rot.as_matrix().T.dot(acc_des)

    acc_des = np.linalg.norm(u_des)

    rot_err = np.cross(u_des / acc_des, np.array([0, 0, 1]))

    eulers = rot.as_euler("ZYX")
    yaw = eulers[0]
    omega_des = - self.kp_rot * rot_err
    omega_des[2] += - self.yaw_gain * (yaw - 0.0)
      

    self.adaptation_terms[1:] = self.wind_adapt_term

    adaptation_input = np.r_[obs, acc_des, omega_des]
    if fl!=0.0:
      self.history = np.concatenate((adaptation_input[None, :, None], self.history[:, :, :-1]), axis=2)

    self.count += 1
    self.v_prev = state.vel
    return acc_des, omega_des
  

if __name__=='__main__':
  cntrl = PIDController()
