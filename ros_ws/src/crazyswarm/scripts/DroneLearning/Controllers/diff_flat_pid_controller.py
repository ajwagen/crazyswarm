import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone
from scipy.spatial.transform import Rotation as R
from cf_utils.rigid_body import State_struct
import torch
# https://kilthub.cmu.edu/articles/thesis/Dynamical_Model_Learning_and_Inversion_for_Aggressive_Quadrotor_Flight/19700077

class diffFlat_PIDController(ControllerBackbone):
  def __init__(self,isSim, policy_config=None, adaptive = False):
    super().__init__(isSim, policy_config)

    self.kp_pos = 6.0
    self.kd_pos = 4.0
    self.ki_pos = 0 # 0 for sim
    self.kp_rot =   90.0/16
    self.kd_rot = 1
    self.yaw_gain = 220.0/16

    self.pos_err_int = np.zeros(3)

  # def response_t(self, t, state, ref, ref_func, fl=1):

  #   self.updateDt(t)
  #   if fl:
  #     self.prev_t = t

  #   pos = state.pos
  #   vel = state.vel
  #   rot = state.rot
  #   p_err = pos - ref.pos

  #   p_err = pos - ref.pos
  #   v_err = vel - ref.vel
    
  #   self.pos_err_int += p_err * self.dt 
  #   acc_des = np.array([0, 0, self.g]) \
  #             - self.kp_pos * (p_err) \
  #             - self.kd_pos * (v_err) \
  #             - self.ki_pos * self.pos_err_int \
  #             + ref.acc

  #   bodyz_acc = np.linalg.norm(acc_des)

  #   # Compute omega_des
  #   z_des = acc_des / np.linalg.norm(acc_des)
  #   bodyz_acc_dot = ref.jerk.dot(z_des)
  #   z_des_dot = 1 / bodyz_acc * (ref.jerk - bodyz_acc_dot * z_des)
  #   # omega_des in world frame, without yaw
  #   omega_des = np.cross(z_des, z_des_dot)
  #   # Convert to body frame
  #   omega_des += rot.as_matrix().T.dot(omega_des)

  #   # Compute alpha_des
  #   bodyz_acc_2dot = ref.snap.dot(z_des) + bodyz_acc * (z_des_dot.dot(z_des_dot))
    
  #   z_des_2dot = 1 / bodyz_acc * (ref.snap - bodyz_acc_2dot * z_des - 2 * bodyz_acc_dot * z_des_dot)
    
  #   alpha_des = np.cross(z_des, z_des_2dot)
  #   # convert to body frame
  #   alpha_des = rot.as_matrix().T.dot(alpha_des)

  #   # Convert everything to body frame
  #   acc_des = rot.as_matrix().T.dot(acc_des)
  #   bodyz_acc = np.linalg.norm(acc_des)
  #   # print('pos', state.pos, 'pos_err', p_err, 'int', self.pos_err_int)

  #   rot_err = np.cross(acc_des / bodyz_acc, np.array([0, 0, 1]))
  #   ang_err = state.ang - omega_des
  #   alpha_fb = - self.kp_rot * rot_err - self.kd_rot * ang_err # + alpha_des
  #   # torque = self.model.I.dot(alpha_fb)

  #   yaw, _, _ = rot.as_euler('zyx')
  #   yaw_ref, _, _ = ref.rot.as_euler('zyx')

  #   yaw_des = yaw_ref
  #   yaw_err = yaw - yaw_des
  #   u_yaw = - self.kp_rot * yaw_err - self.kd_rot * state.ang[2]
  #   # torque[2] = u_yaw

  #   omega_output = alpha_fb
  #   omega_output[2] = u_yaw
  #   # print(acc_des, ref.pos, state.pos)
  #   # if z_des[-1] < 0:
  #   #   exit()
  #   return bodyz_acc, omega_output 

  def response(self, t, state, ref, ref_func, fl=1):

    self.updateDt(t)
    if fl:
      self.prev_t = t

    pos = state.pos
    vel = state.vel
    rot = state.rot
    p_err = pos - ref.pos

    p_err = pos - ref.pos
    v_err = vel - ref.vel
    
    self.pos_err_int += p_err * self.dt 
    acc_des = np.array([0, 0, self.g]) \
              - self.kp_pos * (p_err) \
              - self.kd_pos * (v_err) \
              - self.ki_pos * self.pos_err_int \
              + ref.acc

    bodyz_acc = np.linalg.norm(acc_des)

    # Compute omega_des
    z_des = acc_des / np.linalg.norm(acc_des)
    bodyz_acc_dot = ref.jerk.dot(z_des)
    z_des_dot = 1 / bodyz_acc * (ref.jerk - bodyz_acc_dot * z_des)
    # omega_des in world frame, without yaw
    omega_des = np.cross(z_des, z_des_dot)
    # Convert to body frame
    omega_des += rot.as_matrix().T.dot(omega_des)
    rot_err = np.cross(acc_des / bodyz_acc, np.array([0, 0, 1]))

    omega_op = - self.kp_rot * rot_err + rot_err
    
    yaw, _, _ = rot.as_euler('zyx')
    yaw_ref, _, _ = ref.rot.as_euler('zyx')
    yaw_des = yaw_ref
    yaw_err = yaw - yaw_des
    omega_op[2] = - self.kp_rot * yaw_err

    return bodyz_acc, omega_op
