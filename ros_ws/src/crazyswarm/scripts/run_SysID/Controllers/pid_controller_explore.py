import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone
from scipy.spatial.transform import Rotation as R
from cf_utils.rigid_body import State_struct
import torch
import time


class AirDragKernel:
	def __init__(self):
		return

	def get_dim(self):
		return 3

	def phi(self,x,u):
		if len(x.shape) > 1:
			return x[3:6,:]
		else:
			return x[3:6]
                
# Quaternion routines adapted from rowan to use autograd
def qmultiply(q1, q2):
	return np.concatenate((
		np.array([q1[0] * q2[0] - np.sum(q1[1:4] * q2[1:4])]), # w1w2
		q1[0] * q2[1:4] + q2[0] * q1[1:4] + np.cross(q1[1:4], q2[1:4])))

def qconjugate(q):
	return np.concatenate((q[0:1],-q[1:4]))

def qrotate(q, v):
	quat_v = np.concatenate((np.array([0]), v))
	# quat_v = np.r_[0, v]
	return qmultiply(q, qmultiply(quat_v, qconjugate(q)))[1:]

def qexp(q):
	norm = np.linalg.norm(q[1:4])
	e = np.exp(q[0])
	result_w = e * np.cos(norm)
	if np.isclose(norm, 0):
		result_v = np.zeros(3)
	else:
		result_v = e * q[1:4] / norm * np.sin(norm)
	return np.concatenate((np.array([result_w]), result_v))

def qintegrate(q, v, dt, frame='body'):
	quat_v = np.concatenate((np.array([0]), v*dt/2))
	if frame == 'body':
		return qmultiply(q, qexp(quat_v))		
	if frame == 'world':
		return qmultiply(qexp(quat_v), q)

def qstandardize(q):
  if q[0] < 0:
    q *= -1

  return q / np.linalg.norm(q)
                
class PIDController_explore(ControllerBackbone):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.kp_pos = 6.0
    self.kd_pos = 4.0
    self.ki_pos = 1.5 # 0 for sim
    self.kp_rot =   150.0/16
    self.yaw_gain = 220.0/16
    self.kp_ang =   16

    self.pos_err_int = np.zeros(3)
    self.count = 0
    self.lamb = 0.2
    self.v_prev = np.zeros(3)
    self.adapt_term = np.zeros(3)

    self.history = np.zeros((1, 14, 50))
    

    np.random.seed(1)
    # Adrag = torch.tensor(0.01 * np.random.randn(3,3), dtype=torch.float32)
    # Adrag_norm = torch.linalg.norm(Adrag)
    # Adrag = Adrag @ Adrag.T
    # self.Adrag = Adrag / torch.linalg.norm(Adrag) * Adrag_norm
    self.Aker = np.zeros((3, 3))
    self.kernel = AirDragKernel() 

    self.sysid_H = 50
    self.sysid_state_history = np.zeros((self.sysid_H, 13))
    self.sysid_state_history[:, 9] = 1.
    self.sysid_action_history = np.zeros((self.sysid_H, 4))

    self.dimphi = self.kernel.get_dim()
    self.sysid_data_cov = np.zeros((self.dimphi,self.dimphi))
    self.sysid_data_process = np.zeros((self.dimphi, 3))

    self.mismatches = np.zeros((self.sysid_H - 1, 3))

    # self.mppi_controller = self.set_MPPI_cnotroller()

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

    if self.pseudo_adapt == False:
      if self.runL1:
          # L1 adaptation update
        self.L1_adaptation(dt, v_t, f_t)
      else:
          self.naive_adaptation(a_t, f_t)

    obs = np.hstack((pos, vel, quat))
    # Updating error for integral term.
    self.pos_err_int += p_err * self.dt

    acc_des = (np.array([0, 0, self.g]) 
              - self.kp_pos * (p_err) 
              - self.kd_pos * (v_err) 
              - self.ki_pos * self.pos_err_int 
              + 0.5 * ref.acc
              - 0 * self.wind_adapt_term)

    u_des = rot.as_matrix().T.dot(acc_des)

    acc_des = np.linalg.norm(u_des)

    rot_err = np.cross(u_des / acc_des, np.array([0, 0, 1]))

    eulers = rot.as_euler("ZYX")
    yaw = eulers[0]
    omega_des = - self.kp_rot * rot_err
    omega_des[2] += - self.yaw_gain * (yaw - 0.0)
      

    self.adaptation_terms[1:] = self.wind_adapt_term

    adaptation_input = np.r_[obs, acc_des, omega_des]
    
    u = np.random.randn(4)
    u = np.r_[acc_des, omega_des] + u

    if fl!=0.0:
      self.history = np.concatenate((adaptation_input[None, :, None], self.history[:, :, :-1]), axis=2)
      self.sysid_state_history[self.count % self.sysid_H] = np.r_[pos, vel, quat, state.ang]
      self.sysid_action_history[self.count % self.sysid_H] = u

      self.count += 1
      self.v_prev = state.vel

    if (self.count + 1) % self.sysid_H == 0:
      for h in range(self.sysid_H - 1):
        sysid_z = self.kernel.phi(self.sysid_state_history[h], None)
        self.sysid_data_cov += np.outer(sysid_z,sysid_z)
        self.sysid_data_process += np.outer(sysid_z, self.sysid_state_history[h+1][3:6] - self.next_state(self.sysid_state_history[h], self.sysid_action_history[h], dt)[3:6])

        self.mismatch = self.sysid_state_history[h+1][:3] - self.next_state(self.sysid_state_history[h], self.sysid_action_history[h], dt)[:3] - sysid_z @ self.Aker
        self.mismatches[h] = np.linalg.norm(self.mismatch)

      # print(self.Aker)
      thetahat = np.linalg.inv(self.sysid_data_cov) @ self.sysid_data_process
      self.Aker = thetahat.T

      print(np.mean(self.mismatches))
      print(self.Aker)

    return u[0], u[1:]
  
  def next_state(self, s, a, dt):
      omega_des = a[1:]
      s = s + self.dsdt_dynamics(s, a, dt) * dt
      s[10:] = omega_des
      return s
	# dsdt = f(s,a)
  def dsdt_dynamics(self,s,a, dt):

    dsdt = np.zeros(13)
    q = s[6:10]
    omega = s[10:]

    eta = a
    f_u = np.array([0,0,eta[0]])
    # tau_u = np.array([eta[1],eta[2],eta[3]])

    # dynamics 
    # dot{p} = v 
    dsdt[0:3] = s[3:6] 	# <- implies velocity and position in same frame
    # mv = mg + R f_u  	# <- implies f_u in body frame, p, v in world frame
    dsdt[3:6] = np.array([0., 0., -self.g]) + qrotate(q,f_u) / self.mass
    
    qnew = qintegrate(q, omega, dt, frame='body')
    qnew = qstandardize(qnew)
    # transform qnew to a "delta q" that works with the usual euler integration
    dsdt[6:10] = (qnew - q) / dt

    return dsdt
