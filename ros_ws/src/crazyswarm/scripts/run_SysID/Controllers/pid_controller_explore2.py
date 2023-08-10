import numpy as np
# from Controllers.ctrl_backbone import ControllerBackbone
from Controllers.simple_pid import PIDController
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
                
class PIDController_explore(PIDController):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

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
    self.nominal_mass = 1

    self.rollout_controller = PIDController(**kwargs)
    self.rollout_state = State_struct()

    # self.mppi_controller = self.set_MPPI_cnotroller()

  def _response(self, fl=1, **response_inputs ):
    state = response_inputs.get('state')
    pos = state.pos
    vel = state.vel
    rot = state.rot
    quat = rot.as_quat() 
    
    u = super()._response(fl, **response_inputs) + self.exploration_input()

    if fl!=0.0:
      self.sysid_state_history[self.count % self.sysid_H] = np.r_[pos, vel, quat, state.ang]
      self.sysid_action_history[self.count % self.sysid_H] = u
      self.count += 1

    # if (self.count + 1) % self.sysid_H == 0:
    #   for h in range(self.sysid_H - 1):
    #     sysid_z = self.kernel.phi(self.sysid_state_history[h], None)
    #     self.sysid_data_cov += np.outer(sysid_z,sysid_z)
    #     self.sysid_data_process += np.outer(sysid_z, self.sysid_state_history[h+1][3:6] - self.next_state(self.sysid_state_history[h], self.sysid_action_history[h], self.dt)[3:6])

    #     self.mismatch = self.sysid_state_history[h+1][:3] - self.next_state(self.sysid_state_history[h], self.sysid_action_history[h], self.dt)[:3] - sysid_z @ self.Aker
    #     self.mismatches[h] = np.linalg.norm(self.mismatch)

    #   thetahat = np.linalg.inv(self.sysid_data_cov) @ self.sysid_data_process
    #   self.Aker = thetahat.T
    

    return u[0], u[1:]
  
  # def PID_block(self, pos, vel, rot, p_err, v_err, ref):
  #   acc_des = (np.array([0, 0, self.g]) 
  #           - self.kp_pos * (p_err) 
  #           - self.kd_pos * (v_err) 
  #           - self.ki_pos * self.pos_err_int 
  #           + 0.5 * ref.acc
  #           - 0 * self.wind_adapt_term)

  #   u_des = rot.as_matrix().T.dot(acc_des)

  #   acc_des = np.linalg.norm(u_des)

  #   rot_err = np.cross(u_des / acc_des, np.array([0, 0, 1]))

  #   eulers = rot.as_euler("ZYX")
  #   yaw = eulers[0]
  #   omega_des = - self.kp_rot * rot_err
  #   omega_des[2] += - self.yaw_gain * (yaw - 0.0)

  #   return np.r_[acc_des, omega_des]
  
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
    dsdt[3:6] = np.array([0., 0., -self.g]) + qrotate(q,f_u) / self.nominal_mass
    
    qnew = qintegrate(q, omega, dt, frame='body')
    qnew = qstandardize(qnew)
    # transform qnew to a "delta q" that works with the usual euler integration
    dsdt[6:10] = (qnew - q) / dt

    return dsdt

  def exploration_input(self, explore_type='random'):
    if explore_type=='CEED':
      u = np.random.randn(4)
    elif explore_type=='random':
      u = np.random.randn(4)
    
    return u
  
  def compute_next_input(self, est_instance, hessian, cov, x, h0, U_init, epoch_power, ref):
    # dimx, dimu, dimz, H = est_instance.get_dim()
    dimx = 13
    dimu = 4
    self.power = 0.0001
    self.planning_horizon = 10
    power_to_go = self.sysid_H * self.power - epoch_power
    if self.planning_horizon is None:
      unroll_steps = self.sysid_H - h0
    else:
      unroll_steps = self.planning_horizon
      if unroll_steps > self.sysid_H - h0:
        unroll_steps = self.sysid_H - h0
      else:
        power_to_go = power_to_go * unroll_steps / (self.sysid_H - h0)
    if power_to_go < 0:
      return torch.zeros(dimu), torch.zeros(dimu,unroll_steps)

    U_init2 = torch.zeros(dimu,unroll_steps)
    U_init2[:,0:unroll_steps - 1] = U_init[:, 0:unroll_steps - 1]
    U_init = U_init2[:, :, None]
    U = U_init.repeat(1, 1, self.N) 
    U = U + torch.randn(dimu, unroll_steps, self.N)
    for i in range(self.N):
      U[:,:,i] = torch.sqrt(power_to_go) * U[:, :, i] / torch.norm(U[:, :, i])
    x = x[:, None]
    x = x.repeat(1, self.N)
    cov = cov[:, :, None]
    cov = cov.repeat(1, 1, self.N)
    
    # if self.stab_controller is not None:
    #   plan_stab_con = copy.deepcopy(self.stab_controller)
    #   plan_stab_con.expand_int_error(self.N)

    for h in range(unroll_steps):
      u = U[:,h,:]
      # if self.scaling_mat is not None:
      #   u = self.scaling_mat @ u
      # if self.stab_controller is not None:
      self.rollout_state.vec2struct(x)

      u = u + self.rollout_controller._response(fl=0, 
                                                t = 0.02 * h,
                                                state = self.rollout_state,
                                                ref = ref)
      
      cov = cov + self.get_cov(x, u, parallel=True)
      # x = est_instance.dynamics(x, u, noiseless=True)
      x = self.next_state(x, u, dt = 0.02)

    if self.objective == 'lammin':
      e,_ = torch.linalg.eig(cov[:,:,0])
      min_loss = torch.min(torch.real(e))
    else:
      cov_inv = torch.linalg.inv(cov[:,:,0] + 0.001 * torch.eye(self.dimphi)).contiguous()
      min_loss = torch.trace(hessian @ torch.kron(torch.eye(3),cov_inv))
    min_idx = 0
    for i in range(self.N):
      if self.objective == 'lammin':
        e,_ = torch.linalg.eig(cov[:,:,i])
        loss = torch.min(torch.real(e))
        if loss.detach().numpy() > min_loss.detach().numpy():
          min_loss = loss
          min_idx = i
      else:
        cov_inv = torch.linalg.inv(cov[:,:,i] + 0.001 * torch.eye(self.dimphi)).contiguous()
        loss = torch.trace(hessian @ torch.kron(torch.eye(3),cov_inv))
        if loss.detach().numpy() < min_loss.detach().numpy():
          min_loss = loss
          min_idx = i
    if self.scaling_mat is None:
      return U[:,0,min_idx], U[:,1:,min_idx]
    else:
      return self.scaling_mat @ U[:,0,min_idx], U[:,1:,min_idx]
  
  def get_cov(self,x, u, parallel=False):
      if parallel:
          z = self.kernel.phi(x, None)
          cov = torch.einsum('ik, jk -> ijk',z,z)
          return cov
      else:
          z = self.kernel.phi(x, None)
          return torch.outer(z, z)