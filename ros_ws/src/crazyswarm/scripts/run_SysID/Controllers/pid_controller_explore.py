import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone
from Controllers.math_utils import *
from scipy.spatial.transform import Rotation as R
from cf_utils.rigid_body import State_struct
import torch
import time
from cf_utils.rigid_body import State_struct
from pytorch3d import transforms

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def xyzw(quat):
	w = quat[0]
	quat[0:3] = quat[1:]
	quat[-1] = w
	return quat

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
    self.int_p_e = torch.zeros((500, 3), dtype=torch.float32)
    

    # self.mppi_controller = self.set_MPPI_cnotroller()
    # self.runL1 = False

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
    self.pos_err_int += p_err * self.dt
    obs = np.hstack((pos, vel, quat))

    if self.count > 2:
      vel = state.vel
      a_t = (vel - self.v_prev) / dt
    else:
      a_t = np.array([0, 0, 0])

    unity_mass = 1
    f_t = rot.apply(np.array([0, 0, self.history[0, 10, 0]])) * unity_mass

    if not self.pseudo_adapt:
      if self.runL1:
          # L1 adaptation update
        self.L1_adaptation(self.dt, vel, f_t)
      else:
        self.naive_adaptation(a_t, f_t)
    self.adaptation_terms[1:] = self.wind_adapt_term

    acc_des = (np.array([0, 0, self.g]) 
              - self.kp_pos * (p_err) 
              - self.kd_pos * (v_err) 
              - self.ki_pos * self.pos_err_int 
              + 0.5 * ref.acc
              + 1 * self.wind_adapt_term)
    u_des = rot.as_matrix().T.dot(acc_des)
    acc_des = np.linalg.norm(u_des)

    rot_err = np.cross(u_des / acc_des, np.array([0, 0, 1]))
    eulers = rot.as_euler("ZYX")
    yaw = eulers[0]

    omega_des = - self.kp_rot * rot_err
    omega_des[2] += - self.yaw_gain * (yaw - 0.0)
      
    adaptation_input = np.r_[obs, acc_des, omega_des]
    if fl!=0.0:
      self.history = np.concatenate((adaptation_input[None, :, None], self.history[:, :, :-1]), axis=2)

    self.count += 1
    self.v_prev = state.vel
    return acc_des, omega_des
  
  def response_torch(self, state, time_step, ref):
    N = state.shape[0]
    state_ref = torch.tensor(ref.ref_vec(time_step * 0.02), dtype=torch.float32)
    state_ref = state_ref.repeat((N,1))

    if time_step == 0:
      self.int_p_e = torch.zeros(N,3)


    q = state[:,6:10]
    q_r = state_ref[:,6:10]
    # position controller
    # p_e = state_ref[:,:3] - state[:,:3]
    # v_e = state_ref[:,3:6] - state[:,3:6]
    p_e = state[:,:3] - state_ref[:, :3]
    v_e = state[:,3:6] - state_ref[:,3:6]

    #int_p_e = self.integrate_error(p_e, time)
    # int_p_e = self.integrate_error(p_e, time_step)
    self.int_p_e += 0.02 * p_e
    F_d = - (self.ki_pos * self.int_p_e + self.kp_pos * p_e + self.kd_pos * v_e) 
    F_d[:, 2] += self.g

    F_d_body = transforms.quaternion_apply(q, F_d)
    F_d_mag = torch.linalg.norm(F_d_body, dim=-1)

    rot_err = torch.cross((F_d_body / F_d_mag[:, None]).float(), torch.tensor([[0, 0, 1]]).float(), dim=-1)
    eulers = transforms.quaternion_to_axis_angle(q)
    yaw = eulers[:, 2]
    eulers_ref = transforms.quaternion_to_axis_angle(q_r)
    yaw_ref = eulers_ref[:, 2]

    omega_des = -self.kp_ang * rot_err
    omega_des[:, 2] += -self.yaw_gain * (yaw - yaw_ref)

    # if self.Adrag is not None:
    #   F_d = F_d - (self.Adrag @ state[:,3:6].T).T 
    # T_d = torch.linalg.norm(F_d,dim=1)

    # attitude controller


    # rpy = transforms.quaternion_to_axis_angle(q)
    # rpy_r = transforms.quaternion_to_axis_angle(q_r)

    # z_d_world = torch.divide(F_d, torch.outer(T_d, torch.ones(3))) 
    # z_d_body = qrotate_torch(qconjugate_torch(q).float(), z_d_world.float())
    # temp = torch.tensor([0, 0, 1], dtype=torch.float32).repeat([N,1])
    # att_e = torch.cross(temp, z_d_body.float(), dim=1)
    # att_e[:,2] = rpy_r[:,2] - rpy[:,2]
    # omega_des = self.kp_ang * att_e

    if N == 1:
      #print(T_d.shape,torque.shape,state.shape)
      motorForce = torch.flatten(torch.concatenate((F_d_mag[None,:], omega_des),dim=1))
      #motorForce = torch.clip(motorForce, self.a_min, self.a_max)
    else:
      motorForce = torch.concatenate((F_d_mag[:,None], omega_des),dim=1)

    return motorForce

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


def to_matrix(q, require_unit=True):
	#q = np.asarray(q)
	s = torch.linalg.norm(q)
	m = torch.empty(q.shape[:-1] + (3, 3))
	s = s ** (-1.0)  # For consistency with Wikipedia notation
	m[..., 0, 0] = 1.0 - 2 * s * (q[..., 2] ** 2 + q[..., 3] ** 2)
	m[..., 0, 1] = 2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
	m[..., 0, 2] = 2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
	m[..., 1, 0] = 2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
	m[..., 1, 1] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 3] ** 2)
	m[..., 1, 2] = 2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
	m[..., 2, 0] = 2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
	m[..., 2, 1] = 2 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0])
	m[..., 2, 2] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)
	return m

def to_euler(q, convention="zyx", axis_type="intrinsic"):  # noqa: C901
	#q = np.asarray(q)
	#_validate_unit(q)
	atol = 1e-3

	try:
		# Due to minor numerical imprecision, the to_matrix function could
		# generate a (very slightly) nonorthogonal matrix (e.g. with a norm of
		# 1 + 2e-8). That is sufficient to throw off the trigonometric
		# functions, so it's worthwhile to explicitly clip for safety,
		# especially since we've already checked the quaternion norm.
		mats = torch.clip(to_matrix(q), -1, 1)
	except ValueError:
		raise ValueError("Not all quaternions in q are unit quaternions.")

	# Classical Euler angles
	beta = torch.arcsin(mats[..., 0, 2])
	multiplier = mats[..., 0, 2] if axis_type == "extrinsic" else 1
	where_zero = torch.isclose(torch.cos(beta), torch.tensor(0, dtype=torch.float32), atol=atol)

	gamma = torch.where(where_zero, 0, torch.arctan2(-mats[..., 0, 1], mats[..., 0, 0]))
	alpha = torch.where(where_zero, 0, torch.arctan2(-mats[..., 1, 2], mats[..., 2, 2]))
	zero_terms = torch.arctan2(multiplier * mats[..., 2, 1], mats[..., 1, 1])

	# By convention, the zero terms that we calculate are always based on
	# setting gamma to zero and applying to alpha. We assign them after the
	# fact to enable the memcopy-free swap of alpha and gamma for extrinsic
	# angles. For Python 2 compatibility, we need to index appropriately.
	try:
		alpha[where_zero] = zero_terms[where_zero]
	except IndexError:
		# This is necessary for Python 2 compatibility and limitations with the
		# indexing behavior. Since the only possible case is a single set of
		# inputs, we can just skip any indexing and overwrite directly if
		# needed.
		if where_zero:
			alpha = zero_terms
	return torch.stack((alpha, beta, gamma), axis=-1)
                
class PIDController_explore(PIDController):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.N = 500

    self.kernel = AirDragKernel()
    self.dimx = 13
    self.H = 500
    self.dimu = 4
    self.dimphi = self.kernel.get_dim()
    self.dimz = self.dimphi

    self.power = torch.tensor(0.0001)
    self.U_init = torch.zeros(self.dimu, self.H)
    self.epoch_power = 0
    self.planning_horizon = 10

    self.cov = torch.zeros(self.dimz, self.dimz)

    self.state = State_struct()

    self.plan_stab_controller = PIDController(**kwargs)

    self.hessian = torch.tensor([[ 1.0951e+00,  6.0636e-02, -1.8848e-01, -2.3686e-01,  2.3872e-01,
         -2.7995e-02,  7.9846e-02, -3.1138e-02, -4.4298e-02],
        [ 6.0636e-02,  3.3575e-03, -1.0436e-02, -1.3116e-02,  1.3218e-02,
         -1.5501e-03,  4.4212e-03, -1.7242e-03, -2.4528e-03],
        [-1.8848e-01, -1.0436e-02,  3.2440e-02,  4.0768e-02, -4.1087e-02,
          4.8184e-03, -1.3743e-02,  5.3594e-03,  7.6243e-03],
        [-2.3686e-01, -1.3116e-02,  4.0768e-02,  5.1233e-02, -5.1635e-02,
          6.0553e-03, -1.7271e-02,  6.7352e-03,  9.5816e-03],
        [ 2.3872e-01,  1.3218e-02, -4.1087e-02, -5.1635e-02,  5.2039e-02,
         -6.1027e-03,  1.7406e-02, -6.7879e-03, -9.6566e-03],
        [-2.7995e-02, -1.5501e-03,  4.8184e-03,  6.0553e-03, -6.1027e-03,
          7.1567e-04, -2.0412e-03,  7.9603e-04,  1.1324e-03],
        [ 7.9846e-02,  4.4212e-03, -1.3743e-02, -1.7271e-02,  1.7406e-02,
         -2.0412e-03,  5.8218e-03, -2.2704e-03, -3.2299e-03],
        [-3.1138e-02, -1.7242e-03,  5.3594e-03,  6.7352e-03, -6.7879e-03,
          7.9603e-04, -2.2704e-03,  8.8542e-04,  1.2596e-03],
        [-4.4298e-02, -2.4528e-03,  7.6243e-03,  9.5816e-03, -9.6566e-03,
          1.1324e-03, -3.2299e-03,  1.2596e-03,  1.7919e-03]])
    self.scaling_mat = torch.eye(4)
    self.objective = 'hessian'

  def _response(self, fl=1, **response_inputs):
    t = response_inputs.get('t')
    state = response_inputs.get('state')
    ref = response_inputs.get('ref')
    ref_func_obj = response_inputs.get('ref_func_obj')
    pos = state.pos
    vel = state.vel
    rot = state.rot

    self.state = state
    obs = np.r_[pos, vel, rot.as_quat(), self.state.ang]


    acc_des, omega_des = super()._response(fl=1, **response_inputs)
    ceed_output, self.U_init = self.compute_CEED_term(self.cov, state, ref_func_obj)

    ceed_output = ceed_output.detach().cpu().numpy()
    acc_des += ceed_output[0]
    omega_des += ceed_output[1:]

    u = np.r_[acc_des, omega_des]
    self.cov = self.cov + self.get_cov(torch.tensor(obs), torch.tensor(u))

    self.state.ang = omega_des
    return acc_des, omega_des
  
  def get_cov(self, x, u, parallel=False):
    if parallel:
      z = self.kernel.phi(x,u)
      cov = torch.einsum('ik, jk -> ijk',z,z)
      return cov
    else:
      z = self.kernel.phi(x, u)
      return torch.outer(z,z)
    
  def compute_CEED_term(self, cov, obs, ref):
    power_to_go = self.H * self.power - self.epoch_power

    # if self.planning_horizon is None:
    #   unroll_steps = self.H - self.count
    # else:
    #   unroll_steps = self.planning_horizon
    #   print('yo', unroll_steps, self.H, self.count, self.H - self.count)
    #   if unroll_steps > self.H - self.count:
    #     unroll_steps = self.H - self.count
    #   else:
    #     power_to_go = power_to_go * unroll_steps / (self.H - self.count)
    unroll_steps = self.planning_horizon
    if power_to_go < 0:
      return torch.zeros(self.dimu), torch.zeros(self.dimu, unroll_steps)
    U_init2 = torch.zeros(self.dimu, unroll_steps)
    U_init2[:, 0:unroll_steps-1] = self.U_init[:, 0:unroll_steps - 1]
    self.U_init = U_init2[:, :,  None]
    U = self.U_init.repeat(1, 1, self.N) 
    U = U + torch.randn(self.dimu, unroll_steps, self.N) 

    U = torch.sqrt(power_to_go) * torch.nn.functional.normalize(U, dim=(0,1))

    x = obs.get_vec_state_torch('wxyz')[:, None]
    x = x.repeat(1, self.N)
    cov = cov[:, :, None]
    cov = cov.repeat(1, 1, self.N)

    for h in range(unroll_steps):
      u = U[:, h, :]
      if self.scaling_mat is not None:
        u = self.scaling_mat @ u

        stab_controller_output = self.plan_stab_controller.response_torch(
          time_step=h + self.count,
          state = x.T,
          ref = ref
        )
        u = u + stab_controller_output.T
      cov = cov + self.get_cov(x, u, parallel=True)

      x = self.dynamics(x.T, u.T)

    if self.objective == 'lammin':
      e,_ = torch.linalg.eig(cov[:, :, 0])
      min_loss = torch.min(torch.real(e))
    else:
      cov_inv = torch.linalg.inv(cov[:, :, 0] + 0.001 * torch.eye(self.dimz)).contiguous()
      min_loss = torch.trace(self.hessian.float() @ torch.kron(torch.eye(3).float(), cov_inv.float()))

    if self.objective == 'lammin':
      min_idx = 0
      for i in range(self.N):
        if self.objective == 'lammin':
          e,_ = torch.linalg.eig(cov[:, :, i])
          loss = torch.min(torch.real(e))
          if loss < min_loss:
            min_loss = loss
            min_idx = i
        else:
          cov_inv = torch.linalg.inv(cov[:, :, i] + 0.001 * torch.eye(self.dimz)).contiguous()
          loss = torch.trace(self.hessian.float() @ torch.kron(torch.eye(3).float(), cov_inv.float()))
          if i == 0:
              print(loss)
          if loss < min_loss:
            min_loss = loss
            min_idx = i
    else:
      cov_inv_temp = cov + 0.001 * torch.eye(self.dimz).unsqueeze(-1)
      cov_inv = torch.linalg.inv(cov_inv_temp.permute(2,0,1)).contiguous() 
      kron_product = torch.kron(torch.eye(3).float(), cov_inv.float())
      loss = torch.matmul(self.hessian, kron_product)
      loss = torch.einsum('bii->b', loss) # Shape: [500]

      min_loss, min_idx = loss.squeeze().min(dim=-1)  

    
    if self.scaling_mat is None:
      return U[:, 0, min_idx], U[:, 1:, min_idx]
    else:
      return self.scaling_mat @ U[:, 0, min_idx], U[:, 1:, min_idx]

  def dynamics(self, x, u):
    x += self.f(x, u) * 0.02
    x[:, 10:] = u[:, 1:]
    return x.T

  def f(self, x, a):
    N_temp, _ = x.shape

    dsdt = torch.zeros(N_temp, 13)
    v = x[:, 3:6] # velocity (N, 3)
    q = x[:, 6:10] # quaternion (N, 4)
    omega = x[:, 10:] # angular velocity (N, 3)

    # get input 
    # eta = torch.mm(self.B0, a.T).T # output wrench (N, 4)
    eta = a
    #f_u = torch.zeros(self.N, 3) 
    f_u = torch.zeros(N_temp, 3)
    f_u[:, 2] = eta[:, 0] # total thrust (N, 3)

    # dynamics 
    # \dot{p} = v 
    dsdt[:, :3] = x[:, 3:6] 	# <- implies velocity and position in same frame
    # mv = mg + R f_u  	# <- implies f_u in body frame, p, v in world frame
    dsdt[:, 5] -= self.g
    dsdt[:, 3:6] += qrotate_torch(q.float(), f_u.float()) / self.mass

    # \dot{R} = R S(w)
    # see https://rowan.readthedocs.io/en/latest/package-calculus.html
    qnew = qintegrate_torch(q, omega, 0.02, frame='body')
    qnew = qstandardize_torch(qnew)
    # transform qnew to a "delta q" that works with the usual euler integration
    dsdt[:, 6:10] = (qnew - q) / 0.02
    
    return dsdt

