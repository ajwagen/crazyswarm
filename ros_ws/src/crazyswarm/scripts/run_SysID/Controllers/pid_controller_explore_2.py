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
import sys
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

def xyzw(quat):
	w = quat[0]
	quat[0:3] = quat[1:]
	quat[-1] = w
	return quat

# class PIDController(ControllerBackbone):
#   def __init__(self, **kwargs):
#     super().__init__(**kwargs)

#     if self.explore_type=='none':
#       gains = np.load(self.exploration_dir + 'gains.npy')
#       print(gains)
#       self.kp_pos = gains[0]
#       self.kd_pos = gains[1]
#       self.ki_pos = gains[2] # 0 for sim
#       # self.kp_pos = 6
#       # self.kd_pos = 4
#       # self.ki_pos = 1.5 # 0 for sim
#     else:
#       self.kp_pos = 6
#       self.kd_pos = 4
#       self.ki_pos = 1.5


#     self.kp_rot =   150.0/16
#     self.yaw_gain = 220.0/16
#     # self.kp_ang =   16

#     self.pos_err_int = np.zeros(3)
#     self.count = 0
#     self.lamb = 0.2
#     self.v_prev = np.zeros(3)

#     self.history = np.zeros((1, 14, 50))
#     self.int_p_e = torch.zeros((500, 3), dtype=torch.float32)

#     if self.explore_type == 'none':
#       self.Adrag_np = np.load(self.exploration_dir+'aker.npy')
#       self.Adrag = torch.tensor(self.Adrag_np)
#       self.traj_ceed = None
#     else:
#       self.Adrag = torch.tensor(0.01*np.random.randn(3,3), dtype=torch.float32)
#       self.Adrag_np = self.Adrag.detach().cpu().numpy()
#       self.traj_ceed = np.load(self.exploration_dir+'traj.npy').T

#     # self.mppi_controller = self.set_MPPI_cnotroller()
#     # self.runL1 = False

#   def _response(self, fl=1, **response_inputs ):
    
#     t = response_inputs.get('t')
#     state = response_inputs.get('state')
#     ref = response_inputs.get('ref')


#     self.updateDt(t)
#     # if fl:
#     if self.prev_t != None:
#       dt = t - self.prev_t
#     else:
#       dt = 0.02
#     #   self.prev_t = t

#     # PID
#     pos = state.pos
#     vel = state.vel
#     rot = state.rot
#     p_err = (pos - ref.pos)
#     v_err = (vel - ref.vel)
#     quat = rot.as_quat() 
#     self.pos_err_int += p_err * self.dt
#     obs = np.hstack((pos, vel, quat))

#     if self.count > 2:
#       vel = state.vel
#       a_t = (vel - self.v_prev) / dt
#     else:
#       a_t = np.array([0, 0, 0])

#     unity_mass = 1
#     f_t = rot.apply(np.array([0, 0, self.history[0, 10, 0]])) * unity_mass


#     acc_des = (np.array([0, 0, self.g]) 
#               - self.kp_pos * (p_err) 
#               - self.kd_pos * (v_err) 
#               - self.ki_pos * self.pos_err_int
#               - self.Adrag_np @ vel)
    

#     u_des = rot.as_matrix().T.dot(acc_des)
#     acc_des = np.linalg.norm(u_des)

#     rot_err = np.cross(u_des / acc_des, np.array([0, 0, 1]))
#     eulers = rot.as_euler("ZYX")
#     yaw = eulers[0]

#     omega_des = - self.kp_rot * rot_err
#     omega_des[2] += - self.yaw_gain * (yaw - 0.0)
      
#     adaptation_input = np.r_[obs, acc_des, omega_des]
#     if fl!=0.0:
#       self.history = np.concatenate((adaptation_input[None, :, None], self.history[:, :, :-1]), axis=2)

#     self.count += 1
#     self.v_prev = state.vel
#     return acc_des, omega_des
  
#   def response_torch(self, state, time_step, ref):
#     N = state.shape[0]
#     state_ref = torch.tensor(ref.ref_vec(time_step * 0.02), dtype=torch.float32)
#     state_ref = state_ref.repeat((N,1))

#     if time_step == 0:
#       self.int_p_e = torch.zeros(N,3)


#     q = state[:,6:10]
#     q_r = state_ref[:,6:10]
#     # position controller
#     # p_e = state_ref[:,:3] - state[:,:3]
#     # v_e = state_ref[:,3:6] - state[:,3:6]
#     p_e = 0 * (state[:,:3] - state_ref[:, :3] )
#     v_e = 0 * (state[:,3:6] - state_ref[:,3:6])

#     #int_p_e = self.integrate_error(p_e, time)
#     # int_p_e = self.integrate_error(p_e, time_step)
#     self.int_p_e += 0.02 * p_e
#     F_d = - (self.ki_pos * self.int_p_e + self.kp_pos * p_e + self.kd_pos * v_e) 
#     F_d[:, 2] += self.g
#     # if self.Adrag is not None:
#     F_d = F_d.float() - (self.Adrag.float() @ state[:,3:6].float().T).T 

#     F_d_body = transforms.quaternion_apply(q, F_d)
#     F_d_mag = torch.linalg.norm(F_d_body, dim=-1)

#     rot_err = torch.cross((F_d_body / F_d_mag[:, None]).float(), torch.tensor([[0, 0, 1]]).float(), dim=-1)
#     eulers = transforms.quaternion_to_axis_angle(q)
#     yaw = eulers[:, 2]
#     eulers_ref = transforms.quaternion_to_axis_angle(q_r)
#     yaw_ref = eulers_ref[:, 2]

#     omega_des = -self.kp_rot * rot_err
#     omega_des[:, 2] += -self.yaw_gain * (yaw - yaw_ref)


#     T_d = torch.linalg.norm(F_d,dim=1)

#     if N == 1:
#       motorForce = torch.flatten(torch.concatenate((F_d_mag[None,:], omega_des),dim=1))
#       #motorForce = torch.clip(motorForce, self.a_min, self.a_max)
#     else:
#       motorForce = torch.concatenate((F_d_mag[:,None], omega_des),dim=1)

#     return motorForce

# class AirDragKernel:
# 	def __init__(self):
# 		return

# 	def get_dim(self):
# 		return 3

# 	def phi(self,x,u):
# 		if len(x.shape) > 1:
# 			return x[3:6,:]
# 		else:
# 			return x[3:6]


# def to_matrix(q, require_unit=True):
# 	#q = np.asarray(q)
# 	s = torch.linalg.norm(q)
# 	m = torch.empty(q.shape[:-1] + (3, 3))
# 	s = s ** (-1.0)  # For consistency with Wikipedia notation
# 	m[..., 0, 0] = 1.0 - 2 * s * (q[..., 2] ** 2 + q[..., 3] ** 2)
# 	m[..., 0, 1] = 2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0])
# 	m[..., 0, 2] = 2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0])
# 	m[..., 1, 0] = 2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0])
# 	m[..., 1, 1] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 3] ** 2)
# 	m[..., 1, 2] = 2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0])
# 	m[..., 2, 0] = 2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0])
# 	m[..., 2, 1] = 2 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0])
# 	m[..., 2, 2] = 1.0 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)
# 	return m

# def to_euler(q, convention="zyx", axis_type="intrinsic"):  # noqa: C901
# 	#q = np.asarray(q)
# 	#_validate_unit(q)
# 	atol = 1e-3

# 	try:
# 		# Due to minor numerical imprecision, the to_matrix function could
# 		# generate a (very slightly) nonorthogonal matrix (e.g. with a norm of
# 		# 1 + 2e-8). That is sufficient to throw off the trigonometric
# 		# functions, so it's worthwhile to explicitly clip for safety,
# 		# especially since we've already checked the quaternion norm.
# 		mats = torch.clip(to_matrix(q), -1, 1)
# 	except ValueError:
# 		raise ValueError("Not all quaternions in q are unit quaternions.")

# 	# Classical Euler angles
# 	beta = torch.arcsin(mats[..., 0, 2])
# 	multiplier = mats[..., 0, 2] if axis_type == "extrinsic" else 1
# 	where_zero = torch.isclose(torch.cos(beta), torch.tensor(0, dtype=torch.float32), atol=atol)

# 	gamma = torch.where(where_zero, 0, torch.arctan2(-mats[..., 0, 1], mats[..., 0, 0]))
# 	alpha = torch.where(where_zero, 0, torch.arctan2(-mats[..., 1, 2], mats[..., 2, 2]))
# 	zero_terms = torch.arctan2(multiplier * mats[..., 2, 1], mats[..., 1, 1])

# 	# By convention, the zero terms that we calculate are always based on
# 	# setting gamma to zero and applying to alpha. We assign them after the
# 	# fact to enable the memcopy-free swap of alpha and gamma for extrinsic
# 	# angles. For Python 2 compatibility, we need to index appropriately.
# 	try:
# 		alpha[where_zero] = zero_terms[where_zero]
# 	except IndexError:
# 		# This is necessary for Python 2 compatibility and limitations with the
# 		# indexing behavior. Since the only possible case is a single set of
# 		# inputs, we can just skip any indexing and overwrite directly if
# 		# needed.
# 		if where_zero:
# 			alpha = zero_terms
# 	return torch.stack((alpha, beta, gamma), axis=-1)
                
class PIDController_explore(ControllerBackbone):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.N = 500
    self.pid_controller = self.set_PID_torch()
    self.count = 0
    # # self.int_p_e = torch.zeros((self.N, 3), dtype=torch.float32)
    # # print(self.int_p_e.shape)
    # # exit()

    # self.kernel = AirDragKernel()
    # self.dimx = 13
    # self.H = 500
    # self.dimu = 4
    # self.dimphi = self.kernel.get_dim()
    # self.dimz = self.dimphi

    # self.power = torch.tensor(0.0001)
    # self.U_init = torch.zeros(self.dimu, self.N)
    # self.epoch_power = 0
    # self.planning_horizon = 10
    # self.state = State_struct()

    # self.plan_stab_controller = PIDController(**kwargs)
    
    # if not self.init_run:
    #   hessian = np.load(self.exploration_dir+'hess.npy')
    #   self.cov = torch.tensor(np.load(self.exploration_dir+'cov.npy'))
    #   self.hessian = torch.tensor(hessian)
    #   self.Aker = torch.tensor(np.load(self.exploration_dir+'aker.npy'))
    # else:
    #   self.hessian = None
    #   self.cov = torch.zeros(self.dimz,self.dimz)
    #   self.Aker = None

    # self.scaling_mat = torch.eye(4)
    # self.objective = 'hessian'
    # self.input_power = torch.tensor(0, dtype=torch.float32)
    # self.max_power = torch.tensor(2.0)


    # arm_length = 0.046 # m
    # arm = 0.707106781 * arm_length
    # t2t = 0.006 # thrust-to-torque ratio
    # self.B0 = torch.tensor([
    #   [1, 1, 1, 1],
    #   [-arm, -arm, arm, arm],
    #   [-arm, arm, arm, -arm],
    #   [-t2t, t2t, -t2t, t2t]
    #   ])
    # self.J = torch.diag(torch.tensor([16.571710e-6, 16.655602e-6, 29.261652e-6]))
    # self.J_inv = torch.linalg.inv(self.J)

    # self.padding = torch.zeros(self.dimx,3)
    # self.padding[3:6,:] = torch.eye(3)

  def _response(self, fl=1, **response_inputs):
    t = response_inputs.get('t')
    state = response_inputs.get('state')
    ref = response_inputs.get('ref')
    ref_func_obj = response_inputs.get('ref_func_obj')


    pos = state.pos - self.offset_pos
    vel = state.vel
    rot = state.rot

    self.state = state

    obs = np.r_[pos, vel, np.roll(rot.as_quat(), 1), self.state.ang]

    self.pid_controller.set_ref_traj(ref_func_obj)

    u = self.pid_controller.get_input(torch.tensor(obs, dtype=torch.float), self.count)
    if self.explore_type == 'random':
      u += self.compute_random_term()
    u = u.detach().cpu().numpy()
    self.count += 1

    acc_des = u[0]
    omega_des = u[1:]


    # if self.explore_type=='traj_ceed':
    #   # ref_state = State_struct()
    #   # ref_state.pos = self.traj_ceed[self.count][:3]
    #   # ref_state.vel = self.traj_ceed[self.count][3:6]
    #   # response_inputs['ref'] = ref_state
    #   acc_des, omega_des = super()._response(fl=1, **response_inputs)

    # else:
    #   acc_des, omega_des = super()._response(fl=1, **response_inputs)

    #   if self.explore_type != 'none':
    #     if self.explore_type == 'ceed':
    #       ceed_output, self.U_init = self.compute_ceed_term(self.cov, state, ref_func_obj)
    #     else:
    #       ceed_output = self.compute_random_term()
            
    #     ceed_output = ceed_output.detach().cpu().numpy()
    #     acc_des += ceed_output[0]
    #     omega_des += ceed_output[1:]

    # u = np.r_[acc_des, omega_des]
    # self.cov = self.cov + self.get_cov(torch.tensor(obs), torch.tensor(u))
    # # print(self.get_cov(torch.tensor(obs), torch.tensor(u)))
    # # print(self.cov)
    # # print('-----------')
    # self.state.ang = omega_des
    return acc_des, omega_des
  
  def compute_random_term(self,):
      u_std = torch.tensor([1.0, 1.0, 1.0, 1.0])
      u_mean = torch.tensor([0., 0., 0., 0.])
      U = torch.normal(u_mean, u_std)
      # u = torch.sqrt(self.max_power / self.dimu) * torch.randn(self.dimu)
      # self.input_power = self.input_power + u @ u
      return U
  # def get_cov(self, x, u, parallel=False):
  #   if parallel:
  #     z = self.kernel.phi(x,u)
  #     cov = torch.einsum('ik, jk -> ijk',z,z)
  #     return cov
  #   else:
  #     z = self.kernel.phi(x, u)
  #     return torch.outer(z,z)
  
  # def compute_random_term(self,):
  #     u_std = torch.tensor([1.0, 1.0, 1.0, 1.0])
  #     u_mean = torch.tensor([0., 0., 0., 0.])
  #     U = torch.normal(u_mean, u_std)
  #     # u = torch.sqrt(self.max_power / self.dimu) * torch.randn(self.dimu)
  #     # self.input_power = self.input_power + u @ u
  #     return U
  
  # def compute_ceed_term(self, cov, obs, ref):
  #   power_to_go = self.H * self.max_power - self.epoch_power

  #   # if self.planning_horizon is None:
  #   #   unroll_steps = self.H - self.count
  #   # else:
  #   #   unroll_steps = self.planning_horizon
  #   #   print('yo', unroll_steps, self.H, self.count, self.H - self.count)
  #   #   if unroll_steps > self.H - self.count:
  #   #     unroll_steps = self.H - self.count
  #   #   else:
  #   #     power_to_go = power_to_go * unroll_steps / (self.H - self.count)
  #   unroll_steps = self.planning_horizon
  #   if power_to_go < 0:
  #     return torch.zeros(self.dimu), torch.zeros(self.dimu, unroll_steps)
  #   # U_init2 = torch.zeros(self.dimu, unroll_steps)
  #   # U_init2[:, 0:unroll_steps-1] = self.U_init[:, 0:unroll_steps - 1]
  #   # self.U_init = U_init2[:, :,  None]
  #   # U = self.U_init.repeat(1, 1, self.N) 
  #   # U = U + torch.randn(self.dimu, unroll_steps, self.N)
  #   u_std = torch.tensor([0.25, 60.0, 60.0, 10.0])
  #   u_std = u_std[:, None, None]
  #   u_std = u_std.repeat(1, unroll_steps, self.N)

  #   u_mean = torch.tensor([0., 0., 0., 0.])
  #   u_mean = u_mean[:, None, None]
  #   u_mean = u_mean.repeat(1, unroll_steps, self.N)

  #   U = torch.normal(u_mean, u_std)
  #   # for i in range(self.N):
  #   #     U[:,:,i] = torch.sqrt(power_to_go) * U[:,:,i] / torch.norm(U[:,:,i])
  #   # U = u_mean

  #   x = obs.get_vec_state_torch('wxyz')[:, None]
  #   x = x.repeat(1, self.N)
  #   cov = cov[:, :, None]
  #   cov = cov.repeat(1, 1, self.N)

  #   for h in range(unroll_steps):
  #     u = U[:, h, :]
  #     if self.scaling_mat is not None:
  #       stab_controller_output = self.plan_stab_controller.response_torch(
  #         time_step=h + self.count,
  #         state = x.T,
  #         ref = ref
  #       )
  #       u = u + stab_controller_output.T
  #     cov = cov + self.get_cov(x, u, parallel=True)
  #     x = self.dynamics(x.T, u.T)

  #   if self.objective == 'lammin':
  #     e,_ = torch.linalg.eig(cov[:, :, 0])
  #     min_loss = torch.min(torch.real(e))
  #   else:
  #     cov_inv = torch.linalg.inv(cov[:, :, 0] + 0.001 * torch.eye(self.dimz)).contiguous()
  #     min_loss = torch.trace(self.hessian.float() @ torch.kron(torch.eye(3).float(), cov_inv.float()))

  #       # pass
  #   # min_idx = 0
  #   # for i in range(self.N):
  #     # if self.objective == 'lammin':
  #       # e,_ = torch.linalg.eig(cov[:, :, i])
  #       # loss = torch.min(torch.real(e))
  #       # if loss < min_loss:
  #       #   min_loss = loss
  #       #   min_idx = i
  #   #   else:
  #   #     cov_inv = torch.linalg.inv(cov[:, :, i] + 0.001 * torch.eye(self.dimz)).contiguous()
  #   #     loss = torch.trace(self.hessian.float() @ torch.kron(torch.eye(3).float(), cov_inv.float()))
  #   #     # if i == 0:
  #   #         # print(loss)
  #   #     if loss < min_loss:
  #   #       min_loss = loss
  #   #       min_idx = i
  #   # else:

  #   e,_ = torch.linalg.eig(cov.permute(2,1,0))
  #   cov_inv_temp = cov + 0.001 * torch.eye(self.dimz).unsqueeze(-1)
  #   cov_inv = torch.linalg.inv(cov_inv_temp.permute(2,0,1)).contiguous() 
  #   kron_product = torch.kron(torch.eye(3).float(), cov_inv.float())
  #   loss = torch.matmul(self.hessian, kron_product)
  #   loss = torch.einsum('bii->b', loss) # Shape: [500]
  #   min_loss, min_idx = loss.squeeze().min(dim=-1)
  #   max_loss, _ = loss.squeeze().max(dim=-1)
  #   U_ = torch.sum((U * (max_loss - loss) / torch.sum(max_loss - loss))[:,:,:200], dim = -1)

  #   return U_[:, 0], U_[:, 1:]
  #   # if self.scaling_mat is None:
  #   #   return U[:, 0, min_idx], U[:, 1:, min_idx]
  #   # else:
  #   #   return self.scaling_mat @ U[:, 0, min_idx], U[:, 1:, min_idx]

  # def dynamics(self, x, u):
  #   x += self.f(x, u) * 0.02
  #   x[:, 10:] = u[:, 1:]

  #   x += (self.padding @ self.Aker.float() @ self.kernel.phi(x.T,u.T).float()).T
  #   return x.T

  # def f(self, x, a):
  #   N_temp, _ = x.shape

  #   dsdt = torch.zeros(N_temp, 13)
  #   v = x[:, 3:6] # velocity (N, 3)
  #   q = x[:, 6:10] # quaternion (N, 4)
  #   omega = x[:, 10:] # angular velocity (N, 3)

  #   # get input 
  #   # eta = torch.mm(self.B0, a.T).T # output wrench (N, 4)
  #   eta = a
  #   #f_u = torch.zeros(self.N, 3) 
  #   f_u = torch.zeros(N_temp, 3)
  #   f_u[:, 2] = eta[:, 0] # total thrust (N, 3)
  #   tau_u = eta[:, 1:] # torque (N, 3)

  #   # dynamics 
  #   # \dot{p} = v 
  #   dsdt[:, :3] = x[:, 3:6] 	# <- implies velocity and position in same frame
  #   # mv = mg + R f_u  	# <- implies f_u in body frame, p, v in world frame
  #   dsdt[:, 5] -= self.g
  #   dsdt[:, 3:6] += qrotate_torch(q.float(), f_u.float())

  #   # \dot{R} = R S(w)
  #   # see https://rowan.readthedocs.io/en/latest/package-calculus.html
  #   qnew = qintegrate_torch(q, omega, 0.02, frame='body')
  #   qnew = qstandardize_torch(qnew)
  #   # transform qnew to a "delta q" that works with the usual euler integration
  #   dsdt[:, 6:10] = (qnew - q) / 0.02
  #   # print(self.J.shape, omega.shape)
  #   # Jomega = torch.mm(self.J, omega.T).T # J*omega (N, 3)
  #   # dsdt[:, 10:] = torch.cross(Jomega, omega) + tau_u
  #   # dsdt[:, 10:] = torch.mm(self.inv_J, dsdt[:, 10:].T).T

  #   return dsdt

