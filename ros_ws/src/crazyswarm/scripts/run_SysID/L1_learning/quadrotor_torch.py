from gym import Env
import gym
import numpy as np
import torch
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from param_torch import Param
from param_torch import Timer
import time
from math_utils import *
import rowan
try:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
except:
	torch.set_default_tensor_type('torch.FloatTensor')


class Quadrotor(Env):
	def __init__(self, param : Param):

		# init
		self.times = param.sim_times
		self.s = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
		self.time_step = 0
		self.param = param
		self.ref_traj_func = self.param.ref_traj_func

		# system dimensions 
		# state = [position, velocity, quaternion (wxyz), angular velocity] in SI
		# action = [force_1, force_2, force_3, force_4] in N
		self.n = 13
		self.m = 4

		self.limits = param.limits

		# parameters
		self.mass = param.mass
		self.J = np.ones(3)
		self.g = np.array([0,0,-param.g])
		self.inv_mass = 1 / self.mass
		if self.J.shape == (3,3):
			self.inv_J = np.linalg.pinv(self.J) # full matrix -> pseudo inverse
		else:
			self.inv_J = 1 / self.J # diagonal matrix -> division
		self.B0 = param.B0
		self.B0_inv = np.linalg.inv(self.B0)
		self.d = param.d
		self.rho = param.rho
		self.Cs = param.Cs
		self.Ct = param.Ct
		self.k1 = param.k1
		self.k2 = param.k2

		# reward function stuff
		# ref: row 8, Table 3, USC sim-to-real paper
		self.alpha_p = param.alpha_p
		self.alpha_w = param.alpha_w
		self.alpha_a = param.alpha_a
		self.alpha_R = param.alpha_R
		self.alpha_v = param.alpha_v
		self.alpha_yaw = param.alpha_yaw

		#self.isangvel = param.isangvel
		self.isangvel = param.isangvel
		self.input_torque = False
		
		# plotting stuff
		self.states_name = [
			'Position X [m]',
			'Position Y [m]',
			'Position Z [m]',
			'Velocity X [m/s]',
			'Velocity Y [m/s]',
			'Velocity Z [m/s]',
			'qw',
			'qx',
			'qy',
			'qz',
			'Angular Velocity X [rad/s]',
			'Angular Velocity Y [rad/s]',
			'Angular Velocity Z [rad/s]']
		self.deduced_state_names = [
			'Roll [deg]',
			'Pitch [deg]',
			'Yaw [deg]',
		]
		self.actions_name = [
			'Motor Force 1 [N]',
			'Motor Force 2 [N]',
			'Motor Force 3 [N]',
			'Motor Force 4 [N]']


	def step(self,a):
		self.s = self.next_state(self.s, a)
		r = self.reward(a)
		self.time_step += 1
		return self.s, r

	def reward(self, a):
		# see USC sim-to-real paper, eq (14)
		ref_func = self.param.ref_traj_func
		ep = np.linalg.norm(self.s[0:3] - ref_func.pos(self.time_step * self.param.dt))
		ev = np.linalg.norm(self.s[3:6] - ref_func.vel(self.time_step * self.param.dt))
		ew = np.linalg.norm(self.s[10:13] - ref_func.angvel(self.time_step * self.param.dt))
		eR = rowan.geometry.sym_distance(self.s[6:10], ref_func.quat(self.time_step * self.param.dt))

		# compute yaw error
		Rd = qtoR(ref_func.quat(self.time_step * self.param.dt))
		R = qtoR(self.s[6:10])
		Re = Rd.T @ R
		eyaw = np.arctan2(Re[1,0], Re[0,0]) ** 2
		
		cost = (
			   self.alpha_p * ep \
			 + self.alpha_v * ev \
			 + self.alpha_w * ew \
			 + self.alpha_a * np.linalg.norm(a) \
			 + self.alpha_yaw * eyaw \
			 + self.alpha_R * eR) * self.param.dt

		return -cost

	def reset(self, initial_state = None):
		if initial_state is None:
			self.s = np.empty(self.n)
			# position and velocity
			limits = self.limits
			self.s[0:6] = np.random.uniform(-limits[0:6], limits[0:6], 6)
			# rotation
			rpy = np.radians(np.random.uniform(-self.rpy_limit, self.rpy_limit, 3))
			q = rowan.from_euler(rpy[0], rpy[1], rpy[2], 'xyz')
			self.s[6:10] = q
			# angular velocity
			self.s[10:13] = np.random.uniform(-limits[10:13], limits[10:13], 3)
		else:
			self.s = initial_state
		self.time_step = 0
		return np.array(self.s)


	# dsdt = f(s,a)
	def f(self,s,a):
		# input:
		# 	s, nd array, (n,)
		# 	a, nd array, (m,1)
		# output
		# 	dsdt, nd array, (n,1)

		dsdt = np.zeros(self.n)
		q = s[6:10]
		omega = s[10:]

		# get input 
		# a = np.reshape(a,(self.m,))
		# eta = np.dot(self.B0,a)
		eta = a
		f_u = np.array([0,0,eta[0]])
		tau_u = np.array([eta[1],eta[2],eta[3]])

		# dynamics 
		# dot{p} = v 
		dsdt[0:3] = s[3:6] 	# <- implies velocity and position in same frame
		# mv = mg + R f_u  	# <- implies f_u in body frame, p, v in world frame
		dsdt[3:6] = self.g + qrotate(q,f_u)
		
		# dot{R} = R S(w)
		# to integrate the dynamics, see
		# https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/
		qnew = qintegrate(q, omega, self.param.dt, frame='body')
		qnew = qstandardize(qnew)
		# transform qnew to a "delta q" that works with the usual euler integration
		dsdt[6:10] = (qnew - q) / self.param.dt

		if self.isangvel:
			return dsdt
		# # mJ = Jw x w + tau_u 
		dsdt[10:] = self.inv_J * (np.cross(self.J * omega,omega) + tau_u)
		
		# # wind model (force and torque)
		# if self.param.wind:
		# 	Vinf = self.param.Vwind - np.linalg.norm(s[3:6]) # velocity of air relative to quadrotor in world frame orientation
		# 	Vinf_B = qrotate(qconjugate(q), Vinf)	 # use _B suffix to denote vector is in body frame
		# 	Vz_B = np.array([0.0, 0.0, Vinf_B[2]]) 		     # z-body-axis aligned velocity
		# 	Vs_B = Vinf_B - Vz_B						     # cross wind
		# 	alpha = np.arcsin(np.linalg.norm(Vz_B)/np.linalg.norm(Vinf_B))
		# 	n = np.sqrt(np.multiply(a, self.B0[0,:]) / (self.Ct * self.rho * self.d**4))
		# 	Fs_B = (Vs_B/np.linalg.norm(Vs_B)) * self.Cs * self.rho * sum(n**self.k1) * (np.linalg.norm(Vinf)**(2-self.k1)) * (self.d**(2+self.k1)) * ((np.pi/2)**2 - alpha**2) * (alpha + self.k2)
		# 	# print(alpha, n, self.g, rowan.rotate(q,f_u) / self.mass, Fs_B)
		# 	dsdt[3:6] += qrotate(q, Fs_B) / self.mass 
		# 	dsdt[10:] += self.inv_J * np.cross(np.array([0.0, 0.0, self.d/4]), Fs_B)
		
		# # adding process noise
		noise_tran = np.random.normal(scale=self.param.noise_process_std[0], size=3)
		noise_rot = np.random.normal(scale=self.param.noise_process_std[1], size=3)
		dsdt[3:6] += noise_tran
		dsdt[10:] += noise_rot
		return dsdt


	def next_state(self,s,a):
		if self.input_torque:
			a[:, 1:] = self.torque_to_angvel(s,a)
		dt = self.param.dt
		dsdt = self.f(s,a)
		s = s + dsdt * dt

		if self.isangvel:
			s[10:] = a[1:]
		return s

	# def next_state(self,s,a,noiseless=False):
	# 	if self.input_torque:
	# 		a[:, 1:] = self.torque_to_angvel(s,a)
	# 	s = s + self.ave_dt * self.f(s,a,noiseless=noiseless)
	# 	if self.isangvel:
	# 		s[:, 10:] = a[:, 1:]
	# 	return s

	def torque_to_angvel(self,s,a):
		tau_u = a[:, 1:]
		omega = s[:, 10:]
		Jomega = np.mm(self.J, omega.T).T 
		omega_dsdt = np.cross(Jomega, omega) + tau_u
		omega_dsdt = np.mm(self.inv_J, omega_dsdt.T).T
		return self.ave_dt * omega_dsdt

	def deduce_state(self, s):
		rpy = np.degrees(rowan.to_euler(s[6:10], 'xyz'))
		return rpy

	def visualize(self,states,dt):
		# Create a new visualizer
		vis = meshcat.Visualizer()
		vis.open()

		vis["/Cameras/default"].set_transform(
			tf.translation_matrix([0,0,0]).dot(
			tf.euler_matrix(0,np.radians(-30),-np.pi/2)))

		vis["/Cameras/default/rotated/<object>"].set_transform(
			tf.translation_matrix([1, 0, 0]))

		vis["Quadrotor"].set_object(g.StlMeshGeometry.from_file('./crazyflie2.stl'))
		
		vertices = np.array([[0,0.5],[0,0],[0,0]]).astype(np.float32)
		vis["lines_segments"].set_object(g.Line(g.PointsGeometry(vertices), \
			                             g.MeshBasicMaterial(color=0xff0000,linewidth=100.)))
		
		while True:
			for state in states:
				vis["Quadrotor"].set_transform(
					tf.translation_matrix([state[0], state[1], state[2]]).dot(
					  tf.quaternion_matrix(state[6:10])))
				vis["lines_segments"].set_transform(
					tf.translation_matrix([state[0], state[1], state[2]]).dot(
					  tf.quaternion_matrix(state[6:10])))				
				time.sleep(dt)

class Quadrotor_torch():
	'''
	a torch version of the quadsim which is used for MPPI rollout
	'''
	def __init__(self, param : Param, config):

		# init
		self.times = param.sim_times
		self.time_step = 0
		#self.ave_dt = self.times[1]-self.times[0]
		self.ave_dt = param.dt
		self.param = param
		self.n = 13
		self.m = 4
		self.dt = param.dt # 0.01

		# system dimensions 
		# state = [position, velocity, quaternion (wxyz), angular velocity] in SI
		# action = [force_1, force_2, force_3, force_4] in Newton
		self.N = config[config['controller']]['N']
		self.s = torch.zeros(self.N, 13) # batch state: N x 13
		self.s[:,6] = 1.0

		# initial conditions
		self.s_min = param.s_min
		self.s_max = -self.s_min
		self.rpy_limit = param.rpy_limit
		self.limits = param.limits

		# control bounds
		self.a_min = torch.as_tensor(param.a_min, dtype=torch.float32)
		self.a_max = torch.as_tensor(param.a_max, dtype=torch.float32)

		# parameters
		self.mass = 1
		self.J = self.param.J
		self.g = param.g
		self.inv_mass = 1 / self.mass
		self.J = torch.diag(torch.tensor([1., 1., 1.], dtype=torch.float32))
		if self.J.shape == (3,3):
			self.J = torch.as_tensor(self.J, dtype=torch.float32)
			self.inv_J = torch.linalg.inv(self.J)
		else:
			self.J = torch.diag(torch.as_tensor(self.J, dtype=torch.float32))
			self.inv_J = torch.linalg.inv(self.J)
		self.B0 = torch.as_tensor(param.B0, dtype=torch.float32)
		self.B0_inv = torch.linalg.inv(self.B0)

		# reward function stuff
		# ref: row 8, Table 3, USC sim-to-real paper
		self.ref_trajectory = torch.as_tensor(self.param.ref_trajectory, dtype=torch.float32)
		self.ref_traj_func = self.param.ref_traj_func
		
		self.alpha_p = param.alpha_p
		self.alpha_w = param.alpha_w
		self.alpha_a = param.alpha_a
		self.alpha_R = param.alpha_R
		self.alpha_v = param.alpha_v
		self.alpha_yaw = param.alpha_yaw

		self.isangvel = param.isangvel
		self.input_torque = False
		# timer for code profiling
		self.timer = Timer(topics=['reward', 'wrench', 'pos dynamics', 'att kinematics', 'att dynamics'])


	def done(self):
		if (self.time_step+1) >= len(self.times):		
			return True
		return False


	def step(self,a):
		self.s = self.next_state(self.s, a, noiseless=True)
		# import pdb;pdb.set_trace()
		d = self.done()
		r = self.reward(a)
		self.time_step += 1
		return self.s, r, False


	def reward(self,a,s=None,time_step=None):
		# see USC sim-to-real paper, eq (14)
		# input:
		#   a, tensor, (N, 4)
		# output:
		#   r, tensor, (N,)
		if time_step is None:
			time_step = self.time_step
		self.timer.tic()
		# state_ref = self.ref_trajectory[:, self.time_step]
		# p_des = state_ref[0:3]
		# v_des = state_ref[3:6]
		# w_des = state_ref[10:]
		# q_des = state_ref[6:10]
		# p_des = torch.from_numpy(self.ref_traj_func.pos(time_step * 0.01)).to(a.device).float()
		# v_des = torch.from_numpy(self.ref_traj_func.vel(time_step * 0.01)).to(a.device).float()
		# q_des = torch.from_numpy(self.ref_traj_func.quat(time_step * 0.01)).to(a.device).float()
		# w_des = torch.from_numpy(self.ref_traj_func.angvel(time_step * 0.01)).to(a.device).float()
		p_des = torch.from_numpy(self.ref_traj_func.pos(time_step * self.ave_dt)).to(a.device).float()
		v_des = torch.from_numpy(self.ref_traj_func.vel(time_step * self.ave_dt)).to(a.device).float()
		q_des = torch.from_numpy(self.ref_traj_func.quat(time_step * self.ave_dt)).to(a.device).float()
		w_des = torch.from_numpy(self.ref_traj_func.angvel(time_step * self.ave_dt)).to(a.device).float()


		if s is None:
			s_temp = self.s
			N_temp = self.N
		else:
			s_temp = s
			if len(s.shape) > 1:
				N_temp = s.shape[0]
			else:
				N_temp = 1

		if self.alpha_p > 0:
			ep = torch.linalg.norm(s_temp[:, 0:3] - p_des, dim=1)
		else:
			ep = 0.

		if self.alpha_v > 0:
			ev = torch.linalg.norm(s_temp[:, 3:6] - v_des, dim=1)
		else:
			ev = 0.

		if self.alpha_w > 0:	
			ew = torch.linalg.norm(s_temp[:, 10:] - w_des, dim=1)
		else:
			ew = 0.

		if self.alpha_R > 0:
			eR = qdistance_torch(s_temp[:, 6:10], q_des.view(1,4).repeat(N_temp,1))
			eR = torch.abs(eR)
		else:
			eR = 0.

		if self.alpha_a > 0:
			ea = torch.linalg.norm(a, dim=1)
		else:
			ea = 0.

		if self.alpha_yaw > 0:
			qe = qmultiply_torch(qconjugate_torch(q_des).view(1,4).repeat(N_temp,1), s_temp[:, 6:10])
			Re = qtoR_torch(qe)
			eyaw = torch.atan2(Re[:,1,0], Re[:,0,0]) ** 2
		else:
			eyaw = 0.

		cost = (
			   self.alpha_p * ep \
			 + self.alpha_v * ev \
			 + self.alpha_w * ew \
			 + self.alpha_a * ea \
			 + self.alpha_yaw * eyaw \
			 + self.alpha_R * eR) * self.ave_dt
		self.timer.toc('reward')

		#goal = torch.tensor(self.ref_traj_func.ref_vec(0.01*time_step), dtype=torch.float32)
		#N = s.shape[0]
		#cost = torch.linalg.norm(goal.tile((N,1)) - s) / N
		return -cost


	# dsdt = f(s,a)
	def f(self,s,a,noiseless=False):
		# input:
		# 	s, tensor, (N, 13)
		# 	a, tensor, (N, 4)
		# output:
		# 	dsdt, tensor, (N, 13)

		N_temp, _ = s.shape

		dsdt = torch.zeros(N_temp, 13)
		v = s[:, 3:6] # velocity (N, 3)
		q = s[:, 6:10] # quaternion (N, 4)
		omega = s[:, 10:] # angular velocity (N, 3)

		self.timer.tic()
		# get input 
		# eta = torch.mm(self.B0, a.T).T # output wrench (N, 4)
		eta = a
		#f_u = torch.zeros(self.N, 3) 
		f_u = torch.zeros(N_temp, 3) 
		f_u[:, 2] = eta[:, 0] # total thrust (N, 3)
		tau_u = eta[:, 1:] # torque (N, 3)
		self.timer.toc('wrench')

		self.timer.tic()
		# dynamics 
		# \dot{p} = v 
		dsdt[:, :3] = s[:, 3:6] 	# <- implies velocity and position in same frame
		# mv = mg + R f_u  	# <- implies f_u in body frame, p, v in world frame
		dsdt[:, 5] -= self.g
		dsdt[:, 3:6] += qrotate_torch(q, f_u)
		self.timer.toc('pos dynamics')

		self.timer.tic()
		# \dot{R} = R S(w)
		# see https://rowan.readthedocs.io/en/latest/package-calculus.html
		qnew = qintegrate_torch(q, omega, self.ave_dt, frame='body')
		qnew = qstandardize_torch(qnew)
		# transform qnew to a "delta q" that works with the usual euler integration
		dsdt[:, 6:10] = (qnew - q) / self.ave_dt
		self.timer.toc('att kinematics')

		self.timer.tic()
		# J\dot{w} = Jw x w + tau_u

		if not self.isangvel:
			Jomega = torch.mm(self.J, omega.T).T # J*omega (N, 3)
			dsdt[:, 10:] = torch.cross(Jomega, omega) + tau_u
			dsdt[:, 10:] = torch.mm(self.inv_J, dsdt[:, 10:].T).T
		# self.timer.toc('att dynamics')

		# # adding noise
		# if noiseless is False:
		# 	dsdt[:, 3:6] += torch.normal(mean=0, std=self.param.noise_process_std[0], size=(N_temp, 3))
		# 	dsdt[:, 10:] += torch.normal(mean=0, std=self.param.noise_process_std[1], size=(N_temp, 3))
		return dsdt


	def next_state(self,s,a,noiseless=False):
		if self.input_torque:
			a[:, 1:] = self.torque_to_angvel(s,a)
		s = s + self.ave_dt * self.f(s,a,noiseless=noiseless)
		if self.isangvel:
			s[:, 10:] = a[:, 1:]
		return s

	def torque_to_angvel(self,s,a):
		tau_u = a[:, 1:]
		omega = s[:, 10:]
		Jomega = torch.mm(self.J, omega.T).T 
		omega_dsdt = torch.cross(Jomega, omega) + tau_u
		omega_dsdt = torch.mm(self.inv_J, omega_dsdt.T).T
		return self.ave_dt * omega_dsdt

	def deduce_state(self, s):
		rpy = torch.from_numpy(np.degrees(rowan.to_euler(s[:, 6:10], 'xyz')))
		return rpy
	
	def reset(self, initial_state = None, N = None):
		if N is None:
			N_temp = self.N
		else:
			N_temp = N
		if initial_state is None:
			s_temp = np.empty((N,13))
			# position and velocity
			limits = self.limits
			s_temp[:,0:6] = np.random.uniform(-limits[0:6], limits[0:6], (N_temp,6))
			# rotation
			rpy = np.radians(np.random.uniform(-self.rpy_limit, self.rpy_limit, (N_temp,3)))
			q = rowan.from_euler(rpy[:,0], rpy[:,1], rpy[:,2], 'xyz')
			s_temp[:,6:10] = q
			# angular velocity
			s_temp[:,10:13] = np.random.uniform(-limits[10:13], limits[10:13], (N_temp,3))
		else:
			try:
				s_temp = np.outer(np.ones(N), initial_state.detach().cpu().numpy())
			except:
				s_temp = np.outer(np.ones(N), initial_state)

		self.s = torch.tensor(s_temp, dtype=torch.float32)
		self.time_step = 0
		return self.s.clone()

