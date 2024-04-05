import numpy as np
from Controllers.ctrl_backbone import ControllerBackbone

from scipy.spatial.transform import Rotation as R
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR
from quadsim.visualizer import Vis
import torch
from torch.autograd.functional import jacobian
from stable_baselines3.common.env_util import make_vec_env
import time
class MPPI_thrust_omega():	
	'''
	using torch to sample and rollout
	a = [thrust (N), omega_des (rad/s)]
	'''
	def __init__(self, env, config):
		self.name = 'MPPI-thrust_omega'
		self.env = env
		self.thrust_hover = env.mass * env.g

		# MPPI parameters
		self.lam = config['mppi']['lam'] # temparature
		self.H = config['mppi']['H'] # horizon
		self.N = config['mppi']['N'] # number of samples
		self.gamma_mean = config['mppi']['gamma_mean']
		self.gamma_Sigma = config['mppi']['gamma_Sigma']
		self.omega_gain = config['mppi']['omega_gain']
		self.discount = config['mppi']['discount']
		sample_std = config['mppi']['sample_std']

		self.a_mean = torch.zeros(self.H, 4) # mean of actions: tensor, (H, 4)
		self.a_mean[:, 0] = self.thrust_hover
		self.a_Sigma = torch.zeros(self.H, 4, 4) # covariance of actions: tensor, (H, 4, 4)
		for h in range(self.H):
			self.a_Sigma[h, 0, 0] = sample_std[0]**2 * self.thrust_hover**2
			self.a_Sigma[h, 1, 1] = sample_std[1]**2
			self.a_Sigma[h, 2, 2] = sample_std[2]**2
			self.a_Sigma[h, 3, 3] = sample_std[3]**2

		self.a_min = torch.tensor(config['mppi']['a_min'])
		self.a_max = torch.tensor(config['mppi']['a_max'])
		self.a_max[0] = self.env.a_max[0]*4

		# timer for code profiling
		self.timer = Timer(topics=['shift', 'sample', 'rollout', 'update'])

	def sample(self):
		# A: tensor, (N, H, 4)
		A = torch.zeros(self.N, self.H, 4)
		for h in range(self.H):
			L = torch.linalg.cholesky(self.a_Sigma[h,:,:]) # decompose Sigma as Sigma = L @ L.T
			mean = self.a_mean[h,:].view(1,4).repeat(self.N,1)
			temp = torch.normal(mean=0., std=1., size=(self.N,4))
			temp = torch.mm(L, temp.T).T
			temp += mean
			#print(temp[:,0])
			temp = torch.clip(torch.as_tensor(temp, dtype=torch.float32), self.a_min, self.a_max)
			#print(temp[:,0])
			A[:, h, :] = temp
		#print(A[:,0,0])
		return A

	def omega_controller(self, state, a):
		# input:
		#   state: tensor, (N, 13)
		#   a (thrust and omega_des): tensor, (N, 4)
		# output:
		#   motorForce: tensor, (N, 4) 
		T_d = a[:, 0]
		omega_d = a[:, 1:]
		omega = state[:, 10:13]
		omega_e = omega_d - omega

		torque = self.omega_gain * omega_e # tensor, (N, 3)
		torque = torch.mm(self.env.J, torque.T).T
		torque -= torch.cross(torch.mm(self.env.J, omega.T).T, omega)
		
		wrench = torch.cat((T_d.view(self.N,1), torque), dim=1) # tensor, (N, 4)
		return wrench
		motorForce = torch.mm(self.env.B0_inv, wrench.T).T
		motorForce = torch.clip(motorForce, self.env.a_min, self.env.a_max)
		return motorForce

	def policy(self, state, time, time_step=None):
		# input:
		#   state: tensor, (13,)
		# output:
		#   motorForce: tensor, (4,)
		
		if np.abs((time / 0.2) % 1) < 0.001: 
			print(time, '/', self.env.param.sim_tf)
			# print(self.a_mean)
			# print(self.a_Sigma)
		# shift operator
		self.timer.tic()
		a_mean_old = self.a_mean.clone()
		a_Sigma_old = self.a_Sigma.clone()
		self.a_mean[:-1,:] = a_mean_old[1:,:]
		self.a_Sigma[:-1,:,:] = a_Sigma_old[1:,:,:]
		self.timer.toc('shift')

		# sample
		self.timer.tic()
		A = self.sample()
		Cost = torch.zeros(self.N)
		self.timer.toc('sample')

		# rollout
		self.timer.tic()
		self.env.s = state.view(1,13).repeat(self.N,1) # tensor, (N, 13)
		self.env.time_step = int(np.ceil(time / self.env.ave_dt))
		for h in range(self.H):
			u = self.omega_controller(self.env.s, A[:, h, :])
			_, reward, done = self.env.step(u)
			Cost -= reward * self.discount**h
			if done:
				# adding terminal cost
				if self.discount < 1:
					Cost -= reward * self.discount**(h+1) * (1-self.discount**(self.H-h-1)) / (1-self.discount)
				else:
					Cost -= reward * (self.H-h-1)
				break
		self.timer.toc('rollout')

		# compute weight
		self.timer.tic()
		Cost -= torch.min(Cost) 
		Cost = torch.exp(-1./self.lam*Cost)
		Weight = Cost / torch.sum(Cost) # tensor, (N,)
		Weight_mean = Weight.view(self.N,1).repeat(1,4) # tensor, (N,4)
		Weight_Sigma = Weight.view(self.N,1,1).repeat(1,4,4) # tensor, (N,4,4)

		# update mean and Sigma
		for h in range(self.H):
			self.a_mean[h,:] = (1 - self.gamma_mean) * self.a_mean[h,:]
			self.a_Sigma[h,:,:] = (1 - self.gamma_Sigma) * self.a_Sigma[h,:,:]
			
			new_mean = Weight_mean * A[:,h,:] # (N,4)
			new_mean = torch.sum(new_mean, dim=0)
			self.a_mean[h,:] += self.gamma_mean * new_mean

			m = A[:,h,:] - self.a_mean[h,:].view(1,4).repeat(self.N,1) # (N,4)
			new_Sigma = Weight_Sigma * torch.einsum("bi,bj->bij", m, m) # (N,4,4) einsum is for batch outer product
			new_Sigma = torch.sum(new_Sigma, dim=0)
			self.a_Sigma[h,:,:] += self.gamma_Sigma * new_Sigma
		self.timer.toc('update')

		# output the final command motorForce
		a_final = self.a_mean[0,:] # (4,)
		T_d = a_final[0]
		omega_d = a_final[1:]
		omega = state[10:13]
		omega_e = omega_d - omega
		torque = self.omega_gain * omega_e # tensor, (3,)
		torque = torch.mv(self.env.J, torque)
		torque -= torch.cross(torch.mv(self.env.J, omega), omega)
		wrench = torch.cat((T_d.view(1), torque)) # tensor, (4,)
		return wrench
		motorForce = torch.mv(self.env.B0_inv, wrench)
		motorForce = torch.clip(motorForce, self.env.a_min, self.env.a_max)

		return motorForce
