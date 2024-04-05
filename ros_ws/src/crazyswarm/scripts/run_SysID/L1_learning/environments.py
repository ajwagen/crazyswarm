import numpy as np
import torch
import copy
from quadrotor_torch import Quadrotor_torch
import time


class Quadrotor:
	def __init__(self,param,config,H,Aker,kernel,cost=None,noise_cov=0.1,maximize=False,goal_H=None,k_delay=1):
		'''
		Inputs:
			A - dimx x dimx system matrix
			B - dimx x dimu control matrix
			H - horizon
			noise_cov - noise covariance. If scalar, noise covariance is sqrt{noise_cov} * I, otherwise
							noise covariance is noise_cov
			cost - cost object
			exact_cost - if True, cost is computed in closed-form when possible; otherwise cost is computed via sampling
		'''
		self.quadrotor = Quadrotor_torch(param,config)
		self.Aker = Aker
		self.kernel = kernel
		self.H = H
		if H_goal is None:
			self.H_goal = H
		else:
			self.H_goal = H_goal
		self.dimx = 13
		self.dimu = 4
		self.dimphi = kernel.get_dim()
		self.init_state = None
		self.hessian = None
		self.parallel_N = -1
		self.cost = cost
		self.maximize = maximize
		self.ref_traj = param.ref_traj_func
		self.multitask = False

		self.data_cov = torch.zeros(self.dimphi,self.dimphi)
		self.data_process = torch.zeros(self.dimphi,self.dimx)
		self.u_play = None
		self.k_delay = k_delay

		if torch.is_tensor(noise_cov) is False:
			noise_cov = torch.tensor(noise_cov, dtype=torch.float32)
		if len(noise_cov.shape) == 0:
			self.noise_std = torch.sqrt(noise_cov) * torch.eye(self.dimx)
		else:
			U,S,V = torch.svd(noise_cov)
			self.noise_std = U @ torch.diag(torch.sqrt(S)) @ V.T

	def dynamics(self,x,u,noiseless=False):
		'''
		Dynamics of system.

		Inputs: 
			x - matrix of size dimx x T denoting system state (if T > 1, this corresponds to
				running T trajectories in parallel)
			u - matrix of size dimu x T denoting system input

		Outputs:
			x_plus - matrix of size dimx x T denote next state
		'''
		if self.u_play is None:
			self.u_play = u
		else:
			self.u_play = self.u_play + self.k_delay * (u - self.u_play)
		if len(x.shape) > 1:
			z = self.kernel.phi(x,u)
			return self.quadrotor.next_state(x.T,self.u_play.T,noiseless=noiseless).T + self.Aker @ self.kernel.phi(x,u)
		else:
			return torch.flatten(self.quadrotor.next_state(x[None,:],self.u_play[None,:],noiseless=noiseless)) + self.Aker @ self.kernel.phi(x,u)

	def base_dynamics(self,x,u,noiseless=False):
		if len(x.shape) > 1:
			return self.quadrotor.next_state(x.T,u.T,noiseless=noiseless).T
		else:
			return torch.flatten(self.quadrotor.next_state(x[None,:],u[None,:],noiseless=noiseless))

	def loss(self,x,u,h):
		if len(x.shape) > 1:
			return self.quadrotor.reward(u.T,s=x.T,time_step=h)
		else:
			return self.quadrotor.reward(u[None,:],x[None,:],time_step=h)

	def get_init_state(self,N=1):
		self.u_play = None
		if self.init_state is None:
			x_init = self.quadrotor.reset(N=N)
		else:
			x_init = self.quadrotor.reset(initial_state=self.init_state,N=N)
		if N == 1:
			return torch.flatten(x_init)
		else:
			return x_init.T

	def controller_cost(self,controller,T=200,noiseless=False,regularization=None):
		'''
		Estimates cost of controller.

		Inputs: 
			controller - controller object
			T - number of trajectories to average costs across
			noiseless - if True, rollout trajectories on system with no noise
			exact_cost - if True, compute cost exactly (when this is possible)

		Outputs:
			estimated cost of controller
		'''
		input_norm = None
		if self.parallel_N > 0:
			loss = torch.zeros(self.parallel_N)
		else:
			loss = torch.zeros(1)
		if T > 1000:
			T0 = 0
			while T0 < T:
				x = self.get_init_state(N=1000)
				for h in range(self.H_goal):
					u = controller.get_input(x,h)
					if self.cost is None:
						loss = loss + torch.sum(self.loss(x,u,h))
					else:
						loss = loss + self.cost.get_cost(x,u,h)
					if regularization is not None:
						if input_norm is None:
							input_norm = torch.linalg.norm(u)**2
						else:
							input_norm = input_norm + torch.linalg.norm(u)**2
					#u = u + torch.sqrt(torch.tensor(0.00001 / 4)) * torch.randn(4,1000)
					x = self.dynamics(x,u,noiseless=noiseless)
				T0 = T0 + 1000
		else:
			#for t in range(T):
			x = self.get_init_state(N=T)
			for h in range(self.H_goal):
				u = controller.get_input(x,h)
				if self.cost is None:
					loss = loss + torch.sum(self.loss(x,u,h))
				else:
					loss = loss + self.cost.get_cost(x,u,h)
				if regularization is not None:
					if input_norm is None:
						input_norm = torch.linalg.norm(u)**2
					else:
						input_norm = input_norm + torch.linalg.norm(u)**2
				#u = u + torch.sqrt(torch.tensor(0.00001 / 4)) * torch.randn(4,T)
				x = self.dynamics(x,u,noiseless=noiseless)
		if self.maximize:
			if regularization is not None:
				return -loss / T + regularization * input_norm / T
			else:
				return -loss / T
		else:
			if regularization is not None:
				return loss / T + regularization * input_norm / T
			else:
				return loss / T

	def get_noise_cov(self,noise_std,T=5000):
		noise_cov = torch.zeros(self.data_cov.shape)
		if T > 1000:
			T0 = 0
			while T0 < T:
				x = self.get_init_state(N=1000)
				for h in range(self.H):
					u = noise_std @ torch.randn(self.dimu,1000)
					x = self.dynamics(x,u)
					z = self.get_z(x,u)
					noise_cov = noise_cov + (z @ z.T)/T                    
				T0 = T0 + 1000
		else:
			x = self.get_init_state(N=T)
			for h in range(self.H):
				u = noise_std @ torch.randn(self.dimu,T)
				x = self.dynamics(x,u)
				z = self.get_z(x,u)
				noise_cov = noise_cov + (z @ z.T)/T
		return noise_cov

	def get_cov(self,x,u,parallel=False):
		if parallel:
			z = self.kernel.phi(x,u)
			cov = torch.einsum('ik, jk -> ijk',z,z)
			return cov
		else:
			z = self.get_z(x,u)
			return torch.outer(z,z)

	def get_z(self,x,u):
		return self.kernel.phi(x,u)

	def update_parameter_estimates(self,states,inputs,action_delay=False):
		T = len(inputs)
		for t in range(T):
			action = inputs[t][0]
			for h in range(self.H):
				if action_delay:
					action = 0.4 * inputs[t][h] + 0.6 * action
				else:
					action = inputs[t][h]
				z = self.get_z(states[t][h],action)
				self.data_cov += torch.outer(z,z)
				self.data_process += torch.outer(z,states[t][h+1] - self.base_dynamics(states[t][h],action,noiseless=True))
		thetahat = torch.linalg.inv(self.data_cov) @ self.data_process
		self.Aker = thetahat.T

	def compute_opt_val(self,policy_opt,T=50000):
		ce_opt_controller = policy_opt.optimize(self)
		return self.controller_cost(ce_opt_controller,T=T).detach().cpu().numpy()

	def compute_est_error(self,est_instance,metrics):
		Ahat = est_instance.get_dynamics()
		thetast = torch.flatten(self.Aker)
		thetahat = torch.flatten(Ahat)
		if 'thetast_est_error' in metrics:
			metrics['thetast_est_error'].append(torch.linalg.norm(Ahat - self.Aker).detach().cpu().numpy())
		else:
			metrics['thetast_est_error'] = [torch.linalg.norm(Ahat - self.Aker).detach().cpu().numpy()]
		if self.multitask and self.hessian is not None:
			for cost_id in self.hessian:
				if 'hess_est_error_' + cost_id in metrics:
					metrics['hess_est_error_' + cost_id].append(((thetast - thetahat) @ (self.hessian[cost_id] / torch.norm(self.hessian[cost_id])) @ (thetast - thetahat)).detach().cpu().numpy())
				else:
					metrics['hess_est_error_' + cost_id] = [((thetast - thetahat) @ (self.hessian[cost_id] / torch.norm(self.hessian[cost_id])) @ (thetast - thetahat)).detach().cpu().numpy()]
		else:
			if 'hess_est_error' in metrics and self.hessian is not None:
				metrics['hess_est_error'].append(((thetast - thetahat) @ (self.hessian / torch.norm(self.hessian)) @ (thetast - thetahat)).detach().cpu().numpy())
			elif self.hessian is not None:
				metrics['hess_est_error'] = [((thetast - thetahat) @ (self.hessian / torch.norm(self.hessian)) @ (thetast - thetahat)).detach().cpu().numpy()]
		return metrics

	def reset_parameters(self):
		self.Aker = torch.zeros(self.dimx,self.dimphi)

	def get_dim(self):
		return self.dimx, self.dimu, self.dimphi, self.H

	def set_dynamics(self,Aker):
		self.Aker = Aker

	def get_dynamics(self):
		return self.Aker

	def set_theta(self,theta):
		theta_rs = torch.reshape(theta,(self.dimx,self.dimphi))
		self.Aker = theta_rs

	def get_theta(self):
		return torch.flatten(self.Aker)

	def set_hessian(self,hess,multitask=False):
		self.hessian = hess
		self.multitask = multitask

	def get_hessian(self):
		if self.hessian is None:
			return None
		else:
			if self.multitask:
				return_hessians = {}
				for cost_id in self.hessian:
					return_hessians[cost_id] = self.hessian[cost_id].clone().detach()
				return return_hessians
			else:
				return self.hessian.clone().detach()

	def set_init_state(self,new_init_state):
		self.init_state = new_init_state

	def get_data_cov(self):
		return self.data_cov

	def get_kron_dim(self):
		return 3

	def set_cost(self,cost):
		self.cost = cost
		self.ref_traj = cost.ref_traj_func

	def get_ref_traj(self):
		return self.ref_traj

	def get_data_suff_stat(self):
		return self.data_cov.detach().cpu().numpy(), self.data_process.detach().cpu().numpy()

	def set_data_suff_state(self, data_cov, data_process):
		self.data_cov = data_cov
		self.data_process = data_process





class QuadrotorAirDrag(Quadrotor):
	def __init__(self,param,config,H,Aker,kernel,cost=None,noise_cov=0.1,maximize=False,H_goal=None,k_delay=1):
		'''
		Inputs:
			A - dimx x dimx system matrix
			B - dimx x dimu control matrix
			H - horizon
			noise_cov - noise covariance. If scalar, noise covariance is sqrt{noise_cov} * I, otherwise
							noise covariance is noise_cov
			cost - cost object
			exact_cost - if True, cost is computed in closed-form when possible; otherwise cost is computed via sampling
		'''
		self.quadrotor = Quadrotor_torch(param,config)
		self.Aker = Aker
		self.kernel = kernel
		self.H = H
		if H_goal is None:
			self.H_goal = H
		else:
			self.H_goal = H_goal
		self.dimx = 13
		self.dimu = 4
		self.dimphi = kernel
		self.init_state = None
		self.hessian = None
		self.parallel_N = -1
		self.cost = cost
		self.maximize = maximize
		self.param = param
		self.config = config
		self.noise_cov = noise_cov
		self.ref_traj = param.ref_traj_func
		self.multitask = False
		self.u_play = None
		self.k_delay = k_delay

		self.P = torch.zeros(self.dimx,3)
		self.P[3:6,:] = torch.eye(3)

		self.data_cov = torch.zeros(self.dimphi,self.dimphi)
		self.data_process = torch.zeros(self.dimphi,3)

		if torch.is_tensor(noise_cov) is False:
			noise_cov = torch.tensor(noise_cov, dtype=torch.float32)
		if len(noise_cov.shape) == 0:
			self.noise_std = torch.sqrt(noise_cov) * torch.eye(self.dimx)
		else:
			U,S,V = torch.svd(noise_cov)
			self.noise_std = U @ torch.diag(torch.sqrt(S)) @ V.T

	def generate_duplicate(self):
		return QuadrotorAirDrag(self.param,self.config,self.H,self.Aker.detach().clone(),self.kernel,cost=self.cost,noise_cov=self.noise_cov,maximize=self.maximize,H_goal=self.H_goal)

	def dynamics(self,x,u,noiseless=False):
		'''
		Dynamics of system.

		Inputs: 
			x - matrix of size dimx x T denoting system state (if T > 1, this corresponds to
				running T trajectories in parallel)
			u - matrix of size dimu x T denoting system input

		Outputs:
			x_plus - matrix of size dimx x T denote next state
		'''
		if self.u_play is None:
			self.u_play = u
		else:
			self.u_play = self.u_play + self.k_delay * (u - self.u_play)
		if len(x.shape) > 1:
			return self.quadrotor.next_state(x.T,self.u_play.T,noiseless=noiseless).T + self.P @ self.Aker @ self.kernel.phi(x,u)
		else:
			return torch.flatten(self.quadrotor.next_state(x[None,:],self.u_play[None,:],noiseless=noiseless)) + self.P @ self.Aker @ self.kernel.phi(x,u)

	def update_parameter_estimates(self,states,inputs,action_delay=True,regularizer=0):
		T = len(inputs)
		for t in range(T):
			action = inputs[t][0]
			for h in range(len(inputs[t])-1):
				if action_delay:
					action = self.k_delay * inputs[t][h] + (1 - self.k_delay) * action
				else:
					action = inputs[t][h]
				z = self.get_z(states[t][h],action)
				self.data_cov += torch.outer(z,z)
				self.data_process = self.data_process + torch.outer(z,states[t][h+1][3:6] - self.base_dynamics(states[t][h],action,noiseless=True)[3:6])
		thetahat = torch.linalg.inv(self.data_cov + regularizer*torch.eye(self.data_cov.shape[0])) @ self.data_process
		self.Aker = thetahat.T

	def reset_parameters(self):
		self.Aker = torch.zeros(3,3)

	def set_theta(self,theta):
		theta_rs = torch.reshape(theta,(3,3))
		self.Aker = theta_rs





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

class AirDragKernel2:
	def __init__(self):
		return

	def get_dim(self):
		return 7

	def phi(self,x,u):
		if len(x.shape) > 1:
			N = x.shape[1]
			v_norm = torch.outer(torch.ones(3),torch.linalg.norm(x[3:6,:],2,dim=0))
			vs = torch.multiply(v_norm,x[3:6,:])
			#z = torch.concat((x[3:6,:],vs,torch.sin(x[3:6,:]),torch.cos(x[3:6,:]),torch.ones((1,N), dtype=torch.float32)),dim=0)
			z = torch.concat((x[3:6,:],vs,torch.ones((1,N), dtype=torch.float32)),dim=0)
			return z
		else:
			v_norm = torch.linalg.norm(x[3:6],2)
			return torch.concat((x[3:6],v_norm*x[3:6],torch.tensor([1], dtype=torch.float32)))
			#return torch.concat((x[3:6],v_norm*x[3:6],torch.sin(x[3:6]),torch.cos(x[3:6]),torch.tensor([1], dtype=torch.float32)))



class LinearKernel:
	def __init__(self,n,dimx,dimu):
		self.n = n
		self.dimx = dimx
		self.dimu = dimu
		np.random.seed(1)
		A = 0.01*np.random.randn(n,dimx)
		np.random.seed(int(time.time()))
		self.A = torch.tensor(A, dtype=torch.float32)

	def get_dim(self):
		return self.n

	def phi(self,x,u):
		return self.A @ x


class FourierKernel:
	def __init__(self,n,dimx,dimu):
		self.n = n
		self.dimx = dimx
		self.dimu = dimu
		#self.weights = torch.randn(n,dimx+dimu)
		#self.offsets = 2*torch.pi*(torch.rand(n) - 0.5)
		self.weights = torch.zeros(n,dimx+dimu)
		self.offsets = torch.zeros(n)
		for i in range(n):
			self.offsets[i] = 2*torch.pi*(i/n - 0.5)
			for j in range(dimx+dimu):
				self.weights[i,j] = torch.tensor(np.cos(0.1*np.pi*i/n+np.pi*j/(dimx+dimu)))

	def get_dim(self):
		return self.n

	def phi(self,x,u):
		z = torch.concat((x,u),dim=0)
		if len(x.shape) > 1:
			N = x.shape[1]
			phi_out = torch.zeros(self.n,N)
			for i in range(self.n):
				phi_out[i,:] = 0.1*torch.cos(self.weights[i,:] @ z + self.offsets[i])
			return phi_out
		else:
			phi_out = torch.zeros(self.n)
			for i in range(self.n):
				phi_out[i] = 0.1*torch.cos(self.weights[i,:] @ z + self.offsets[i])
			return phi_out



