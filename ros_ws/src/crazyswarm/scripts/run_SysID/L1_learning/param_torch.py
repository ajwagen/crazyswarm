import numpy as np 
import time
# import refs

d = 1E6

class Timer():
	'''
	for code profiling
	'''
	def __init__(self, topics):
		self.t = 0.
		self.stats = {}
		for topic in topics:
			self.stats[topic] = []

	def tic(self):
		self.t = time.time()

	def toc(self, topic):
		self.stats[topic].append(time.time() - self.t)

class Param:
	def __init__(self, config, MPPI=False):
		self.env_name = 'Quadrotor-Crazyflie'

		# Quadrotor parameters
		self.sim_t0 = 0
		self.sim_tf = config['sim_tf']
		self.sim_dt = config['sim_dt']
		self.dt = config['sim_dt']
		self.sim_times = np.arange(self.sim_t0, self.sim_tf, self.sim_dt)
		if MPPI:
			self.sim_dt = config['sim_dt_MPPI']
			self.sim_times = np.arange(self.sim_t0, self.sim_tf, self.sim_dt)

		# control limits [N]
		self.a_min = np.array([0., 0., 0., 0.])
		self.a_max = np.array([12., 12., 12., 12.]) / 1000 * 9.81 # g->N

		# Crazyflie 2.0 quadrotor
		self.mass = 0.034 # kg
		self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])
		self.d = 0.047

		# Sideforce model parameters for wind perturbations
		if config['Vwind'] == 0:
			self.wind = False
			self.Vwind = None
		else:
			self.wind = True
			self.Vwind = np.array(config['Vwind']) # velocity of wind in world frame
		self.Ct = 2.87e-3
		self.Cs = 2.31e-5
		self.k1 = 1.425
		self.k2 = 3.126
		self.rho = 1.225 # air density (in SI units)

		# Note: we assume here that our control is forces
		arm_length = 0.046 # m
		arm = 0.707106781 * arm_length
		t2t = 0.006 # thrust-to-torque ratio
		self.t2t = t2t
		self.B0 = np.array([
			[1, 1, 1, 1],
			[-arm, -arm, arm, arm],
			[-arm, arm, arm, -arm],
			[-t2t, t2t, -t2t, t2t]
			])
		self.a_min = np.array([0, 0, 0, 0])
		self.a_max = np.array([12, 12, 12, 12]) / 1000 * 9.81 # g->N
		self.g = 9.81 # not signed

		# Exploration parameters: state boundary and initial state sampling range
		self.s_min = np.array( \
			[-8, -8, -8, \
			  -5, -5, -5, \
			  -1.001, -1.001, -1.001, -1.001,
			  -20, -20, -20])
		self.rpy_limit = np.array([5, 5, 5])
		self.limits = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,0,0,0,0])

		# measurement noise
		self.noise_measurement_std = np.zeros(13)
		# self.noise_measurement_std[:3] = 0.005
		# self.noise_measurement_std[3:6] = 0.005
		# self.noise_measurement_std[6:10] = 0.01
		# self.noise_measurement_std[10:] = 0.01

		# process noise
		#self.noise_process_std = [0.3, 2.]
		self.noise_process_std = [0.75, 2]

		# Reference trajectory
		self.ref_trajectory = np.zeros((13, len(self.sim_times))) 
		self.ref_trajectory[6, :] = 1.

		if config['traj'] == 'fig-8':
			for step, time in enumerate(self.sim_times):
				self.ref_trajectory[0, step] = 0.5*np.sin(time)
				self.ref_trajectory[2, step] = 0.5*np.cos(2*time + np.pi/2)
				self.ref_trajectory[3, step] = 0.5*np.cos(time)
				self.ref_trajectory[5, step] = -1*np.sin(2*time + np.pi/2)
		elif config['traj'] == 'zig-zag':
			p = 2. # period
			t = self.sim_times + p/4
			self.ref_trajectory[0, :] = 2 * np.abs(t/p - np.floor(t/p+0.5)) - 0.5
		elif config['traj'] == 'zig-zag-yaw':
			for step, time in enumerate(self.sim_times):
				if time <= 1.:
					self.ref_trajectory[0, step] = time
					self.ref_trajectory[3, step] = 1.
				else:
					self.ref_trajectory[0, step] = 2 - time
					self.ref_trajectory[3, step] = -1.
					self.ref_trajectory[6, step] = 0.
					self.ref_trajectory[9, step] = 1.
		else:
			print('WARNING: No such a trajecotry type!')
		self.ref_traj_func = None
		

		self.alpha_p = config['alpha_p']
		self.alpha_w = config['alpha_w']
		self.alpha_a = config['alpha_a']
		self.alpha_R = config['alpha_R']
		self.alpha_v = config['alpha_v']
		self.alpha_yaw = config['alpha_yaw']

		self.isangvel = True
