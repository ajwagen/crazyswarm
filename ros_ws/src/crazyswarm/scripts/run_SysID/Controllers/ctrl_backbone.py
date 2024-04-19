import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from quadsim.learning.train_policy import DroneTask, RLAlgo, SAVED_POLICY_DIR, import_config, CONFIG_DIR, TEST_POLICY_DIR
from quadsim.learning.utils.adaptation_network import AdaptationNetwork
from stable_baselines3.common.env_util import make_vec_env

# importing MPPI
from quadrotor_torch import Quadrotor_torch
from param_torch import Param, Timer
import controller_torch
from pathlib import Path
import os
from Controllers.ctrl_config import select_policy_config_

from Opt_Nonlinear_SysID_Quad.controllers import QuadrotorPIDController
from Opt_Nonlinear_SysID_Quad.environments import LearnedKernel, FixedLearnedKernel, AirDragKernel2, NeuralFly


class ControllerBackbone():
	def __init__(self, isSim, 
				 policy_config='trajectory', 
				 adaptive=False, 
				 pseudo_adapt=True,
				 adapt_smooth = False,
				 explore_type = 'random',
				 init_run = False):
		self.isSim = isSim
		self.mass = 0.04
		self.g = 9.8
		self.e_dims = 0
		self.adaptive = adaptive

		self.pseudo_adapt = pseudo_adapt
		self.adaptation_terms = np.zeros(4)
		self.adaptation_mean = np.zeros((1, 4))
		self.adapt_smooth = adapt_smooth
		self.explore_type = explore_type
		self.exploration_dir = '/home/rwik/proj/Drones/icra_23/Opt_Nonlinear_SysID_Quad/Opt_Nonlinear_SysID_Quad/hessian_bank/traj_ceed_1/1D/seed0/'

		self.init_run = init_run
		# self.I = np.array([[3.144988, 4.753588, 4.640540],
		#                    [4.753588, 3.151127, 4.541223],
		#                    [4.640540, 4.541223, 7.058874]])*1e-5
		self.I = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])
		
		self.prev_t = None
		self.dt_prev = None
		
		self.dt = None
		self.offset_pos = np.zeros(3)

		self.time_horizon = 10
		self.policy_config = policy_config
		
		# Classical adaptation
		self.v_prev = np.zeros(3)
		self.v_hat = np.zeros(3)
		self.wind_adapt_term = np.zeros(3)
		self.wind_adapt_term_t = np.zeros(3)


		# naive params
		self.lamb = 0.1

		# L1 params
		self.runL1 = True # L1 v/s naive toggle
		self.filter_coeff = 5
		self.A = -0.2
		self.count = 0

	def updateDt(self,t):

		if self.prev_t is None:
			self.dt = 0.02
		else:
			self.dt = t - self.prev_t
		
		if self.dt < 0:
			self.dt = 0

	def select_policy_configs(self,):
		policy_dict = select_policy_config_(self.policy_config)
		self.task = policy_dict["task"]
		self.policy_name = policy_dict["policy_name"]
		self.config_filename = policy_dict["config_filename"]
		self.adaptive_policy_name = policy_dict["adaptive_policy_name"]
		self.body_frame = policy_dict["body_frame"]
		self.relative = policy_dict["relative"]
		self.log_scale = policy_dict["log_scale"]
		self.e_dims = policy_dict["e_dims"]
		self.u_struct = policy_dict["u_struct"]
		self.adaptation_warmup = policy_dict['adaptation_warmup']

	def set_policy(self,):

		self.select_policy_configs()

		self.algo = RLAlgo.PPO
		# self.config = None

		self.algo_class = self.algo.algo_class()

		config = import_config(self.config_filename)
		# self.config = config
		try :
			self.env = self.task.env()(
				config=config,
				log_scale = self.log_scale,
				body_frame = self.body_frame,
				relative = self.relative,
				u_struct = self.u_struct,
			)
		except:
			self.env = self.task.env()(
				config=config,
				log_scale = self.log_scale,
				body_frame = self.body_frame,
			)
		
		self.policy = self.algo_class.load(TEST_POLICY_DIR / f'{self.policy_name}', self.env)
		if self.adaptive == True and self.pseudo_adapt == False:
			self.adaptive_policy = AdaptationNetwork(14, self.e_dims, complex=True)
			self.adaptive_policy.load_state_dict(torch.load(TEST_POLICY_DIR / f'{self.adaptive_policy_name}', map_location='cuda:0'))
		self.prev_pos = 0.
	
	def set_MPPI_controller(self,):
		config_dir = config_dir = os.path.dirname(os.path.abspath(__file__)) + "/mppi_config"

		# config_dir = "/mnt/hdd/drones/crazyswarm/ros_ws/src/crazyswarm/scripts/DroneLearning/Controllers/mppi_config"
		# config_dir = "/home/guan/ya/rwik/drones/crazyswarm/ros_ws/src/crazyswarm/scripts/DroneLearning/Controllers/mppi_config"
		with open(config_dir + "/hover.yaml") as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
		
		self.param_MPPI = Param(config, MPPI=True)
		env_MPPI = Quadrotor_torch(self.param_MPPI, config)
		controller = controller_torch.MPPI_thrust_omega(env_MPPI, config)
		
		return controller

	def set_BC_policy(self, ):
		from imitation.algorithms import bc
		BC = 'MPPI_BC'
		# bc_policy_name = BC + '/' 'ppo_mppi_bc'
		# bc_policy_name = BC + '/' 'ppo-mppi_zigzag_bf_rel_bc'
		# bc_policy_name = BC + '/' 'ppo-mppi_zigzag_bc'
		bc_policy_name = BC + '/' 'ppo-mppi_zigzag_xy_bc'
		self.body_frame = False
		self.relative = False
		self.bc_policy = bc.reconstruct_policy(TEST_POLICY_DIR / f'{bc_policy_name}')
	
	def set_PID_torch(self, ):
		from Opt_Nonlinear_SysID_Quad.param_torch import Param as param_explore

		with open('/home/drones/drones_project/Opt_Nonlinear_SysID_Quad/zigzag.yaml') as f:
			config = yaml.load(f, Loader=yaml.FullLoader)
		
		param = param_explore(config, MPPI=True)
		param.ref_traj_func = None

		param.sim_dt = 0.02
		param.dt = 0.02

		kernel = FixedLearnedKernel(weights_name='weights_new_checkpoint_xy_relu_dim5', feature_type='torch_nn')
	 
		#kernel = AirDragKernel2()
		#kernel = NeuralFly()
		# Aker = np.load(self.exploration_dir+'aker.npy')
		aker = torch.tensor(0.01*np.random.randn(3,3), dtype=torch.float32)
		controller = QuadrotorPIDController(torch.tensor([6, 4, 1.5], dtype=torch.float32), param, Adrag=torch.zeros(3,12), optimize_Adrag=True, control_angvel=True, kernel=kernel, Att_p=torch.tensor(150/16, dtype=torch.float32), Att_p_yaw=torch.tensor(220/16, dtype=torch.float32))
		# controller_params = np.load('/home/rwik/proj/Drones/icra_23/Opt_Nonlinear_SysID_Quad/Opt_Nonlinear_SysID_Quad/data/explore_circle_aggressive_opt_controller.npz', allow_pickle=True)
		# controller_params = np.load('/home/rwik/proj/Drones/icra_23/Opt_Nonlinear_SysID_Quad/Opt_Nonlinear_SysID_Quad/data/random_plate_0_policy_params.npz', allow_pickle=True)
		# gains = torch.tensor([8.9898, 8.2800, 3.7587], dtype=torch.float)
		gains = torch.tensor([ 6.0,  4.0, 1.0])
		#aker = torch.tensor([[ 1.3392, -0.1821,  0.2880,  0.1450,  0.0265,  0.2777, -0.0643],
		#[ 0.6081,  0.6013, -0.3648,  0.1887, -0.7169,  0.3071,  0.0058],
		#[ 1.1289, -1.3013,  1.1387,  0.4944, -1.0385, -0.9167, -1.7765]])
		aker = torch.tensor([[ 1.9150, -0.2709, -1.9437, -0.1780, 0.6429, 0.2477, -0.2496, -0.3725,
	 -0.1729, 0.3114, 0.1166, 0.0616],
	[-3.2938, -3.9685, -1.6172, -2.5190, -1.1896, 0.3737, -0.1477, 0.5953,
	 -0.7244, 0.6338, -0.2503, -0.1346],
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0]], dtype=torch.float)
		controller.update_params([gains, 0.0*aker])

		if False:
			from Opt_Nonlinear_SysID_Quad.environments import QuadrotorAirDrag
			from Opt_Nonlinear_SysID_Quad.crazyflie_helper import load_cf_data, convertDict2Array
			from Opt_Nonlinear_SysID_Quad.param_torch import Param as Param_sysid
			files = ['ch_poly_seed3_plate','circle_r0.5p2.0_plate_0','ch_poly_seed0_plate','ch_poly_seed4_plate', \
			'ch_poly_seed2_plate','circle_r0.3p3.0_plate_0','ch_poly_seed1_plate','circle_r0.5p4.0_plate_0']
			class args_:
				takeofftime = 4.2
				bh = 0.6 # change to 0.6 if using with plate and fan
				runtime = 10.0
				hovertime = 4.0
			states = []
			inputs = []
			for file_id in files:
				fileNames = ['./data/' + file_id + '.npz']
				# args = parser.parse_args()
				# args.takeofftime = 4.2
				# args.bh = 0.6 # change to 0.6 if using with plate and fan
				# args.runtime = 10.0
				# args.hovertime = 4.0
				args = args_()
				data_dict = load_cf_data(fileNames, args)
				states2, inputs2 = convertDict2Array(data_dict, mass=1)
				states.append(states2[0])
				inputs.append(inputs2[0])


			for i in range(len(states)):
				filter_cutoff = int(0.07 * len(states[i]))
				states_fft = torch.fft.fft(states[i], dim=0)
				states_fft[filter_cutoff:-filter_cutoff+1,:] = 0
				states[i] = torch.real(torch.fft.ifft(states_fft, dim=0))

				inputs_fft = torch.fft.fft(inputs[i], dim=0)
				inputs_fft[filter_cutoff:-filter_cutoff+1,:] = 0
				inputs[i] = torch.real(torch.fft.ifft(inputs_fft, dim=0))

			# param_instance = Param_s(config, MPPI=True)
			H = 500
			H_goal = 500
			Aker = torch.zeros(3,12)
			true_instance = QuadrotorAirDrag(param, config, H, Aker, kernel, maximize=True, H_goal=H_goal, k_delay=1.0)
			true_instance.update_parameter_estimates(states,inputs,action_delay=True,regularizer=1)
			aest = true_instance.get_dynamics().detach()
			controller.update_params([(1/0.02)*aest])

			exit()
	 
		return controller

	def _response(self, fl1, response_inputs):
		raise NotImplementedError
	
	def response(self, fl=1, **kwargs):
		return self._response(fl, **kwargs)
	
	# Classical Adaptation techniques
	def naive_adaptation(self, a_t, f_t):
		unity_mass = 1
		g_vec = np.array([0, 0, -1]) * self.g

		adapt_term = unity_mass * a_t - unity_mass * g_vec - f_t

		self.wind_adapt_term = (1 - self.lamb) * self.wind_adapt_term + self.lamb * adapt_term

	def L1_adaptation(self, dt, v_t, f_t):
		unit_mass = 1
		g_vec = np.array([0, 0, -1]) * self.g
		# alpha = np.exp(-dt * self.filter_coeff)
		alpha = 0.99
		# print(alpha)
		phi = 1 / self.A * (np.exp(self.A * dt) - 1)

		a_t_hat = g_vec + f_t / unit_mass - self.wind_adapt_term_t + self.A * (self.v_hat - v_t)
		
		self.v_hat += a_t_hat * dt
		v_tilde = self.v_hat - v_t
		
		self.wind_adapt_term_t = 1 / phi * np.exp(self.A * dt) * v_tilde
		self.wind_adapt_term = -(1 - alpha) * self.wind_adapt_term_t + alpha * self.wind_adapt_term
'''
         def L1_learned(self, inputs):
                class residual_net(torch.nn.Module):
                        def __init__(self):
                                dim_hidden = 32
                                super(residual_net, self).__init__()
                                self.fc1 = nn.Linear(6, dim_hidden)
                                self.fc2 = nn.Linear(dim_hidden, dim_hidden)
                                self.fc4 = nn.Linear(dim_hidden, 5)
                        def forward(self, x):
                                x = F.relu(self.fc1(x))
                                x = F.relu(self.fc2(x))
                                x = self.fc4(x)
                                return x
                network = residual_net()
                # FIXME: REVISE PATH
                network.load_state_dict(torch.load('./model_weights/weights_new_checkpoint_L1_test_small_ft'))
                return network(x)
'''
