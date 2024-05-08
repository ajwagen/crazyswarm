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
                self.L1_adapt_term = np.zeros(3)
                # This one is used for logging. Not quite sure if using the same one
                # for both logging and storing the term value is fine
                self.L1_adaptation_terms = np.zeros(3)
                self.final_adapt_term = np.zeros(3)

                # naive params
                self.lamb = 0.1

                # L1 params
                self.runL1 = True # L1 v/s naive toggle
                self.filter_coeff = 5
                self.A = -0.2
                self.count = 0
                
                # L1 learned variables
                self.network = residual_net()
                self.network.load_state_dict(torch.load('/home/drones/drones_project/crazyswarm/ros_ws/src/crazyswarm/scripts/run_SysID/L1_learning/model_weights/L1_test_small_ft2'))
                self.network = self.network.to('cuda:0')
               # self.A_hat = torch.tensor([[-1.5033e+00,  1.5552e+00,  4.3266e-01, -6.2234e-01, -4.0937e-03],
                # [-6.0332e-02, -8.8851e-02, -7.6040e-01,  2.8345e-01,  5.2903e-01],
                # [ 1.1084e+01, -2.4040e+01,  1.7124e+01, -1.0613e+01, -3.1320e+00]])
                # Weights from base flight with 2 fans both at 3 speed, using multiple trials of data
               # self.A_hat = torch.tensor([[-0.0745,  0.0047,  0.1710, -0.0743,  0.0441],
       # [-0.0461,  0.0121,  0.0720, -0.1329,  0.0431],
       # [ 0.2924, -0.0592, -0.1780,  1.0491, -0.3419]])
                # Weights generated from a single file base flight with 2 fans both at speed 3
               # self.A_hat = torch.tensor([[-0.1058,  0.0466,  0.1227, -0.0775,  0.0389],
       # [-0.2928,  0.1253,  0.3351, -0.2091,  0.1090],
       # [ 1.7134, -0.7290, -1.6789,  1.3886, -0.6495]])
                # Weights generated using 5 files from mppi_straight_line_3fans_speed3_no_adapt_trial*.npz
                #self.A_hat = torch.tensor([[-1.2060, -0.5683,  0.7540,  0.2808,  0.5327],
       # [ 0.0985, -0.2683, -0.3582,  0.2839, -0.0753],
       # [ 0,  0, 0,  0, 0]])
                # Weights generated usign 1 file from mppi_straight_line_3fans_speed3_no_adapt_trial3.npz
               # self.A_hat = torch.tensor([[-1.5672, -0.6562,  0.9373, -0.1178,  0.4431],
        #[ 0.0527, -0.4192, -0.4385,  0.1952, -0.1537],
        #[ 0,  0, 0,  0, 0]])
                # weights generated using 5 files with mppi 3 fans all in a straight line 
                self.A_hat = torch.tensor([[-0.4550, -0.5356,  0.0455,  0.4540,  0.1207],
        [ 0.0402,  0.0164, -0.4497,  0.1441, -0.0973],
        [ 1.8827,  0.7237,  0.2655,  1.0013,  0.0217]])
                #self.A_hat = torch.tensor()
                
                self.A_hat = self.A_hat.to('cuda:0')

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

        def L1_learned(self, inputs):
            self.L1_adapt_term = self.A_hat @ self.network(inputs.to('cuda:0'))
            self.L1_adaptation_terms = self.L1_adapt_term.cpu().detach()
            #return ret_val
'''
        def train_linear(self, file_ids):
            states, inputs, L1_est, _, filenames = self.load_data(files=file_ids)
            L1_p = L1_est[i][:idx_split-1,:].T
            states_m = states[i][:idx_split-1,:]
            inputs_m = inputs[i][:idx_split-1,:]
            phi = self.network.forward(states_m[:,3:9])
            U,S,V = torch.linalg.svd(phi.T @ phi)
            Ahat_i = torch.linalg.inv(phi.T @ phi + 0.1 * torch.eye(5)) @ phi.T @ L1_p.T
            Ahat_i = Ahat_i.T
            return Ahat_i

        def load_cf_data2(self, filenames, takeofftime, hovertime, runtime, baseheight):
            data_dicts=[]

            for i in filenames:
                data = {}
                saved_data = dict(np.load(i, allow_pickle=True))

                minimum_len = np.inf
                for key in saved_data.keys():
                    k = len(saved_data[key])
                    if k<minimum_len:
                        minimum_len = k
                
                for key in saved_data.keys():
                    saved_data[key] = saved_data[key][:minimum_len]

                t_mask = (saved_data['ts'] > takeofftime + hovertime) * (saved_data['ts'] < runtime + takeofftime + hovertime)

                data['ts'] = saved_data['ts'][t_mask]
                data['ref_positions'] = saved_data['ref_positions'][t_mask] #- saved_data['ref_positions'][st]

                data['pose_positions'] = saved_data['pose_positions'][t_mask] #- saved_data['pose_positions'][st]
                data['pose_positions'][:, :2] -= data["ref_positions"][0, :2]
                data['pose_positions'][:, 2] -= baseheight

                data['ref_positions'][:, :2] -= data['ref_positions'][0, :2]
                data['ref_positions'][:, -1] -= baseheight

                data['pose_orientations_euler'] = saved_data['pose_orientations'][t_mask]
                
                rot_obj = R.from_euler('zyx', np.deg2rad(data['pose_orientations_euler']))
                data['pose_orientations_quat'] = rot_obj.as_quat()
                ft = []
                for t in range(len(t_mask)):
                    if t_mask[t]:
                        rot = R.from_euler('zyx', np.deg2rad(saved_data['pose_orientations'][t,:]))
                        ft.append(rot.apply(np.array([0, 0, saved_data['thrust_cmds'][t]])))
                data['ft'] = np.array(ft)

                data['ref_orientation'] = saved_data['ref_orientation'][t_mask]
                data['thrust_cmds'] = saved_data['thrust_cmds'][t_mask]
                data['ang_vel_cmds'] = saved_data['ang_vel_cmds'][t_mask]
                data['L1_est'] = saved_data['adaptation_terms'][t_mask]
                
                data_dicts.append(data)
            
            return data_dicts

        def convertDict2Array2(self,data_dicts,mass=1):
            batch_states = []
            batch_inputs = []
            batch_L1 = []
            batch_ft = []

            for i in range(len(data_dicts)):
                pos = data_dicts[i]["pose_positions"]
                dt = np.diff(data_dicts[i]["ts"])

                vel = np.diff(data_dicts[i]["pose_positions"], axis=0) / dt[:, None]
                vel = np.vstack((vel[0][None, :], vel))

                quat = data_dicts[i]["pose_orientations_quat"]

                if "pose_omega" in data_dicts[i]:
                    omega = data_dicts[i]["pose_omega"]
                else:
                    quat_diff = np.diff(quat, axis=0)
                    omega = 2 * quat_diff[:, 1:] / dt[:, None]
                    omega = np.vstack((omega[0][None, :], omega))

                quat = data_dicts[i]['ft']
                states_ = torch.tensor(np.hstack((pos, vel, quat, omega)), dtype=torch.float32)
                batch_states.append(states_)

                inputs_ = torch.tensor(mass*np.hstack((data_dicts[i]['thrust_cmds'][:,None],  np.deg2rad(data_dicts[i]['ang_vel_cmds']))), dtype=torch.float32)
                batch_inputs.append(inputs_)
                
                L1_ = torch.tensor(data_dicts[i]['L1_est'][:,1:], dtype=torch.float32)
                batch_L1.append(L1_)
                
                ft_ = torch.tensor(data_dicts[i]['ft'], dtype=torch.float32)
                batch_ft.append(ft_)
            return batch_states, batch_inputs, batch_L1, batch_ft

        def load_data(self, files=None):
            takeofftime = 4.0
            baseheight = 1.0
            runtime = 14.0 #14.0
            hovertime = 0.0
            rootdir = '/home/drones/drones_project/crazyswarm/logs/icra2023_sysid/feature_learning_data/real/'
            fileNames = []
            for file in os.listdir(rootdir):
                if ".npz" in file:
                    if files is None:
                        fileNames.append(rootdir + file)
                    elif file in files:
                        fileNames.append(rootdir + file)
            data_dict = self.load_cf_data2(fileNames, takeofftime, hovertime, runtime, baseheight)
            states, inputs, L1_est, ft = self.convertDict2Array2(data_dict, mass=1)   
            return states, inputs, L1_est, ft, fileNames
'''
