import numpy as np 
import matplotlib.pyplot as plt
import torch
import copy
import os
import pickle
import sys
import shutil
import yaml
import time
from enum import Enum
# from refs import *
# from param_torch import Param, Timer
# from exploration import RandomExploration, CEEDExploration, DynamicOED, TrajTrack
# from system_id import system_id
# from environments import QuadrotorAirDrag, FourierKernel, LinearKernel, AirDragKernel
# from policy_optimizers import PolicySearch, PolicyGridSearch, compute_hessian_jacobian, PolicyGradient, OptimizeMPPI
# from controllers import QuadrotorPIDController, MPPI, NeuralNetController
# from costs import QuadraticCost, TrackingCost
# from quadrotor_torch import Quadrotor
from scipy.spatial.transform import Rotation as R



def load_cf_data(filenames, args):
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

        t_mask = (saved_data['ts'] > args.takeofftime + args.hovertime) * (saved_data['ts'] < args.runtime + args.takeofftime + args.hovertime)

        data['ts'] = saved_data['ts'][t_mask]
        data['ref_positions'] = saved_data['ref_positions'][t_mask] #- saved_data['ref_positions'][st]

        data['pose_positions'] = saved_data['pose_positions'][t_mask] #- saved_data['pose_positions'][st]
        data['pose_positions'][:, :2] -= data["ref_positions"][0, :2]
        data['pose_positions'][:, 2] -= args.baseheight

        data['ref_positions'][:, :2] -= data['ref_positions'][0, :2]
        data['ref_positions'][:, -1] -= args.baseheight

        data['pose_orientations_euler'] = saved_data['pose_orientations'][t_mask]
        
        rot_obj = R.from_euler('zyx', np.deg2rad(data['pose_orientations_euler']))
        data['pose_orientations_quat'] = rot_obj.as_quat()

        data['ref_orientation'] = saved_data['ref_orientation'][t_mask]
        data['thrust_cmds'] = saved_data['thrust_cmds'][t_mask]
        data['ang_vel_cmds'] = saved_data['ang_vel_cmds'][t_mask]
        
        data_dicts.append(data)
    
    return data_dicts

def convertDict2Array(data_dicts,mass=1):
    batch_states = []
    batch_inputs = []

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

        states_ = torch.tensor(np.hstack((pos, vel, quat, omega)), dtype=torch.float32)
        batch_states.append(states_)

        inputs_ = torch.tensor(mass*np.hstack((data_dicts[i]['thrust_cmds'][:,None],  np.deg2rad(data_dicts[i]['ang_vel_cmds']))), dtype=torch.float32)
        batch_inputs.append(inputs_)
    return batch_states, batch_inputs

class Controller_type(Enum):
	PID = 'pid'
	MPPI = 'mppi'
	PID_TORCH = 'pid_torch'


	def controller(self, param, config, env):
		print(Controller_type(self._value_))
		return{
			Controller_type.PID : controller_torch.PID(param),
			Controller_type.MPPI : controller_torch.MPPI_thrust_omega(env, config),
			Controller_type.PID_TORCH: controller_torch.PID_torch(param)
		}[Controller_type(self._value_)]

class TrajectoryRef(Enum):
	# LINE_REF = 'line_ref'
	# SQUARE_REF = 'square_ref'
	CIRCLE_REF = 'circle_ref'
	RANDOM_ZIGZAG = 'random_zigzag'
	# RANDOM_ZIGZAG_YAW = 'random_zigzag_yaw'
	# SETPOINT = 'setpoint'
	# POLY_REF = 'poly_ref'
	HOVER = 'hover'
	HOVER2 = 'hover2'
	CHAINED_POLY_REF = 'chained_poly_ref'
	# CHAINED_POLY_REF2 = 'chained_poly_ref2'
	# MIXED_REF = 'mixed_ref'
	# GEN_TRAJ = 'gen_traj'
	STAR = 'star'
	POLYGON = 'polygon'

	def ref(self, y_max=0.0, seed=None, init_ref=None, diff_axis=False, z_max=0.0, env_diff_seed=False, **kwargs):

		return {
			TrajectoryRef.CIRCLE_REF: CircleRef(altitude=0, rad=0.5, period=2.0),
			TrajectoryRef.RANDOM_ZIGZAG: RandomZigzag(),
			TrajectoryRef.CHAINED_POLY_REF: ChainedPolyRef(altitude=0, use_y=(y_max > 0), seed=seed, min_dt=1.5, max_dt=4.0, degree=3, env_diff_seed=env_diff_seed, **kwargs),
			# TrajectoryRef.CHAINED_POLY_REF2: ChainedPolyRef2(altitude=0, use_y=(y_max > 0), seed=seed, min_dt=1.5, max_dt=4.0, degree=3, env_diff_seed=env_diff_seed, **kwargs),
			TrajectoryRef.STAR: NPointedStar(),
			TrajectoryRef.POLYGON: ClosedPoly(sides=5),
			TrajectoryRef.POLYGON: ClosedPoly(sides=5),
			TrajectoryRef.HOVER: hover_ref(),
			TrajectoryRef.HOVER2: hover2_ref()
		}[TrajectoryRef(self._value_)]
	
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", help='configuration yaml file')
	parser.add_argument("--vis", action='store_true', default=False, help='visualization using meshcat or not')
	parser.add_argument('--ref', type=TrajectoryRef, default=None)
	parser.add_argument('--cntrl', type=Controller_type, default=Controller_type.PID_TORCH)
	parser.add_argument('-rt', '--run_torch', type=bool,default=False)
	opt = parser.parse_args()

