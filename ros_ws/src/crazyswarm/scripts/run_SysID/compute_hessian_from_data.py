import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch
import yaml
from enum import Enum
from Opt_Nonlinear_SysID_Quad.param_torch import Param, Timer
from Opt_Nonlinear_SysID_Quad.environments import QuadrotorAirDrag, AirDragKernel
from Opt_Nonlinear_SysID_Quad.policy_optimizers import compute_hessian_jacobian, PolicyGradient
from Opt_Nonlinear_SysID_Quad.controllers import QuadrotorPIDController
from Opt_Nonlinear_SysID_Quad import controller_torch
# from quadsim.learning.refs import 
from quadsim.learning.refs.random_zigzag import RandomZigzag
from quadsim.learning.refs.chained_poly_ref import ChainedPolyRef
from quadsim.learning.refs.circle_ref import CircleRef
from quadsim.learning.refs.pointed_star import NPointedStar
from quadsim.learning.refs.closed_polygon import ClosedPoly
from quadsim.learning.refs.hover_ref import hover_ref


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
    CHAINED_POLY_REF = 'chained_poly_ref'
    # MIXED_REF = 'mixed_ref'
    # GEN_TRAJ = 'gen_traj'
    STAR = 'star'
    POLYGON = 'polygon'

    def ref(self, y_max=0.0, seed=None, init_ref=None, diff_axis=False, z_max=0.0, env_diff_seed=False, **kwargs):

        return {
            TrajectoryRef.CIRCLE_REF: CircleRef(altitude=0, rad=0.5, period=2.0),
            TrajectoryRef.RANDOM_ZIGZAG: RandomZigzag(),
            TrajectoryRef.CHAINED_POLY_REF: ChainedPolyRef(altitude=0, use_y=(y_max > 0), seed=seed, min_dt=1.5, max_dt=4.0, degree=3, env_diff_seed=env_diff_seed, **kwargs),
            TrajectoryRef.STAR: NPointedStar(),
            TrajectoryRef.POLYGON: ClosedPoly(sides=5),
            TrajectoryRef.POLYGON: ClosedPoly(sides=5),
            TrajectoryRef.HOVER: hover_ref(),
        }[TrajectoryRef(self._value_)]
    
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
        
        rot_obj = R.from_euler('zyx', data['pose_orientations_euler'])
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

        inputs_ = torch.tensor(mass*np.hstack((data_dicts[i]['thrust_cmds'][:,None], data_dicts[i]['ang_vel_cmds'])), dtype=torch.float32)
        batch_inputs.append(inputs_)
    return batch_states, batch_inputs

def compute_hessian(states,inputs,param,config,H_task=50,pg_iters=300,pg_lr=0.0005):
    H = len(inputs[0]) - 1
    Aker = torch.tensor(0.01*np.random.randn(3,3), dtype=torch.float32)
    kernel = AirDragKernel() 
    instance = QuadrotorAirDrag(param, config, H, Aker, kernel, maximize=True)
    instance.update_parameter_estimates(states,inputs)

    Aker = instance.get_dynamics()
    instance = QuadrotorAirDrag(param, config, H_task, Aker.clone().detach(), kernel, maximize=True)
    controller = QuadrotorPIDController(torch.tensor([1.5,6,4], dtype=torch.float32),param,Adrag=Aker.clone().detach(),optimize_Adrag=True)
    policy_opt = PolicyGradient(controller,pg_iters,pg_lr,T=1000,N_hess=10)

    opt_controller = policy_opt.optimize(instance)
    _, hessian = compute_hessian_jacobian(instance,policy_opt,num_grads=10) 
    return hessian, opt_controller




if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--runtime", type=float, default=30)
    parser.add_argument("--hovertime",type=float,default=0)
    parser.add_argument("-tt", "--takeofftime",type=float,default=4.2)
    parser.add_argument("-bh", "--baseheight", type=float, default=0.6)
    args = parser.parse_args()
    with open('zigzag.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['controller'] = Controller_type.PID_TORCH._value_ 
    ref = TrajectoryRef(config['traj_'])
    param = Param(config, MPPI=True)
    param.ref_traj_func = ref.ref()

    fileNames = [ '/home/rwik/proj/Drones/crazyswarm/logs/icra2023_sysid/aug_23/sim/run_0.npz']
    data_dict = load_cf_data(fileNames, args)
    states, inputs = convertDict2Array(data_dict,mass=param.mass)
    # hessian, opt_controller = compute_hessian(states, inputs, param, config)
    import pdb;pdb.set_trace()