import numpy as np
import matplotlib.pyplot as plt
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
        
        rot_obj = R.from_euler('zyx', data['pose_orientations_euler'])
        data['pose_orientations_quat'] = rot_obj.as_quat()

        data['ref_orientation'] = saved_data['ref_orientation'][t_mask]
        data['thrust_cmds'] = saved_data['thrust_cmds'][t_mask]
        data['ang_vel_cmds'] = saved_data['ang_vel_cmds'][t_mask]
        
        data_dicts.append(data)
    
    return data_dicts

def convertDict2Array(data_dicts):
    batch_states = []

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

        states_ = np.hstack((pos, vel, quat, omega))
        batch_states.append(states_)

if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--runtime", type=float, default=30)
    parser.add_argument("--hovertime",type=float,default=0)
    parser.add_argument("-tt", "--takeofftime",type=float,default=4.2)
    parser.add_argument("-bh", "--baseheight", type=float, default=0.6)
    args = parser.parse_args()

    parent = "aug_02/real"
    prefix = "hover"
    numFiles = 5
    fileNames = []
    for i in range(numFiles):
        fileNames.append(parent + "/" + prefix + "_" + str(i) + ".npz")

    data_dict = load_cf_data(fileNames, args)

    convertDict2Array(data_dict)

    
