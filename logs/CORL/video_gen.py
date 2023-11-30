import numpy as np
import argparse
import matplotlib.pyplot as plt
from plt_utils import load_cf_data

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def plot_npz(filename):

    data_dict = load_cf_data(filenames, args)

    x_max, x_min = np.max(data_dict[filename[0]]['ref_positions'][:, 0]), np.min(data_dict[filename[0]]['ref_positions'][:, 0])
    y_max, y_min = np.max(data_dict[filename[0]]['ref_positions'][:, 1]), np.min(data_dict[filename[0]]['ref_positions'][:, 1])

    for key in data_dict.keys():
        x_max_t, x_min_t = np.max(data_dict[filename[0]]['ref_positions'][:, 0]), np.min(data_dict[filename[0]]['ref_positions'][:, 0])
        y_max_t, y_min_t = np.max(data_dict[filename[0]]['ref_positions'][:, 1]), np.min(data_dict[filename[0]]['ref_positions'][:, 1])
        
        if x_max_t> x_max:
            x_max = x_max_t
        
        if y_max_t> y_max:
            y_max = y_max_t
        
        if x_min_t< x_min:
            x_min = x_min_t
        
        if y_min_t< y_min:
            y_min = y_min_t

    leg_dict = {
        'mppi' : 'MPC',
        'pid'  : 'nonlinear baseline',
        'ppo'  : 'DATT',
    }
    for i in range(len(data_dict[filename[0]]['ref_positions'])):

        plt.figure(5)
        plt.clf()
        plt.plot(data_dict[filename[0]]['ref_positions'][:i, 0], data_dict[filename[0]]['ref_positions'][:i, 1], '--')
        # for key in data_dict.keys():
        #     if 'mppi' in key:
        #         lab = leg_dict['mppi']
        #     if 'pid' in key:
        #         lab = leg_dict['pid']
        #     if 'ppo' in key:
        #         lab = leg_dict['ppo']
            
        #     if 'L1' in key:
        #         lab = 'L1-' + lab

        #     p = plt.plot(data_dict[key]['pose_positions'][:i,0], data_dict[key]['pose_positions'][:i, 1], label=lab)
        #     plt.plot(data_dict[key]['pose_positions'][i,0], data_dict[key]['pose_positions'][i, 1], '*', color = p[0].get_color())

        # plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.xlim((-0.2 + x_min, 0.2 + x_max))
        plt.ylim((-0.2 + y_min, 0.2 + y_max))
        plt.grid()
        plt.savefig('imgs/' + str(i).zfill(4)+".png")
    # print(pose_orientations.shape, pose_orientations.shape, cf_positions.shape, ts.shape, thrust_cmds.shape)
    
    # plt.figure(0)
    # ax1 = plt.subplot(3, 1, 1)
    # for key in data_dict.keys():
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0], label='/cf/pose position')
    #     # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 0], label='cf.position()')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 0])
    # plt.subplot(3, 1, 2, sharex=ax1)
    # for key in data_dict.keys():
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 1], label='/cf/pose position')
    #     # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 1], label='cf.position()')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 1])
    # plt.subplot(3, 1, 3, sharex=ax1)
    # for key in data_dict.keys():
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 2], label='/cf/pose position')
    #     # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 2], label=key+'_cf.position()')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 2],label=key+'_ref position')
    # plt.legend()
    # plt.suptitle('PPO curriculum')
    
    # pos_rmse = np.sqrt(np.mean(data_dict[key]['cf_positions'] - data_dict[key]['ref_positions'], axis=0) ** 2)
    # print("position RMSE : ", pos_rmse)
    # print("total position RMSE", np.linalg.norm(pos_rmse))
    # error = data_dict[key]['ref_positions'] - data_dict[key]['cf_positions']

    # plt.figure(1)
    # ax1 = plt.subplot(3, 1, 1)
    # for key in data_dict.keys():
    #     # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0], label='/cf/pose position')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['pose_orientations'][:, 0], label='cf.position()')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['ref_orientation'][:, 0])
    # plt.subplot(3, 1, 2, sharex=ax1)
    # for key in data_dict.keys():
    #     # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 1], label='/cf/pose position')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['pose_orientations'][:, 1], label='cf.position()')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['ref_orientation'][:, 1])
    # plt.subplot(3, 1, 3, sharex=ax1)
    # for key in data_dict.keys():
    #     # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 2], label='/cf/pose position')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['pose_orientations'][:, 2], label=key+'_cf.position()')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['ref_orientation'][:, 2],label=key+'_ref position')
    # plt.legend()
    # plt.suptitle('PPO curriculum attitude')


    # plt.figure(2)
    # ax1 = plt.subplot(3, 1, 1)
    # for key in data_dict.keys():
    #     zero_error = np.zeros_like(data_dict[key]['ts'])
    #     # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0], label='/cf/pose position')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 0] - data_dict[key]['pose_positions'][:, 0], label='cf.position()')
    #     plt.plot(data_dict[key]['ts'], zero_error)
    # plt.subplot(3, 1, 2, sharex=ax1)
    # for key in data_dict.keys():
    #     zero_error = np.zeros_like(data_dict[key]['ts'])
    #     # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positionss'][:, 1], label='/cf/pose position')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 1] - data_dict[key]['pose_positions'][:, 1], label='cf.position()')
    #     plt.plot(data_dict[key]['ts'], zero_error)
    # plt.subplot(3, 1, 3, sharex=ax1)
    # for key in data_dict.keys():
    #     zero_error = np.zeros_like(data_dict[key]['ts'])
    #     # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positionss'][:, 2], label='/cf/pose position')
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 2] - data_dict[key]['pose_positions'][:, 2], label=key+'_cf.position()')
    #     plt.plot(data_dict[key]['ts'], zero_error, label=key+'zero error')
    # plt.legend()
    # plt.suptitle('position error')
    # # plt.figure(1)
    # # ax2 = plt.subplot(3, 1, 1)
    # # plt.plot(ts, ang_vel_cmds[:, 0])
    # # plt.plot(ts, mocap_orientation[:,2])
    # # plt.plot(ts, pose_orientations[:, 2], color='red')
    # # plt.subplot(3, 1, 2, sharex=ax2)
    # # plt.plot(ts, ang_vel_cmds[:, 1])
    # # plt.plot(ts, mocap_orientation[:,1])
    # # plt.plot(ts, pose_orientations[:, 1], color='red')
    # # plt.subplot(3, 1, 3, sharex=ax2)
    # # plt.plot(ts, ang_vel_cmds[:, 2], label='Ang Vel Cmd (deg/s)')
    # # plt.plot(ts, mocap_orientation[:,0], label='mocap data')
    # # plt.plot(ts, pose_orientations[:, 0], color='red', label='Euler Angle (deg)')
    # # plt.suptitle('cf/pose orientation (python) & ang vel cmds')
    # # plt.legend()

    # # plt.figure(2)
    # # ax3 = plt.subplot(3, 1, 1)
    # # plt.plot(ts, cf_positions[:, 0])
    # # plt.subplot(3, 1, 2, sharex=ax3)
    # # plt.plot(ts, cf_positions[:, 1])
    # # plt.subplot(3, 1, 3, sharex=ax3)
    # # plt.plot(ts, cf_positions[:, 2])
    # # plt.suptitle('cf.position() (python)')

    # plt.figure(3)
    # for key in data_dict.keys():
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['thrust_cmds'])
    # plt.title('Cmd z acc (python)')

    # # try:
    # plt.figure(4)
    # for key in data_dict.keys():
    #     # print(data_dict[key]['adaptation_terms'])
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['adaptation_terms'])
    # plt.title('adaptation term')
    # except:
    #     pass

    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    parser.add_argument("--runtime", type=float, default=10)
    parser.add_argument("--hovertime",type=float,default=0)
    parser.add_argument("-bh", "--baseheight", type=float, default=1.0)
    parser.add_argument("-tt", "--takeofftime",type=float,default=5.0)

    args = parser.parse_args()
    filenames = args.filename
    # print(filename)
    # exit()

    plot_npz(filenames)
    
