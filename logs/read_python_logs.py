import numpy as np
import argparse
import matplotlib.pyplot as plt

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def plot_npz(filename):

    data_dict={}
    # minimum_len={}
    for i in filename:
        data = {}
        saved_data = np.load(i) 
    # saved_data = np.load(filename)

        minimum_len = np.inf
        for key in saved_data.keys():
            k = len(saved_data[key])
            if k<minimum_len:
                minimum_len = k
        st= first_nonzero(saved_data['ref_positions'],0)[0]

        data['pose_positions'] = saved_data['pose_positions'][st:k]
        data['pose_orientations'] = saved_data['pose_orientations'][st:k]
        data['cf_positions'] = saved_data['cf_positions'][st:k]
        data['ts'] = saved_data['ts'][st:k]
        data['ref_positions'] = saved_data['ref_positions'][st:k]
        data['ref_orientation'] = saved_data['ref_orientation'][st:k]
        data['thrust_cmds'] = saved_data['thrust_cmds'][st:k]
        data['ang_vel_cmds'] = saved_data['ang_vel_cmds'][st:k]
        data['mocap_orientation'] = saved_data['motrack_orientation'][st:k]

        data_dict[i]= data
    # print(pose_orientations.shape)
    # print(np.diff(pose_orientations).shape)
    # ang_vels = np.diff(pose_orientations,axis=-1)/ts
    # ang_vels = np.append(ang_vels,ang_vels[-1],axis=0)
    # offs = mocap_orientation[0] - pose_orientations[0]

    

    # print(pose_orientations.shape, pose_orientations.shape, cf_positions.shape, ts.shape, thrust_cmds.shape)
    
    plt.figure(0)
    ax1 = plt.subplot(3, 1, 1)
    for key in data_dict.keys():
        # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 0], label='cf.position()')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 0])
    plt.subplot(3, 1, 2, sharex=ax1)
    for key in data_dict.keys():
        # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 1], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 1], label='cf.position()')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 1])
    plt.subplot(3, 1, 3, sharex=ax1)
    for key in data_dict.keys():
        # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 2], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 2], label=key+'_cf.position()')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 2],label=key+'_ref position')
    plt.legend()
    plt.suptitle('PPO curriculum')

    plt.figure(1)
    ax1 = plt.subplot(3, 1, 1)
    for key in data_dict.keys():
        # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_orientations'][:, 0], label='cf.position()')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_orientation'][:, 0])
    plt.subplot(3, 1, 2, sharex=ax1)
    for key in data_dict.keys():
        # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 1], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_orientations'][:, 1], label='cf.position()')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_orientation'][:, 1])
    plt.subplot(3, 1, 3, sharex=ax1)
    for key in data_dict.keys():
        # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 2], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_orientations'][:, 2], label=key+'_cf.position()')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_orientation'][:, 2],label=key+'_ref position')
    plt.legend()
    plt.suptitle('PPO curriculum attitude')
    # plt.figure(1)
    # ax2 = plt.subplot(3, 1, 1)
    # plt.plot(ts, ang_vel_cmds[:, 0])
    # plt.plot(ts, mocap_orientation[:,2])
    # plt.plot(ts, pose_orientations[:, 2], color='red')
    # plt.subplot(3, 1, 2, sharex=ax2)
    # plt.plot(ts, ang_vel_cmds[:, 1])
    # plt.plot(ts, mocap_orientation[:,1])
    # plt.plot(ts, pose_orientations[:, 1], color='red')
    # plt.subplot(3, 1, 3, sharex=ax2)
    # plt.plot(ts, ang_vel_cmds[:, 2], label='Ang Vel Cmd (deg/s)')
    # plt.plot(ts, mocap_orientation[:,0], label='mocap data')
    # plt.plot(ts, pose_orientations[:, 0], color='red', label='Euler Angle (deg)')
    # plt.suptitle('cf/pose orientation (python) & ang vel cmds')
    # plt.legend()

    # # plt.figure(2)
    # # ax3 = plt.subplot(3, 1, 1)
    # # plt.plot(ts, cf_positions[:, 0])
    # # plt.subplot(3, 1, 2, sharex=ax3)
    # # plt.plot(ts, cf_positions[:, 1])
    # # plt.subplot(3, 1, 3, sharex=ax3)
    # # plt.plot(ts, cf_positions[:, 2])
    # # plt.suptitle('cf.position() (python)')

    # plt.figure(3)
    # plt.plot(ts, thrust_cmds)
    # plt.title('Cmd z acc (python)')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")

    args = parser.parse_args()
    filenames = args.filename
    # print(filename)
    # exit()

    plot_npz(filenames)
    
