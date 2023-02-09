import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_npz(filename):
    saved_data = np.load(filename)

    pose_positions = saved_data['pose_positions']
    pose_orientations = saved_data['pose_orientations']
    cf_positions = saved_data['cf_positions']
    ts = saved_data['ts']
    thrust_cmds = saved_data['thrust_cmds']
    ang_vel_cmds = saved_data['ang_vel_cmds']

    print(pose_orientations.shape, pose_orientations.shape, cf_positions.shape, ts.shape, thrust_cmds.shape)

    plt.figure(0)
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(ts, pose_positions[:, 0], label='/cf/pose position')
    plt.plot(ts, cf_positions[:, 0], label='cf.position()')
    plt.subplot(3, 1, 2, sharex=ax1)
    plt.plot(ts, pose_positions[:, 1], label='/cf/pose position')
    plt.plot(ts, cf_positions[:, 1], label='cf.position()')
    plt.subplot(3, 1, 3, sharex=ax1)
    plt.plot(ts, pose_positions[:, 2], label='/cf/pose position')
    plt.plot(ts, cf_positions[:, 2], label='cf.position()')
    plt.legend()
    plt.suptitle('positions (python)')

    plt.figure(1)
    ax2 = plt.subplot(3, 1, 1)
    plt.plot(ts, ang_vel_cmds[:, 0])
    plt.plot(ts, pose_orientations[:, 2], color='red')
    plt.subplot(3, 1, 2, sharex=ax2)
    plt.plot(ts, ang_vel_cmds[:, 1])
    plt.plot(ts, pose_orientations[:, 1], color='red')
    plt.subplot(3, 1, 3, sharex=ax2)
    plt.plot(ts, ang_vel_cmds[:, 2], label='Ang Vel Cmd (deg/s)')
    plt.plot(ts, pose_orientations[:, 0], color='red', label='Euler Angle (deg)')
    plt.suptitle('cf/pose orientation (python) & ang vel cmds')
    plt.legend()

    # plt.figure(2)
    # ax3 = plt.subplot(3, 1, 1)
    # plt.plot(ts, cf_positions[:, 0])
    # plt.subplot(3, 1, 2, sharex=ax3)
    # plt.plot(ts, cf_positions[:, 1])
    # plt.subplot(3, 1, 3, sharex=ax3)
    # plt.plot(ts, cf_positions[:, 2])
    # plt.suptitle('cf.position() (python)')

    plt.figure(3)
    plt.plot(ts, thrust_cmds)
    plt.title('Cmd z acc (python)')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")

    args = parser.parse_args()
    filename = args.filename

    plot_npz(filename)
    
