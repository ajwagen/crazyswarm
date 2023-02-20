import numpy as np
import argparse
import matplotlib.pyplot as plt
import CF_functions as cff

def sync_data(filename):
    python_filename = filename + '.npz'

    saved_data = np.load(python_filename)

    pose_positions = saved_data['pose_positions']
    pose_orientations = saved_data['pose_orientations']
    cf_positions = saved_data['cf_positions']
    ts = saved_data['ts']
    thrust_cmds = saved_data['thrust_cmds']
    ang_vel_cmds = saved_data['ang_vel_cmds']

    logData = cff.decode(filename)
    log_ts = logData['tick']
    thrust_cmds_drone = logData['ctrlRwik.cmd_z_acc']
    drone_x = logData['ctrlRwik.state_x']
    drone_y = logData['ctrlRwik.state_y']
    drone_z = logData['ctrlRwik.state_z']

    differences = np.abs(np.diff(drone_x))
    kernel = np.ones(15)
    z = np.convolve(differences, kernel)
    started = (z >= 15*1e-4) & (z <= 15*1e-3)
    start_index = np.where(started)[0][0]

    start_time_drone = log_ts[start_index]

    differences_python = np.abs(np.diff(pose_positions[:, 0]))
    z = np.convolve(differences_python, kernel)
    started = (z >= 15*1e-4) & (z <= 15*1e-3)
    start_index_python = np.where(started)[0][0]

    start_time_python = ts[start_index_python]
    end_time_python = ts[-1]

    end_time_drone = start_time_drone + (end_time_python - start_time_python)*1000

    # end_time_drone = start_time_drone + 1000*ts[-1]

    bounds = (log_ts >= start_time_drone) & (log_ts <= end_time_drone)
    bounds_python = (ts >= start_time_python) & (ts <= end_time_python)

    drone_ts = log_ts[bounds] / 1000 - start_time_drone / 1000
    python_ts = ts[bounds_python] - start_time_python

    plt.plot(python_ts, pose_positions[:, 0][bounds_python])
    plt.plot(drone_ts, drone_x[bounds])

    plt.figure()

    plt.plot(python_ts, thrust_cmds[bounds_python], label='python')
    plt.plot(drone_ts, thrust_cmds_drone[bounds], label='drone')

    plt.legend()

    # drone_ts = log_ts[bounds] / 1000 - start_time_drone / 1000
    # plt.plot(drone_ts, drone_x[bounds])

    # print(thrust_cmds_drone)
    # # print(np.where(thrust_cmds_drone != 0))
    # first_nonzero_idx = np.where(thrust_cmds_drone != 0)[0][0]
    # start_time_drone = log_ts[first_nonzero_idx]
    # end_time_drone = start_time_drone + 1000*ts[-1]

    # bounds = (log_ts >= start_time_drone) & (log_ts <= end_time_drone)

    # len_py = len(ts)
    
    # drone_x = logData['ctrlRwik.state_x'][bounds]
    # drone_y = logData['ctrlRwik.state_y'][bounds]
    # drone_z = logData['ctrlRwik.state_z'][bounds]

    # drone_ts = log_ts[bounds] / 1000 - start_time_drone / 1000 + 10

    # plt.figure(0)
    # ax1 = plt.subplot(3, 1, 1)
    # plt.plot(ts, pose_positions[:, 0], label='/cf/pose position')
    # plt.plot(ts, cf_positions[:, 0], label='cf.position()')
    # plt.plot(drone_ts, drone_x)
    # plt.subplot(3, 1, 2, sharex=ax1)
    # plt.plot(ts, pose_positions[:, 1], label='/cf/pose position')
    # plt.plot(ts, cf_positions[:, 1], label='cf.position()')
    # plt.plot(drone_ts, drone_y)
    # plt.subplot(3, 1, 3, sharex=ax1)
    # plt.plot(ts, pose_positions[:, 2], label='/cf/pose position')
    # plt.plot(ts, cf_positions[:, 2], label='cf.position()')
    # plt.plot(drone_ts, drone_z, label='Drone estimated position')
    # plt.legend()
    # plt.suptitle('positions (python)')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")

    args = parser.parse_args()
    filename = args.filename

    sync_data(filename)
