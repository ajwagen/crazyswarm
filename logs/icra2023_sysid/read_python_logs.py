import numpy as np
import argparse
import matplotlib.pyplot as plt
from plt_utils import load_cf_data
from Opt_Nonlinear_SysID_Quad.math_utils import *
from scipy.spatial.transform.rotation import Rotation as R
import rowan

class quad():
    def __init__(self):
        self.a_prev = np.array([0., 0., 0., 0.])
        self.k = 1.0

    def f(self, s,a, dt):
        # input:
        # 	s, nd array, (n,)
        # 	a, nd array, (m,1)
        # output
        # 	dsdt, nd array, (n,1)

        dsdt = np.zeros(13)
        q = s[6:10]
        omega = s[10:]

        # get input 
        # a = np.reshape(a,(self.m,))
        # eta = np.dot(self.B0,a)
        eta = a
        f_u = np.array([0, 0, eta[0]])

        # dynamics 
        # dot{p} = v 
        dsdt[0:3] = s[3:6] 	# <- implies velocity and position in same frame
        # mv = mg + R f_u  	# <- implies f_u in body frame, p, v in world frame
        dsdt[3:6] = np.array([0, 0, -9.81]) + rowan.rotate(q, f_u)
        
        # dot{R} = R S(w)
        # to integrate the dynamics, see
        # https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/
        qnew = qintegrate(q, omega, dt, frame='body')
        qnew = qstandardize(qnew)
        # transform qnew to a "delta q" that works with the usual euler integration
        dsdt[6:10] = (qnew - q) / dt

        return dsdt

    def next_state(self, s, a, dt):
        a_delay = self.a_prev + self.k * (a - self.a_prev)
        dsdt = self.f(s, a_delay, dt)
        s = s + dsdt * dt

        s[10:] = a[1:]
        return s


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def plot_npz(filename):

    data_dict = load_cf_data(filenames, args)
    plt.figure(5)
    plt.plot(data_dict[filename[0]]['ref_positions'][:, 0], data_dict[filename[0]]['ref_positions'][:, 1])
    for key in data_dict.keys():
        plt.plot(data_dict[key]['pose_positions'][:,0], data_dict[key]['pose_positions'][:, 1], label=key)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal', 'box')

    # print(pose_orientations.shape, pose_orientations.shape, cf_positions.shape, ts.shape, thrust_cmds.shape)
    
    quadrotor = quad()
    for key in data_dict.keys():
        p = data_dict[key]['pose_positions']
        v = data_dict[key]['pose_vel']
        r = R.from_euler('zyx', np.deg2rad(data_dict[key]['pose_orientations']))
        q = np.roll(r.as_quat(), 1)
        w = data_dict[key]['ang_vel_cmds']
        fullstates = np.hstack((p, v, q, w))
        data_dict[key]['nominal_dyn'] = fullstates.copy()

        a_t = data_dict[key]['thrust_cmds']
        w = np.deg2rad(w)
        a = np.hstack((a_t[:, None], w))
        dt = np.diff(data_dict[key]['ts'])
        dt = np.r_[0.02, dt]
        for i in range(len(fullstates)):
            # import pdb;pdb.set_trace()
            data_dict[key]['nominal_dyn'][i] = quadrotor.next_state(fullstates[i].copy(), a[i].copy(), dt[i].copy())
    

    plt.figure(0)
    ax1 = plt.subplot(3, 1, 1)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0], label='/cf/pose position')
        # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 0], label='cf.position()')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 0])
    plt.subplot(3, 1, 2, sharex=ax1)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 1], label='/cf/pose position')
        # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 1], label='cf.position()')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 1])
    plt.subplot(3, 1, 3, sharex=ax1)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 2], label='/cf/pose position')
        # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 2], label=key+'_cf.position()')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 2],label=key+'_ref position')
    plt.legend()
    plt.suptitle('PPO curriculum')
    
    # pos_rmse = np.sqrt(np.mean(data_dict[key]['cf_positions'] - data_dict[key]['ref_positions'], axis=0) ** 2)
    # print("position RMSE : ", pos_rmse)
    # print("total position RMSE", np.linalg.norm(pos_rmse))
    # error = data_dict[key]['ref_positions'] - data_dict[key]['cf_positions']

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


    plt.figure(2)
    ax1 = plt.subplot(3, 1, 1)
    for key in data_dict.keys():
        zero_error = np.zeros_like(data_dict[key]['ts'])
        # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 0] - data_dict[key]['pose_positions'][:, 0], label='cf.position()')
        plt.plot(data_dict[key]['ts'], zero_error)
    plt.subplot(3, 1, 2, sharex=ax1)
    for key in data_dict.keys():
        zero_error = np.zeros_like(data_dict[key]['ts'])
        # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positionss'][:, 1], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 1] - data_dict[key]['pose_positions'][:, 1], label='cf.position()')
        plt.plot(data_dict[key]['ts'], zero_error)
    plt.subplot(3, 1, 3, sharex=ax1)
    for key in data_dict.keys():
        zero_error = np.zeros_like(data_dict[key]['ts'])
        # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positionss'][:, 2], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 2] - data_dict[key]['pose_positions'][:, 2], label=key+'_cf.position()')
        plt.plot(data_dict[key]['ts'], zero_error, label=key+'zero error')
    plt.legend()
    plt.suptitle('position error')

    plt.figure(3)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['thrust_cmds'])
    plt.title('Cmd z acc (python)')

    plt.figure(4)
    ax1 = plt.subplot(3, 1, 1)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_vel'][:, 0], label='/cf/pose vel')
        plt.plot(data_dict[key]['ts'], data_dict[key]['nominal_dyn'][:, 3])
    plt.subplot(3, 1, 2, sharex=ax1)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_vel'][:, 1], label='/cf/pose vel')
        plt.plot(data_dict[key]['ts'], data_dict[key]['nominal_dyn'][:, 4])

    plt.subplot(3, 1, 3, sharex=ax1)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_vel'][:, 2], label='/cf/pose vel')
        plt.plot(data_dict[key]['ts'], data_dict[key]['nominal_dyn'][:, 5], label='nominal')

    plt.legend()
    plt.suptitle('comparison vel')

    plt.figure(5)
    ax1 = plt.subplot(3, 1, 1)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['nominal_dyn'][:, 0])
    plt.subplot(3, 1, 2, sharex=ax1)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 1], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['nominal_dyn'][:, 1])

    plt.subplot(3, 1, 3, sharex=ax1)
    for key in data_dict.keys():
        plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 2], label='/cf/pose position')
        plt.plot(data_dict[key]['ts'], data_dict[key]['nominal_dyn'][:, 2], label='nominal')
    plt.legend()
    plt.suptitle('comparison pos')
    

    # # try:
    # plt.figure(4)
    # for key in data_dict.keys():
    #     # print(data_dict[key]['adaptation_terms'])
    #     plt.plot(data_dict[key]['ts'], data_dict[key]['adaptation_terms'])
    # plt.title('adaptation term')
    # except:
    #     pass

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    parser.add_argument("--runtime", type=float, default=10)
    parser.add_argument("--hovertime",type=float,default=3.97)
    parser.add_argument("-bh", "--baseheight", type=float, default=1.0)
    parser.add_argument("-tt", "--takeofftime",type=float,default=5.0)

    args = parser.parse_args()
    filenames = args.filename
    # print(filename)
    # exit()

    plot_npz(filenames)
    

