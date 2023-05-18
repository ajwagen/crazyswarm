import numpy as np
import argparse
import matplotlib.pyplot as plt

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def rmse(ref, act):
    return np.sqrt(np.mean((ref - act) ** 2, axis=0))

def plot_npz(filename, files_n_types, args):

    data_dict={}
    # minimum_len={}
    for i in filename:
        data = {}
        saved_data = dict(np.load(i, allow_pickle=True))

        minimum_len = np.inf
        for key in saved_data.keys():
            k = len(saved_data[key])
            if k<minimum_len:
                minimum_len = k
        
        for key in saved_data.keys():
            saved_data[key] = saved_data[key][:minimum_len]

        t_mask = (saved_data['ts'] > 5) * (saved_data['ts'] < args.runtime + 5)

        st= first_nonzero(saved_data['ref_positions'],0)[0]

        data['ts'] = saved_data['ts'][t_mask]
        data['pose_positions'] = saved_data['pose_positions'][t_mask] #- saved_data['pose_positions'][st]
        data['pose_positions'] -= data["pose_positions"][0]
        data['pose_orientations'] = saved_data['pose_orientations'][t_mask]

        # data['cf_positions'] = saved_data['cf_positions'][t_mask] - saved_data['cf_positions'][st]
        data['ref_positions'] = saved_data['ref_positions'][t_mask] #- saved_data['ref_positions'][st]
        data['ref_positions'] -= data['ref_positions'][0]

        data['ref_orientation'] = saved_data['ref_orientation'][t_mask]
        data['thrust_cmds'] = saved_data['thrust_cmds'][t_mask]
        data['ang_vel_cmds'] = saved_data['ang_vel_cmds'][t_mask]


        data_dict[i] = data

    sim_key = list(data_dict.keys())[0]
    # pos_rmse = np.sqrt(np.mean((data_dict[sim_key]['pose_positions'] - data_dict[sim_key]['ref_positions']) ** 2, axis=0))
    sim_rmse = rmse(data_dict[sim_key]['ref_positions'], data_dict[sim_key]['pose_positions'])
    print("sim RMSE : ", sim_rmse)

    real_rmse = []
    for key in list(data_dict.keys())[1:] :
        pos_rmse = rmse(data_dict[key]['ref_positions'], data_dict[key]['pose_positions']) 
        real_rmse.append(pos_rmse)

    real_rmse = np.mean(real_rmse, axis=0)
    print("real RMSE : ", real_rmse)

    if args.showgraph:
        plt.figure(0)
        ax1 = plt.subplot(3, 1, 1)
        plt.plot(data_dict[filename[0]]['ts'], data_dict[filename[0]]['ref_positions'][:, 0])
        for key in data_dict.keys():
            plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0])
            # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 0], label='cf.position()')
        plt.grid()
        
        plt.subplot(3, 1, 2, sharex=ax1)
        plt.plot(data_dict[filename[0]]['ts'], data_dict[filename[0]]['ref_positions'][:, 1])
        for key in data_dict.keys():
            plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 1])
            # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 1], label='cf.position()')
        plt.grid()
        
        plt.subplot(3, 1, 3, sharex=ax1)
        plt.plot(data_dict[filename[0]]['ts'], data_dict[filename[0]]['ref_positions'][:, 2],label='ref')
        for key in data_dict.keys():
            plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 2], label=files_n_types[key])
            # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 2], label=key+'_cf.position()')
        plt.grid()

        plt.legend()
        plt.suptitle(args.title + '   Position \n Sim RMSE {} \n Real RMSE {}'.format(np.round(sim_rmse, 3), np.round(real_rmse,3)))
        if args.showgraph:
            plt.savefig('temp', dpi=199)
            os.rename('temp.png', args.exp_date + '/plots/' + args.simtag + '.png')
        
        plt.figure(1)
        ax1 = plt.subplot(3, 1, 1)
        for key in data_dict.keys():
            # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 0], label='/cf/pose position')
            plt.plot(data_dict[key]['ts'], data_dict[key]['pose_orientations'][:, 0], label=key+'_cf.position()')
            plt.plot(data_dict[key]['ts'], data_dict[key]['ref_orientation'][:, 0])
        plt.subplot(3, 1, 2, sharex=ax1)
        for key in data_dict.keys():
            # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 1], label='/cf/pose position')
            plt.plot(data_dict[key]['ts'], data_dict[key]['pose_orientations'][:, 1], label=key+'_cf.position()')
            plt.plot(data_dict[key]['ts'], data_dict[key]['ref_orientation'][:, 1])
        plt.subplot(3, 1, 3, sharex=ax1)
        for key in data_dict.keys():
            # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 2], label='/cf/pose position')
            plt.plot(data_dict[key]['ts'], data_dict[key]['pose_orientations'][:, 2], label=key+'_cf.position()')
            plt.plot(data_dict[key]['ts'], data_dict[key]['ref_orientation'][:, 2],label=key+'_ref position')
        plt.legend()
        plt.suptitle(args.title + '  Attitude')
    
    
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
            # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 1], label='/cf/pose position')
            plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 1] - data_dict[key]['pose_positions'][:, 1], label='cf.position()')
            plt.plot(data_dict[key]['ts'], zero_error)
        plt.subplot(3, 1, 3, sharex=ax1)
        for key in data_dict.keys():
            zero_error = np.zeros_like(data_dict[key]['ts'])
            # plt.plot(data_dict[key]['ts'], data_dict[key]['pose_positions'][:, 2], label='/cf/pose position')
            plt.plot(data_dict[key]['ts'], data_dict[key]['ref_positions'][:, 2] - data_dict[key]['pose_positions'][:, 2], label=key+'_cf.position()')
            plt.plot(data_dict[key]['ts'], zero_error, label=key+'zero error')
        plt.legend()
        plt.suptitle(args.title + '  position error')
        if args.showgraph == 2:
            plt.show()

if __name__ == "__main__":

    from argparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument("--simtag", type = str, default=None)
    parser.add_argument("--exp-date", dest="exp_date", type=str, default="may_17")
    parser.add_argument("--plotall", type = bool, default = False)
    parser.add_argument("--showgraph", type=int, default=0)
    parser.add_argument("--runtime", type=float, default=10)
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    exp_date = args.exp_date
    plotall = args.plotall
    simtag = args.simtag

    print("RUN : ", simtag)

    files = []
    files_n_types = {}
    
    # sim
    files.append(exp_date + "/sim/" + simtag + ".npz")
    files_n_types = {exp_date + "/sim/" + simtag + ".npz" : "sim"}

    # real
    for i in range(3):
        if os.path.exists(exp_date + "/real/" + simtag + "_{}.npz".format(i)):
            files.append(exp_date + "/real/" + simtag + "_{}.npz".format(i))
            files_n_types[exp_date + "/real/" + simtag + "_{}.npz".format(i)] = "real_{}".format(i)

        else:
            break

    plot_npz(files, files_n_types, args)
    print("\n")

    
