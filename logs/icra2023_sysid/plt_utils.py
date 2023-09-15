import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch

def smoothing(arr, scale=5):
    smooth_horizon = 5
    smooth_arr = []
    horizon = np.zeros(smooth_horizon)
    for i in range(len(arr)):
        
        horizon[i % smooth_horizon] = arr[i]
        if i < smooth_horizon:
            smooth_arr.append(arr[i])
        else:
            smooth_arr.append(np.mean(horizon))
    
    return np.array(smooth_arr) / scale

def load_cf_data(filenames, args):
    data_dict={}
    # minimum_len={}
    # plt.figure(9)
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

        data['pose_orientations'] = saved_data['pose_orientations'][t_mask]
        data['pose_vel'] = np.diff(data['pose_positions'], axis=0) / np.diff(data['ts'])[:, None]#- saved_data['ref_positions'][st]
        data['pose_vel'] = np.r_[data['pose_vel'][0][None,  :], data['pose_vel']]

        # data['cf_positions'] = saved_data['cf_positions'][t_mask] - saved_data['cf_positions'][st]


        data['ref_orientation'] = saved_data['ref_orientation'][t_mask]
        data['thrust_cmds'] = saved_data['thrust_cmds'][t_mask]
        data['ang_vel_cmds'] = saved_data['ang_vel_cmds'][t_mask]

        try:
            if isinstance(saved_data['adaptation_terms'][0], torch.Tensor):
                saved_data['adaptation_terms'] = [saved_data['adaptation_terms'][i].numpy() for i in range(len(saved_data['adaptation_terms']))]
            data['adaptation_terms'] = saved_data['adaptation_terms'][t_mask]
            # if 'real' not in i:
            # plt.plot(saved_data['ts'], np.array(saved_data['adaptation_terms'])[:, 0], label=i)
            # else:
                # plt.plot(saved_data['ts'], np.array(saved_data['adaptation_terms'])[:, 0] / 1.3, label=i)

        except:
            pass
        
        data_dict[i] = data
    # plt.legend()
    # plt.show()

    try :
        plt.figure(10)
        ax0 = plt.subplot(2, 1, 1)
        # plt.plot(data_dict[filenames[0]]['ts'], data_dict[filenames[0]]['adaptation_terms'][:, 1])
        for key in data_dict.keys():

            plt.plot(data_dict[key]['ts'], data_dict[key]['adaptation_terms'][:, 1])
            # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 0], label='cf.position()')
        # plt.grid()
        ax0.set(ylabel='X (m / s^2)')
        ax1 = plt.subplot(2, 1, 2, sharex=ax0)
        # plt.plot(data_dict[filenames[0]]['ts'], data_dict[filenames[0]]['ref_positions'][:, 1])
        for key in data_dict.keys():
            lab = 'without wind'
            if "wind" in key:
                lab = 'with wind'
            plt.plot(data_dict[key]['ts'], data_dict[key]['adaptation_terms'][:, 2], label=lab)
            # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 1], label='cf.position()')
        # plt.grid()
        ax1.set(ylabel='Y (m / s^2)')
        # plt.subplot(3, 1, 3, sharex=ax1)
        # # plt.plot(data_dict[filenames[0]]['ts'], data_dict[filenames[0]]['ref_positions'][:, 2],label='ref')
        # for key in data_dict.keys():
        #     # if 'real' in key:
        #         # data_dict[key]['adaptation_terms'][:, 3] = smoothing(data_dict[key]['adaptation_terms'][:, 3])
        #     plt.plot(data_dict[key]['ts'], data_dict[key]['adaptation_terms'][:, 3], label=key)
            # plt.plot(data_dict[key]['ts'], data_dict[key]['cf_positions'][:, 2], label=key+'_cf.position()')
        # plt.grid()
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])

        # Put a legend below current axis
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
                fancybox=False, shadow=False, ncol=5, frameon=False)

        # AXES_SIZE = 20
        # LEGEND_SIZE = 20
        # plt.rc('legend', fontsize=LEGEND_SIZE)
        # plt.rc('axes', labelsize=AXES_SIZE)
        # plt.rc('xtick', labelsize=AXES_SIZE)
        # plt.rc('ytick', labelsize=AXES_SIZE)
        # plt.xlabel('time (s)')
        # plt.show()
        # plt.legend()
    except:
        pass
    # plt.show()
    # exit()


    # exit()
    
    return data_dict