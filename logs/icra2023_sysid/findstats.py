import numpy as np
import argparse
import matplotlib.pyplot as plt
from plt_utils import load_cf_data

def rmse(ref, act):
    return np.sqrt(np.mean((ref - act) ** 2, axis=0))

def plot_npz(filename):

    data_dict = load_cf_data(filenames, args)

    overall_mean_error = []
    ovr_std = []
    for key in list(data_dict.keys()):
        errors = np.linalg.norm(data_dict[key]['ref_positions'] - data_dict[key]['pose_positions'], axis =1)
        overall_mean_error.append(np.mean(errors))
        print(key, ' : ', np.mean(errors))

    # import pdb;pdb.set_trace()
    ovr_std = np.std(overall_mean_error)
    overall_mean_error_ = np.mean(overall_mean_error)

    print("overall error : ", overall_mean_error_, 'std : ', ovr_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs="+")
    parser.add_argument("--runtime", type=float, default=2.0)
    parser.add_argument("--hovertime",type=float,default=4.0)
    parser.add_argument("-bh", "--baseheight", type=float, default=1.0)
    parser.add_argument("-tt", "--takeofftime",type=float,default=5.0)

    args = parser.parse_args()
    filenames = args.filename
    # print(filename)
    # exit()

    plot_npz(filenames)
    
