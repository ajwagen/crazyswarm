import numpy as np
#from ref_traj import Trajectories
import matplotlib.pyplot as plt


'''
traj_id = 'random_chained_poly'
maxes = [1.0, 0.0, 0.0]
seed = 0

init_pos = np.array([0., 0., 0.])
gui = None
traj = Trajectories(init_pos, gui)

init_ref_func = getattr(traj, traj_id+"_")
ref_kwargs = {'seed':seed}
ref_kwargs['maxes'] = maxes
init_ref_func(**ref_kwargs)
'''

start_idx = 400
stop_idx = start_idx + int(10*1/0.02)

#path1 = './../../../../../logs/icra2023_sysid/nov_16/real_fan/'
path1 = "./../../../../../logs/icra2023_sysid/april_12_24/real/"
#file_ids1 = ['hover_plate_wind_seed0.npz','hover_plate_wind_seed0_1.npz','hover_plate_wind_seed0_2.npz','hover_plate_wind_seed0_3.npz']
#file_ids1 = ['poly_plate_fan_0.npz','poly_plate_fan_1.npz','poly_plate_fan_2.npz','poly_plate_fan_3.npz','poly_plate_fan_4.npz']
#file_ids1 = ['poly_plate_wind_pid2_seed0_0.npz','poly_plate_wind_pid2_seed0_1.npz','poly_plate_wind_pid2_seed0_2.npz']
#file_ids1 = ['naive_ctrl_policy_drag_plate_mppi_spd_1_trial_1.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_1_trial_2.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_1_trial_3.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_1_trial_4.npz','naive_ctrl_policy_drag_plate_mppi_spd_1_trial_5.npz']
#file_ids1 = ['naive_ctrl_policy_drag_plate_mppi_spd_2_trial_1.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_2_trial_2.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_2_trial_3.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_2_trial_4.npz','naive_ctrl_policy_drag_plate_mppi_spd_2_trial_5.npz']
#file_ids1 = ['naive_ctrl_policy_mppi_spd_2_trial_1.npz', 'naive_ctrl_policy_mppi_spd_2_trial_2.npz', 'naive_ctrl_policy_mppi_spd_2_trial_3.npz', 'naive_ctrl_policy_mppi_spd_2_trial_4.npz','naive_ctrl_policy_mppi_spd_2_trial_5.npz']
#file_ids1 = ['naive_ctrl_policy_mppi_spd_1_trial_1.npz', 'naive_ctrl_policy_mppi_spd_1_trial_2.npz', 'naive_ctrl_policy_mppi_spd_1_trial_3.npz', 'naive_ctrl_policy_mppi_spd_1_trial_4.npz','naive_ctrl_policy_mppi_spd_1_trial_5.npz']
#file_ids1 = ['naive_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_1.npz', 'naive_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_2.npz', 'naive_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_3.npz', 'naive_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_4.npz', 'naive_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_5.npz' ]
file_ids1 = ['naive_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_1.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_2.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_3.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_4.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_5.npz']
#file_ids1 = ['naive_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_1.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_2.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_3.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_4.npz', 'naive_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_5.npz']
residual1 = None
residual1_all = None
count = 0
for file in file_ids1:
	data1 = np.load(path1 + file)
	if residual1 is None:
		residual1 = np.abs(data1['cf_positions'][start_idx:stop_idx,:] - data1['ref_positions'][start_idx:stop_idx,:])
		residual1_all = np.zeros((len(file_ids1),len(residual1),3))
		residual1_all[count,:,:] = residual1
	else:
		residual1 += np.abs(data1['cf_positions'][start_idx:stop_idx,:] - data1['ref_positions'][start_idx:stop_idx,:])
		residual1_all[count,:,:] = np.abs(data1['cf_positions'][start_idx:stop_idx,:] - data1['ref_positions'][start_idx:stop_idx,:])
	count += 1
residual1 /= len(file_ids1)

#path2 = './../../../../../logs/icra2023_sysid/nov_16/real_fan/' 
path2 = "./../../../../../logs/icra2023_sysid/april_12_24/real/"
#file_ids2 = ['hover_plate_wind_learned_seed0_2.npz','hover_plate_wind_learned_seed0_3.npz','hover_plate_wind_learned_seed0_4.npz', \
#	'hover_plate_wind_learned_seed0_5.npz']
#file_ids2 = ['poly_plate_fan_learned_0.npz','poly_plate_fan_learned_1.npz','poly_plate_fan_learned_2.npz', \
	#'poly_plate_fan_learned_3.npz', 'poly_plate_fan_learned_4.npz']
#file_ids2 = ['poly_plate_wind_neuralfly_seed0_1.npz']
#file_ids2 = ['L1_adapt_policy_drag_plate_mppi_spd_1_trial_1.npz', 'L1_adapt_policy_drag_plate_mppi_spd_1_trial_2.npz','L1_adapt_policy_drag_plate_mppi_spd_1_trial_3.npz','L1_adapt_policy_drag_plate_mppi_spd_1_trial_4.npz','L1_adapt_policy_drag_plate_mppi_spd_1_trial_5.npz']
#file_ids2 = ['L1_adapt_policy_drag_plate_mppi_spd_2_trial_1.npz', 'L1_adapt_policy_drag_plate_mppi_spd_2_trial_2.npz','L1_adapt_policy_drag_plate_mppi_spd_2_trial_3.npz','L1_adapt_policy_drag_plate_mppi_spd_2_trial_4.npz','L1_adapt_policy_drag_plate_mppi_spd_2_trial_5.npz']
#file_ids2 = ['L1_adapt_policy_mppi_spd_2_trial_1.npz', 'L1_adapt_policy_mppi_spd_2_trial_2.npz','L1_adapt_policy_mppi_spd_2_trial_3.npz','L1_adapt_policy_mppi_spd_2_trial_4.npz','L1_adapt_policy_mppi_spd_2_trial_5.npz']
#file_ids2 = ['L1_adapt_policy_mppi_spd_1_trial_1.npz', 'L1_adapt_policy_mppi_spd_1_trial_2.npz','L1_adapt_policy_mppi_spd_1_trial_3.npz','L1_adapt_policy_mppi_spd_1_trial_4.npz','L1_adapt_policy_mppi_spd_1_trial_5.npz']
#file_ids2 = ['L1_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_1.npz', 'L1_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_2.npz', 'L1_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_3.npz', 'L1_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_4.npz', 'L1_adaptation_policy_mppi_spd_2_2_fans_4_12_trial_5.npz']
file_ids2 = ['L1_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_1.npz', 'L1_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_2.npz', 'L1_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_3.npz', 'L1_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_4.npz', 'L1_ctrl_policy_drag_plate_mppi_spd_2_2_fans_trial_5.npz']
#file_ids2 = ['L1_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_1.npz', 'L1_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_2.npz', 'L1_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_3.npz', 'L1_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_4.npz', 'L1_ctrl_policy_drag_plate_mppi_spd_3_2_fans_trial_5.npz']
residual2 = None
residual2_all = None
count = 0
for file in file_ids2:
	data2 = np.load(path2 + file)
	if residual2 is None:
		residual2 = np.abs(data2['cf_positions'][start_idx:stop_idx,:] - data2['ref_positions'][start_idx:stop_idx,:])
		residual2_all = np.zeros((len(file_ids2),len(residual2),3))
		residual2_all[0,:,:] = residual2
	else:
		residual2 += np.abs(data2['cf_positions'][start_idx:stop_idx,:] - data2['ref_positions'][start_idx:stop_idx,:])
		residual2_all[count,:,:] = np.abs(data2['cf_positions'][start_idx:stop_idx,:] - data2['ref_positions'][start_idx:stop_idx,:])
	count += 1
residual2 /= len(file_ids2)
#for item in data1:
#	print(item)
#	print(data1[item].shape)

std_vals1 = np.std(np.mean(residual1_all, axis=1), axis=0)
std_vals2 = np.std(np.mean(residual2_all, axis=1), axis=0)


names = ['x','y','z']

for i in range(3):
	plt.subplot(1,3,i+1)
	plt.plot(residual1[:,i], label='residual pid')
	plt.plot(residual2[:,i], label='residual learned')
	plt.title(names[i])
	mse1 = np.linalg.norm(residual1[:,i]) / np.sqrt(stop_idx - start_idx)
	mse2 = np.linalg.norm(residual2[:,i]) / np.sqrt(stop_idx - start_idx)
	print(names[i] + ' mse: traj 1 = ' + str(mse1) + ', traj 2 = ' + str(mse2))
	print(names[i] + ' std: traj 1 = ' + str(std_vals1[i]) + ', traj 2 = ' + str(std_vals2[i]))
plt.legend()


plt.figure()
for i in range(3):
	plt.subplot(1,3,i+1)
	plt.plot(np.std(residual1_all[:,:,i], axis=0), label='std residual pid')
	plt.plot(np.std(residual2_all[:,:,i], axis=0), label='std residual learned')
	plt.title(names[i])
	#mse1 = np.linalg.norm(residual1[:,i]) / np.sqrt(stop_idx - start_idx)
	#mse2 = np.linalg.norm(residual2[:,i]) / np.sqrt(stop_idx - start_idx)
	#print(names[i] + ' std: traj 1 = ' + str(mse1) + ', traj 2 = ' + str(mse2))
plt.legend()
plt.show()






