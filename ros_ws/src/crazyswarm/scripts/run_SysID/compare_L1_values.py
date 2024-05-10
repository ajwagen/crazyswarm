import numpy as np

dir_ = '/home/drones/drones_project/crazyswarm/logs/icra2023_sysid/april_12_24/real/'
FILES = []
start_idx = 50
end_idx = 300
num_trials = 4
FILE_NAME = 'mppi_no_policy_3fans_speed3_may_8_trial_' 
for i in range(1, num_trials+1):
    FILES.append(FILE_NAME + str(i)+'.npz')
def main():
    for file in FILES:
        data = np.load(dir_ + file)
        #L1_learned = data['L1_adaptation_terms'][start_idx:end_idx]
        L1_learned = data['L1_adaptation_terms'][start_idx:end_idx]

        #L1_basic = data['adaptation_terms'][start_idx:end_idx]
        L1_basic = data['adaptation_terms'][start_idx:end_idx, 1:4]
        diffs = L1_basic - L1_learned
        diffs = np.abs(diffs)
        print(file)
        print(diffs.shape)
        print('mean difference between L1_basic and L1_learned: ',np.mean(diffs, axis=0))
        print('median difference between L1_basic and L1_learned: ',np.median(diffs, axis=0))
        print(50*'-')
        #print(L1_learned.shape)
        #print(L1_basic.shape)
        #print(data.files)
        #print(data['L1_adaptation_terms'][200:500])
        #print(data['adaptation_terms'][200:500])

if __name__ == '__main__':
    main()
