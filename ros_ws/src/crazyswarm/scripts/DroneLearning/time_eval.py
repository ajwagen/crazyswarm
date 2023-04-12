import numpy as np
import os

t = []
with open("cpp.out") as file:
    for line in file:
        num = line.strip('\n')
        num = float(num)
        t.append(num)

t = np.array(t)

print('c++ cuda mean: ', np.mean(t))
print('c++ cuda std: ', np.std(t))
print('c++ cuda max: ', np.max(t))
print('c++ cuda min: ', np.min(t))

print("--------------------")

t = []
with open("py.out") as file:
    for line in file:
        num = line.strip('\n')
        num = float(num)
        t.append(num)

t = np.array(t)

print('py  cuda mean: ', np.mean(t))
print('py  cuda std: ', np.std(t))
print('py  cuda max: ', np.max(t))
print('py  cuda min: ', np.min(t))