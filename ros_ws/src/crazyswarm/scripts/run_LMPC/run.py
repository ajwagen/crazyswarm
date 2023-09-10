import os
import argparse
import yaml

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time

from quadsim.sim import QuadSim
from quadsim.models import crazyflieModel
from Controllers import LMPCController

# from quadsim.cf_utils.rigid_body import State_struct
class State_struct:
  def __init__(self, pos=np.zeros(3), 
                     vel=np.zeros(3),
                     acc = np.zeros(3),
                     jerk = np.zeros(3), 
                     snap = np.zeros(3),
                     rot=R.from_quat(np.array([0.,0.,0.,1.])), 
                     ang=np.zeros(3)):
    
    self.pos = pos # R^3
    self.vel = vel # R^3
    self.acc = acc
    self.jerk = jerk
    self.snap = snap
    self.rot = rot # Scipy Rotation rot.as_matrix() rot.as_quat()
    self.ang = ang # R^3
    self.t = 0.

from param_torch import Param, Timer
from quadrotor import Quadrotor
from quadrotor_torch import Quadrotor_torch
from quadsim.learning.refs.random_zigzag import RandomZigzag
import controller_torch

import gym
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def run_cf(cf, controller):
    timer = Timer(topics=['MPPI', 'sim'])
    state = State_struct()

    times = env.param.sim_times
    states = np.zeros((len(times), env.n))
    states_ref = np.zeros((len(times), env.n))
    actions = np.zeros((len(times)-1, env.m))

    printperiod = 0.2
    printtime = 0

    while True:
        reward = 0
        for step, t in enumerate(times[:-1]):
            if t + 1e-4 >= printtime:
              print(f'{t:0.01f} / {env.param.sim_tf:0.01f}')
              printtime += printperiod

            # Update the state
            quadsim_state = cf.rb.state()
            state.pos = quadsim_state.pos
            state.rot = quadsim_state.rot
            state.vel = quadsim_state.vel
            state.ang = quadsim_state.ang

            # Run the controller
            pos = state.pos
            vel = state.vel
            rot = state.rot
            ang = state.ang

            quat = rot.as_quat()
            # (x,y,z,w) -> (w,x,y,z)
            quat = np.roll(quat, 1)

            obs = np.r_[pos, vel, quat, ang]
            state_torch = torch.as_tensor(obs, dtype=torch.float32)
            # action = controller.policy_with_ref_func(state=state_torch, time=t, new_ref_func=ref_func)
            # action = action.cpu().numpy()

            # action[0] = action[0] / env.mass
            # z_acc, ang_vel = action[0], action[1:]

            z_acc, ang_vel = controller.response(state=quadsim_state, t=t, ref_func_obj=ref)


            # Step the environment
            obs_state = cf.step_angvel_raw(dt, z_acc * cf.mass, ang_vel, k=0.4)

            # Update the state
            quadsim_state = cf.rb.state()
            state.pos = quadsim_state.pos
            state.rot = quadsim_state.rot
            state.vel = quadsim_state.vel
            state.ang = quadsim_state.ang

            # Compute the reward
            quat = state.rot.as_quat()
            quat = np.roll(quat, 1)
            env.s = np.r_[state.pos, state.vel, quat, state.ang]
            env.time_step = step
            r = env.reward(np.r_[z_acc, ang_vel])
            reward += r

            # Visualization
            cf.vis.set_state(quadsim_state.pos, quadsim_state.rot)
            time.sleep(0.01)
        break
    return reward

def run(env, controller, initial_state):
    timer = Timer(topics=['MPPI', 'sim'])

    # environment
    times = env.param.sim_times
    states = np.zeros((len(times), env.n))
    states_ref = env.param.ref_trajectory.T[:len(times), :]
    actions = np.zeros((len(times)-1, env.m))

    if initial_state is None:
        initial_state = env.reset()

    print("Initial State: ", initial_state)

    states[0] = env.reset(initial_state)
    reward = 0

    printperiod = 0.2
    printtime = 0

    for step, time in enumerate(times[:-1]):
        if time + 1e-4 >= printtime:
          print(f'{time:0.01f} / {env.param.sim_tf:0.01f}')
          printtime += printperiod

        #state = states[step]
        #noise = np.random.normal(scale=env.param.noise_measurement_std)
        #noisystate = state + noise
        #noisystate[6:10] /= np.linalg.norm(noisystate[6:10])
        noisystate = states[step]

        timer.tic()
        s = torch.as_tensor(noisystate, dtype=torch.float32)
        action = controller.policy_with_ref_func(state=s, time=time, new_ref_func=ref_func).cpu().numpy()
        timer.toc('MPPI')

        timer.tic()
        states[step + 1], r, done, _ = env.step(action)
        reward += r
        actions[step] = action.flatten()
        timer.toc('sim')
        if done:
            break

    print('reward: ', reward)
    print('time: ', time)
    env.close()

    '''
    if not os.path.exists('data'):
        os.makedirs('data')
    tag = ''
    np.savetxt('data/' + tag + 'states.dat', states)
    np.savetxt('data/' + tag + 'actions.dat', actions)
    '''

    return timer, (times, states, actions, states_ref)

def plot(times, states, actions, states_ref, visualize):
    # plt.figure(figsize=(12, 12))
    # plt.subplot(3, 2, 1)
    # name = ['x', 'y', 'z']
    # for i in range(3):
    #     line, = plt.plot(times, states[:, i], label=name[i])
    #     plt.plot(times, states_ref[:, i], linestyle='--', color=line.get_color())
    # plt.legend()
    # plt.title('position')
    #
    # plt.subplot(3, 2, 2)
    # for i in range(3):
    #     line, = plt.plot(times, states[:, i+3], label=name[i])
    #     plt.plot(times, states_ref[:, i+3], linestyle='--', color=line.get_color())
    # plt.legend()
    # plt.title('velocity')
    #
    # quats_wlast = states[:, (7, 8, 9, 6)]
    # euler = np.degrees(R.from_quat(quats_wlast).as_euler('ZYX')[:, ::-1])
    # plt.subplot(3, 2, 3)
    # plt.plot(times, euler)
    # plt.legend(['x', 'y', 'z'])
    # plt.title('euler angle (deg)')
    #
    # plt.subplot(3, 2, 4)
    # plt.plot(times, states[:, 10:])
    # plt.legend(['x', 'y', 'z'])
    # plt.title('angular velocity')
    #
    # plt.subplot(3, 2, 5)
    # plt.plot(times[1:], actions[:, :] * 1000 / 9.81)
    # plt.legend(['motor 1', 'motor 2', 'motor 3', 'motor 4'])
    # plt.title('action (g)')
    # plt.show()
    # plt.savefig("rwik.jpg")

    # visualize
    if visualize:
        env.visualize(states, 0.02)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='Controllers/lmpc_config/lmpc.yaml', help='configuration yaml file')
    parser.add_argument("--vis", action='store_true', default=False, help='visualization using meshcat or not')
    parser.add_argument("--no-plot", action='store_true', default=False, help='disable matplotlib plot and timer prints')
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    param = Param(config)
    ref = RandomZigzag(fixed_seed=True)

    def ref_func(t):
        pos_ref = torch.as_tensor(ref.pos(t).T)
        vel_ref = torch.as_tensor(ref.vel(t).T)
        quat_ref = torch.as_tensor(ref.quat(t).T)
        omega_ref = torch.as_tensor(ref.angvel(t).T)
        state_ref = torch.cat((pos_ref, vel_ref, quat_ref, omega_ref), dim=-1)
        return state_ref

    env = Quadrotor(param, ref_func)
    model = crazyflieModel()
    cf = QuadSim(model)
    eu = np.array([0., 0., 0.])
    rot = R.from_euler('xyz', eu)
    init_state = State_struct(rot=rot)
    dt = 0.01

    # param_MPPI = Param(config, MPPI=True)
    # env_MPPI = Quadrotor_torch(param_MPPI, config)
    # #controller = controller_torch.MPPI_thrust_omega(env_MPPI, config)
    # controller = controller_torch.LMPC(env_MPPI, config)

    controller = LMPCController(isSim = True)


    s = np.array(config['initial_state'])
    #timer, runout = run(env, controller, initial_state=s)

    rewards = []
    for i in range(8):
        print(f'Episode {i}')
        ref.seed = i
        ref.reset()
        cf.setstate(init_state)
        reward = run_cf(cf, controller)
        rewards.append(reward)
    mean_cost = -np.mean(rewards)
    print(mean_cost)

    # print('****** Overall code profiling ******')
    # for topic in ['MPPI', 'sim']:
    #     print(topic+': ', np.mean(timer.stats[topic]), np.std(timer.stats[topic]))
    #
    # print('****** MPPI code profiling ******')
    # #'pos dynamics', 'att kinematics'
    # for topic in ['shift', 'sample', 'update', 'reward', 'pos dynamics', 'att kinematics']:
    #     print(topic+': ', np.mean(controller.timer.stats[topic]), np.std(controller.timer.stats[topic]))
    #
    # if not opt.no_plot:
    #   plot(*runout, visualize=opt.vis)

