#!/usr/bin/env python
import time
import argparse
from easydict import EasyDict
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

# import controller 
from Controllers.pid_controller import PIDController
from Controllers.hover_ppo_controller import PPOController

# Actual Drone
import rospy
from geometry_msgs.msg import PoseStamped
from pycrazyswarm import Crazyswarm
# from quadsim.rigid_body import State

# Quadsim simulator
from quadsim.sim import QuadSim
from quadsim.models import IdentityModel,crazyflieModel

from pathlib import Path

np.set_printoptions(linewidth=np.inf)

SLEEP_RATE = 50

class Ref_State:
  def __init__(self, pos=np.zeros(3), 
                     vel=np.zeros(3),
                     acc = np.zeros(3), 
                     rot=R.identity(), 
                     ang=np.zeros(3)):
    
    self.pos = pos # R^3
    self.vel = vel # R^3
    self.acc = acc
    self.rot = rot # Scipy Rotation rot.as_matrix() rot.as_quat()
    self.ang = ang # R^3

# use pop instead
# collection -> deque
def add2Queue(array, new):
    array[0:-1] = array[1:]
    array[-1] = new
    return array

class ctrlCF():
    def __init__(self, cfName, sim=False, config_file="cf_config.yaml", log_file='log.npz', debug=False):
        self.cfName = cfName
        self.isSim = sim
        self.logfile = log_file
        self.debug = debug

        self.initialized = False
        self.state = np.zeros(14)
        self.prev_state = np.zeros(14)
        self.pid_controller  = PIDController(isSim = self.isSim)
        self.ppo_controller = PPOController(isSim = self.isSim)
        time.sleep(2.0)
        if not self.isSim:
            self.swarm = Crazyswarm()
            rospy.Subscriber("/"+self.cfName+"/pose", PoseStamped, self.state_callback)
            rospy.on_shutdown(self.shutdown_callback)
            self.emergency_signal = 0
            self.cf = self.swarm.allcfs.crazyflies[0]
        else:
            model = crazyflieModel()
            self.cf = QuadSim(model, name=self.cfName)
            self.dt = 0.005
        with open(config_file,"r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.set_ref()

    def shutdown_callback(self,):
        # pass
        # if self.emergency_signal:
        self.swarm.allcfs.emergency()
        # else:
        #     pass

    def set_ref(self,):
        ref_pos = np.array([0., 0.0, 1.0])
        ref_vel = np.array([0., 0., 0])
        self.ref = Ref_State(pos=ref_pos, vel=ref_vel)

    def state_callback(self, data):
        pos = data.pose.position
        rot = data.pose.orientation

        t = rospy.get_time()
        self.state[-1] = t
        self.pose_pos = np.array([pos.x, pos.y, pos.z])
        self.pos_pos = self.cf.position()
        self.state[0:3] = self.pos_pos


        # Numerical integration to calculate velocity
        vel = (self.state[0:3] - self.prev_state[0:3]) / (0.1)
        self.state[3:6] = vel        
        self.state[6:10] = np.array([rot.x, rot.y, rot.z, rot.w])
        self.prev_state = self.state.copy()
        self.initialized = True

    def _send2cfClient(self, cf, z_acc, ang_vel):
        pos = self.ref.pos
        vel = [0, 0, 0]
        acc = [0, 0, z_acc]
        yaw = 0
        omega = ang_vel.tolist()
        cf.cmdFullState(pos, vel, acc, yaw, omega)

    def BB_failsafe(self, bound = 1.0):
        # pos = cf.position()#self.state[:3]
        # print(abs(pos[0] - self.init_pos[0]), abs(pos[1] - self.init_pos[1]), pos[-1]>1.0)
        pos = self.cf.position()
        if abs(pos[0] - self.init_pos[0]) > bound/2 or abs(pos[1] - self.init_pos[1]) > bound/2 or pos[2]>1.2:
            print('Out of Bounding Box EMERGENCY STOP!!')
            self.swarm.allcfs.emergency()
            self.write_to_log()
            exit()

    def take_off(self,takeoff_height, takeoff_time, init_pos,t):
        take_offRef = Ref_State(pos = init_pos + np.array([0., 0., min(takeoff_height / takeoff_time * t, takeoff_height)]))
        self.ref = take_offRef
        z_acc, ang_vel = self.pid_controller.response(t,self.state,take_offRef)
        # print(np.hstack((self.state[:3],take_offRef.pos)))
        return z_acc,ang_vel, take_offRef.pos
    
    def land(self,):
        pass

    def main_loop_cf(self,):
        timeHelper = self.swarm.timeHelper

        while not self.initialized:
            pass
        
        init_pos = np.copy(self.state[:3])
        self.init_pos = init_pos
        timeHelper.sleep(0.5)

        t = 0.0
        startTime = timeHelper.time()

        takeoff_time = self.config["takeoff_height"]/self.config["takeoff_rate"]
        takeoff_height = self.config["takeoff_height"]

        landing_time = self.config["landing_height"]/self.config["landing_rate"]
        landing_height = self.config["landing_height"]
        land_flag = 0
        takeoff_flag = 0
        task1_flag = 0
        task2_flag = 0

        self.pose_positions = []
        self.pose_orientations = []
        self.cf_positions = []
        self.ts = []
        self.thrust_cmds = []
        self.ang_vel_cmds = []

        time_limit = 25000 if self.debug else 25.0

        while not rospy.is_shutdown() and t < time_limit:
            if not self.debug:
                self.BB_failsafe()   

            
            t = timeHelper.time() - startTime

            wait_time = 10.0
            if t < wait_time:
                z_acc, ang_vel = 0.0, np.zeros(3)
                pass

            elif t < takeoff_time + 100000:
                if takeoff_flag==0:
                    # print("********* TAKEOFF **********")
                    takeoff_flag = 1
                z_acc,ang_vel, _ref = self.take_off(takeoff_height, takeoff_time, init_pos,t - wait_time)
                # offset_pos = cf.position()


            ########################################################
            elif t < takeoff_time:
                #HOVER
                if task1_flag==0:
                    # print("********* TASK PID********")
                    task1_flag = 1
                    offset_pos = init_pos + np.array([0., 0., takeoff_height])
                    self.ref.pos += offset_pos
                    _ref = self.ref.pos
                    # print(self.ref.pos)
                z_acc, ang_vel = self.pid_controller.response(t - wait_time, self.state, self.ref)

            # elif t<takeoff_time+5.+10.:
            #     #HOVER
            #     if task2_flag==0:
            #         # print("********* TASK PPO********")
            #         task2_flag = 1
            #         _ref = self.ref.pos

            #     z_acc, ang_vel = self.ppo_controller.response(t,self.state,self.ref)
            # # #########################################################
                      
            else:
                if land_flag == 0:
                    # print("********* LAND **********")
                    land_flag = 1
                    land_pos = self.cf.position()
                    land_pos[-1] = landing_height
                    land_ref = Ref_State(pos=land_pos)
                    _ref = land_ref.pos

                z_acc, ang_vel = self.pid_controller.response(t - wait_time, self.state, land_ref)

            self.pose_positions.append(np.copy(self.pose_pos))
            quat = self.state[6:10]
            rot = R.from_quat(quat)
            eulers = rot.as_euler('ZYX', degrees=True)
            self.pose_orientations.append(eulers)
            self.cf_positions.append(self.cf.position())
            self.ts.append(t)
            self.thrust_cmds.append(z_acc)
            self.ang_vel_cmds.append(ang_vel * 180 / (2*np.pi))
            if self.debug:
                z_acc = 0.0
                ang_vel = np.zeros(3)
                # z_acc = 0.3*np.sin(t) + 0.7
                # ang_vel = np.array([0.25*np.sin(t), 0.25*np.cos(t), 0.0])
            print("pos", self.pose_pos, 'zacc', z_acc, "act",self.cf.position(),"t",t)


            self._send2cfClient(self.cf,z_acc, ang_vel)

            timeHelper.sleepForRate(SLEEP_RATE)

    def write_to_log(self):
        LOG_DIR = Path().home() / 'Drones' / 'logs'

        self.pose_positions = np.array(self.pose_positions)
        print(self.pose_positions)
        self.pose_orientations = np.array(self.pose_orientations)
        self.cf_positions = np.array(self.cf_positions)
        self.ts = np.array(self.ts)
        self.thrust_cmds = np.array(self.thrust_cmds)
        self.ang_vel_cmds = np.array(self.ang_vel_cmds)
        np.savez(
            LOG_DIR / self.logfile, 
            pose_positions=self.pose_positions,
            pose_orientations=self.pose_orientations,
            cf_positions=self.cf_positions,
            ang_vel_cmds=self.ang_vel_cmds,
            ts=self.ts,
            thrust_cmds=self.thrust_cmds
        )

    def main_loop_sim(self,):
        i = 0
        
        while i*self.dt < 30.0:
            ref = self.ref

            self.cf.step_angvel_cf(i * self.dt, self.dt, self.ppo_controller, ref=self.ref)
            quadsim_state = self.cf.rb.state()
            self.cf.vis.set_state(quadsim_state.pos, quadsim_state.rot)
            time.sleep(self.dt / 2)
            i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--quadsim', action='store_true')
    parser.add_argument('--config', action='store', type=str, default="cf_config.yaml")
    parser.add_argument('--logfile', action='store', type=str, default='log.npz')
    parser.add_argument('--debug', action='store', type=bool, default=False)
    g = EasyDict(vars(parser.parse_args()))
    debug = g.debug
    x = ctrlCF("cf2", sim=g.quadsim, config_file=g.config, log_file=g.logfile, debug=debug)
    try:
        if g.quadsim:
            x.main_loop_sim()
            # x.learning_loop()
        else:
            x.main_loop_cf()
    except KeyboardInterrupt:
        x.write_to_log()
    else:
        x.write_to_log()
    

