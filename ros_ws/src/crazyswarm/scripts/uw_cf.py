#!/usr/bin/env python
import time
import argparse
from easydict import EasyDict
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

# import controller 
from Controllers.pid_controller import PIDController
from Controllers.hover_PPO_controller import PPOController

# Actual Drone
import rospy
from geometry_msgs.msg import PoseStamped
from pycrazyswarm import Crazyswarm
# from quadsim.rigid_body import State

# Quadsim simulator
from quadsim.sim import QuadSim
from quadsim.models import crazyflieModel, IdentityModel

np.set_printoptions(linewidth=np.inf)

sleepRate = 50

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
    def __init__(self, cfName,sim=False,config_file="cf_config.yaml"):
        self.cfName = cfName
        self.isSim = sim

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
        else:
            model = crazyflieModel()
            self.cf = QuadSim(model,name=self.cfName)
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
        ref_pos = np.array([0.,0.0,0.0])
        ref_vel = np.array([0.,0.,0])
        self.ref = Ref_State(pos=ref_pos,vel = ref_vel)

    def state_callback(self, data):
        pos = data.pose.position
        rot = data.pose.orientation

        t = rospy.get_time()
        self.state[-1] = t
        # print(t - self.prev_state[-1])
        self.state[0:3] = np.array([pos.x,pos.y,pos.z])

        # Numerical integration to calculate velocity
        vel = (self.state[0:3]-self.prev_state[0:3])/(0.1)
        self.state[3:6] = vel        
        self.state[6:10] = np.array([rot.x,rot.y,rot.z,rot.w])
        self.prev_state = self.state.copy()

    def _send2cfClient(self,cf,z_acc,ang_vel):
        pos = [0,0,0]
        vel = [0,0,0]
        acc = [0,0,z_acc]
        yaw = 0
        omega = ang_vel.tolist()
        cf.cmdFullState(pos,vel,acc,yaw,omega)

    def take_off(self,takeoff_height, takeoff_time, init_pos,t):
        take_offRef = Ref_State(pos = init_pos+np.array([0.,0.,takeoff_height/takeoff_time*t]))
        z_acc, ang_vel = self.pid_controller.response(t,self.state,take_offRef)
        # print(np.hstack((self.state[:3],take_offRef.pos)))
        return z_acc,ang_vel, take_offRef.pos
    
    def land(self,):
        pass
    def main_loop_cf(self,):
        timeHelper = self.swarm.timeHelper
        cf = self.swarm.allcfs.crazyflies[0]
        
        init_pos = cf.position()
        timeHelper.sleep(2.0)
        
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


        while not rospy.is_shutdown() and t <25.0:
            # r = rospy.Rate(100) 
            t = timeHelper.time() - startTime
            # print(t)
            if t<takeoff_time:
                if takeoff_flag==0:
                    # print("********* TAKEOFF **********")
                    takeoff_flag = 1
                z_acc,ang_vel, _ref = self.take_off(takeoff_height, takeoff_time, init_pos,t)
                # offset_pos = cf.position()


            ########################################################
            elif t<takeoff_time+5.:
                #HOVER
                if task1_flag==0:
                    # print("********* TASK PID********")
                    task1_flag = 1
                    offset_pos = init_pos+np.array([0.,0.,takeoff_height])
                    self.ref.pos +=offset_pos
                    _ref = self.ref.pos
                    # print(self.ref.pos)
                z_acc, ang_vel = self.pid_controller.response(t,self.state,self.ref)

            elif t<takeoff_time+5.+10.:
                #HOVER
                if task2_flag==0:
                    # print("********* TASK PPO********")
                    task2_flag = 1
                    _ref = self.ref.pos

                z_acc, ang_vel = self.ppo_controller.response(t,self.state,self.ref)
            # #########################################################
                      
            else:
                if land_flag==0:
                    # print("********* LAND **********")
                    land_flag = 1
                    land_pos = cf.position()
                    land_pos[-1] = landing_height
                    land_ref = Ref_State(pos=land_pos)
                    _ref = land_ref.pos

                z_acc, ang_vel = self.pid_controller.response(t,self.state,land_ref)

            print(np.hstack((self.state[:3],_ref,t)))


            self._send2cfClient(cf,z_acc, ang_vel)

        pos = [0,0,0]
        vel = [0,0,0]
        acc = [0,0,z_acc]
        yaw = 0
        omega = ang_vel.tolist()
        cf.cmdFullState(pos,vel,np.array([0.,0.,0.]),yaw,omega)

    def main_loop_sim(self,):
        i = 0
        
        while i*self.dt<30.0:
            ref = self.ref

            self.cf.step_angvel_cf(i*self.dt,self.dt,self.ppo_controller,self.ref)
            quadsim_state = self.cf.rb.state()
            # print(quadsim_state.pos)
            self.cf.vis.set_state(quadsim_state.pos,quadsim_state.rot)
            time.sleep(self.dt/2)
            i+=1

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--quadsim', action='store_true')
    parser.add_argument('--config', action='store', type=str, default="cf_config.yaml")
    g = EasyDict(vars(parser.parse_args()))
    # print(g.config)
    # exit()
    x = ctrlCF("cf1",sim=g.quadsim,config_file=g.config)
    if g.quadsim:
        x.main_loop_sim()
        # x.learning_loop()
    else:
        x.main_loop_cf()
    

