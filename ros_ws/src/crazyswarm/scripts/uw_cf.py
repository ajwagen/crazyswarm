#!/usr/bin/env python
import time
import argparse
from easydict import EasyDict
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
import copy

# import controller 
from Controllers.pid_controller import PIDController
# from Controllers.hover_PPO_controller import PPOController

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

sleepRate = 50

class State_struct:
  def __init__(self, pos=np.zeros(3), 
                     vel=np.zeros(3),
                     acc = np.zeros(3), 
                     rot=R.from_quat(np.array([0.,0.,0.,1.])), 
                     ang=np.zeros(3)):
    
    self.pos = pos # R^3
    self.vel = vel # R^3
    self.acc = acc
    self.rot = rot # Scipy Rotation rot.as_matrix() rot.as_quat()
    self.ang = ang # R^3
    self.t = 0.

class ctrlCF():
    def __init__(self, cfName,sim=False,config_file="cf_config.yaml", log_file='log.npz'):
        self.cfName = cfName
        self.isSim = sim
        self.logfile = log_file

        self.initialized = False
        self.state = State_struct()
        self.prev_state = State_struct()
        self.ref = State_struct()
        # self.state = np.zeros(14)
        # self.prev_state = np.zeros(14)
        self.pid_controller  = PIDController(isSim = self.isSim)
        # self.ppo_controller = PPOController(isSim = self.isSim)
        if not self.isSim:
            self.swarm = Crazyswarm()
            rospy.Subscriber("/"+self.cfName+"/pose", PoseStamped, self.state_callback)
            rospy.on_shutdown(self.shutdown_callback)
            self.emergency_signal = 0
            self.cf = self.swarm.allcfs.crazyflies[0]

        else:
            model = crazyflieModel()
            self.cf = QuadSim(model,name=self.cfName)
            self.dt = 0.005
        with open(config_file,"r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def shutdown_callback(self,):

        self.swarm.allcfs.emergency()

    def set_hover_ref(self,t):
        ref_pos = np.array([0.,0.0,0.3])
        ref_vel = np.array([0.,0.,0])
        self.ref = State_struct(pos=ref_pos,vel = ref_vel)
    
    def set_takeoff_ref(self,t,takeoff_rate,takeoff_height):
        ref_pos = self.init_pos+np.array([0.,0.,min(takeoff_rate*t,takeoff_height)])
        ref_vel = np.array([0.,0.,0])
        self.ref = State_struct(pos=ref_pos,vel = ref_vel)

    # def state_callback_legacy(self, data):
    #     pos = data.pose.position
    #     rot = data.pose.orientation

    #     t = rospy.get_time()
    #     self.state[-1] = t
    #     # print(t - self.prev_state[-1])
    #     # self.state[0:3] = np.array([pos.x,pos.y,pos.z])
    #     self.pose_pos = np.array([pos.x,pos.y,pos.z])
    #     self.pos_pos = self.cf.position()
    #     self.state[0:3] = self.pos_pos


    #     # Numerical integration to calculate velocity
    #     vel = (self.state[0:3]-self.prev_state[0:3])/(0.1)
    #     self.state[3:6] = vel        
    #     self.state[6:10] = np.array([rot.x,rot.y,rot.z,rot.w])
    #     self.prev_state = self.state.copy()
    #     self.initialized = True
    
    def state_callback(self, data):
        pos = data.pose.position
        rot = data.pose.orientation

        self.state.t = rospy.get_time()
        self.pose_pos = np.array([pos.x,pos.y,pos.z])
        self.pos_pos = self.cf.position()
        rot = np.array([rot.x,rot.y,rot.z,rot.w])

        # Adding ROS subscribed data to state
        self.state.pos = self.pos_pos
        self.state.vel = (self.state.pos-self.prev_state.pos)/(0.1) 
        self.state.rot = R.from_quat(rot)
        self.prev_state = copy.deepcopy(self.state)
        self.initialized = True

    def _send2cfClient(self,cf,z_acc,ang_vel):
        pos = self.ref.pos
        vel = [0,0,0]
        acc = [0,0,z_acc]
        yaw = 0
        omega = ang_vel.tolist()
        cf.cmdFullState(pos,vel,acc,yaw,omega)

    def BB_failsafe(self,):
        pos = self.state.pos
        w_bound = self.config["E_BB_width"]
        h_bound = self.config["E_BB_height"]
        if abs(pos[0] - self.init_pos[0]) > w_bound/2 or abs(pos[1] - self.init_pos[1]) > w_bound/2 or pos[2]>h_bound:
            print('Out of Bounding Box EMERGENCY STOP!!')
            self.swarm.allcfs.emergency()
            self.write_to_log()
            exit()


    # def take_off(self,takeoff_height, takeoff_rate, init_pos,t):
    #     take_offRef = State_struct(pos = init_pos+np.array([0.,0.,min(takeoff_rate*t,takeoff_height)]))
    #     self.ref = take_offRef
    #     z_acc, ang_vel = self.pid_controller.response(t,self.state,take_offRef)
    #     return z_acc,ang_vel
    
    def land(self,):
        pass

    def main_loop_cf(self,):
        timeHelper = self.swarm.timeHelper

        while not self.initialized:
            pass
        
        self.init_pos = np.copy(self.state.pos)

        # Logging and plotting
        self.pose_positions = []
        self.cf_positions = []
        self.pose_orientations = []
        self.ref_positions = []
        self.ref_orientation = []
        self.ts = []
        self.thrust_cmds = []
        self.ang_vel_cmds = []

        # Take off and Landing configs
        takeoff_rate = self.config["takeoff_rate"]
        takeoff_height = self.config["takeoff_height"]
        takeoff_time = takeoff_height/takeoff_rate

        landing_time = self.config["landing_height"]/self.config["landing_rate"]
        landing_height = self.config["landing_height"]

        # Print flags
        land_flag = 0
        takeoff_flag = 0
        warmup_flag = 0
        task1_flag = 0
        task2_flag = 0

        # Main loop
        t = 0.0
        startTime = timeHelper.time()
        while not rospy.is_shutdown() and t <30.0:
            self.BB_failsafe()
            t = timeHelper.time() - startTime
            warmup_time = self.config["kalman_warmup"]

            # setting the reference according to a time varying trajectory. 
            # Add the reference trajectory in the set_ref() function

            if t<warmup_time:
                if warmup_time-t<3 and warmup_flag==0:
                    warmup_flag=1
                    print("Taking off in 3 seconds ..... ")
                
                z_acc,ang_vel = 0.0,np.zeros(3)
                pass

            elif t<takeoff_time + warmup_time:
                self.set_takeoff_ref(t-warmup_time,takeoff_height,takeoff_rate)
                if takeoff_flag==0:
                    print("********* TAKEOFF **********")
                    takeoff_flag = 1
                z_acc, ang_vel = self.pid_controller.response(t-warmup_time,self.state,self.ref)
            
            ########################################################
            elif t<takeoff_time + warmup_time + 10.:
                #HOVER
                # Use the reference function here
                self.set_hover_ref(t-warmup_time)
                if task1_flag==0:
                    print("********* TASK PID********")
                    task1_flag = 1
                    offset_pos = self.init_pos+np.array([0.,0.,takeoff_height])

                self.ref.pos +=offset_pos
                z_acc, ang_vel = self.pid_controller.response(t-warmup_time,self.state,self.ref)

            # elif t<takeoff_time+5.+10.:
            #     #HOVER
            #     if task2_flag==0:
            #         # print("********* TASK PPO********")
            #         task2_flag = 1
            #         _ref = self.ref.pos

            #     z_acc, ang_vel = self.ppo_controller.response(t,self.state,self.ref)
            # # #########################################################
                      
            else:
                if land_flag==0:
                    print("********* LAND **********")
                    land_flag = 1
                    land_pos = self.state.pos
                    land_pos[-1] = landing_height
                    land_ref = State_struct(pos=land_pos)

                z_acc, ang_vel = self.pid_controller.response(t-warmup_time,self.state,land_ref)

            # quat = self.state.rot
            # rot = R.from_quat(quat)
            # eulers = rot.as_euler('ZYX', degrees=True)
            self.pose_positions.append(np.copy(self.pose_pos))
            self.pose_orientations.append(self.state.rot.as_euler('ZYX', degrees=True))
            self.cf_positions.append(self.cf.position())
            self.ref_positions.append(self.ref.pos)
            self.ref_orientation.append(self.ref.rot.as_euler('ZYX',degrees=True))
            self.ts.append(t)
            self.thrust_cmds.append(z_acc)
            self.ang_vel_cmds.append(ang_vel * 180 / 2*np.pi)

            self._send2cfClient(self.cf,z_acc, ang_vel)

            timeHelper.sleepForRate(sleepRate)

    def write_to_log(self):
        LOG_DIR = Path().home() / 'Drones' / 'logs'

        self.pose_positions = np.array(self.pose_positions)
        print(self.pose_positions)
        self.pose_orientations = np.array(self.pose_orientations)
        self.cf_positions = np.array(self.cf_positions)
        self.ts = np.array(self.ts)
        self.thrust_cmds = np.array(self.thrust_cmds)
        self.ang_vel_cmds = np.array(self.ang_vel_cmds)
        np.savez(LOG_DIR / self.logfile, 
            pose_positions=self.pose_positions,
            pose_orientations=self.pose_orientations,
            cf_positions=self.cf_positions,
            ang_vel_cmds=self.ang_vel_cmds,
            ts=self.ts,
            thrust_cmds=self.thrust_cmds
        )

    def main_loop_sim(self,):
        i = 0
        t = 0.     
        state = State_struct()   
        while t<30.0:

            self.set_hover_ref(t)
            self.cf.step_angvel_cf(i*self.dt,self.dt,self.pid_controller,self.ref)
            quadsim_state = self.cf.rb.state()
            
            state.pos = quadsim_state.pos
            state.rot = quadsim_state.rot
            
            self.cf.vis.set_state(quadsim_state.pos,quadsim_state.rot)
            time.sleep(self.dt/2)
            t = i*self.dt
            i+=1

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--quadsim', action='store_true')
    parser.add_argument('--config', action='store', type=str, default="cf_config.yaml")
    parser.add_argument('--logfile', action='store', type=str, default='log.npz')
    g = EasyDict(vars(parser.parse_args()))


    x = ctrlCF("cf2",sim=g.quadsim,config_file=g.config,log_file=g.logfile)
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
    

