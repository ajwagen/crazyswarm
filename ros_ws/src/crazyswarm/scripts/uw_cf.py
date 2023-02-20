#!/usr/bin/env python
import time
import signal
import argparse
from easydict import EasyDict
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
import copy

# import controller 
from Controllers.pid_controller import PIDController
from collections import deque
from Controllers.hover_ppo_controller import PPOController
from Controllers.bc_controller import BCController

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
    def __init__(self, cfName,sim=False,config_file="cf_config.yaml", log_file='log.npz', debug=False):
        self.cfName = cfName
        self.isSim = sim
        self.logfile = log_file
        self.debug = debug

        self.initialized = False
        self.state = State_struct()
        self.prev_state = State_struct()
        self.ref = State_struct()
        # self.state = np.zeros(14)
        # self.prev_state = np.zeros(14)
        self.pid_controller  = PIDController(isSim = self.isSim)
        self.ppo_controller = PPOController(isSim = self.isSim)
        self.ppo_controller.response(0.1, self.state, self.ref, fl=0)
        self.bc_controller = BCController(isSim=self.isSim)
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

        # self.swarm.allcfs.emergency()
        print("Shutting down ROS!")
    def emergency_handler(self,signum,frame):
        print("User Emergency Stop!")
        self.swarm.allcfs.emergency()
        self.write_to_log()
        exit()

    def set_hover_ref(self,t):
        ref_pos = np.array([0.,0.0,0.0])
        ref_vel = np.array([0.,0.,0])
        self.ref = State_struct(pos=ref_pos,vel = ref_vel)
    
    def set_takeoff_ref(self,t,takeoff_height,takeoff_rate):
        # print('debig', t, takeoff_rate, takeoff_rate*t,takeoff_height)
        moving_pt = takeoff_rate*t
        ref_pos = self.init_pos+np.array([0.,0.,min(moving_pt,takeoff_height)])
        ref_vel = np.array([0.,0.,0])
        self.ref = State_struct(pos=ref_pos,vel = ref_vel)

    def set_landing_ref(self,t,landing_height,landing_rate):
        # print('debig', t, landing_rate, landing_rate*t,landing_height)
        ref_pos = self.last_state
        ref_pos[-1] = max(self.last_state[-1] - landing_rate*t,landing_height)
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
        self.motrack_orientation = self.cf.orientation()
        self.motrack_orientation = R.from_quat(self.motrack_orientation)
        rot = np.array([rot.x,rot.y,rot.z,rot.w])
        # self.cf_orientation = R>from_quat()

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

        try:
            while not self.initialized:
                pass
        except KeyboardInterrupt:
            exit()
        print('Initialized...')
        
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

        self.ppo_acc = []
        self.ppo_ang = []


        # Take off and Landing configs
        takeoff_rate = self.config["takeoff_rate"]
        takeoff_height = self.config["takeoff_height"]
        takeoff_time = takeoff_height/takeoff_rate

        landing_rate = self.config["landing_rate"]
        # landing_time = self.config["landing_height"]/self.config["landing_rate"]
        landing_height = self.config["landing_height"]
        land_buffer = deque([0.]*5)

        # Print flags
        land_flag = 0
        takeoff_flag = 0
        warmup_flag = 0
        task1_flag = 0
        task2_flag = 0
        task1_time = 8.0
        task2_time = 10.0

        if self.debug:
            takeoff_time = 100000

        # Main loop
        t = 0.0
        startTime = timeHelper.time()
        signal.signal(signal.SIGINT, self.emergency_handler)
        while not rospy.is_shutdown():
            if not self.debug:
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
                z_ppo,ang_ppo=0.0,np.zeros(3)
                pass

            elif t<takeoff_time + warmup_time:
                self.set_takeoff_ref(t-warmup_time,takeoff_height,takeoff_rate)
                if takeoff_flag==0:
                    print("********* TAKEOFF **********")
                    takeoff_flag = 1
                z_acc, ang_vel = self.ppo_controller.response(t-warmup_time,self.state,self.ref)
                z_ppo, ang_ppo = self.bc_controller.response(t-warmup_time,self.state,self.ref)

            ########################################################
            elif t<takeoff_time + warmup_time + task1_time:
                #HOVER
                # Use the reference function here
                self.set_hover_ref(t-warmup_time)
                if task1_flag==0:
                    print("********* TASK PID********")
                    task1_flag = 1
                    offset_pos = self.init_pos+np.array([0.,0.,takeoff_height])

                self.ref.pos += offset_pos
                z_acc, ang_vel = self.ppo_controller.response(t-warmup_time-takeoff_time,self.state,self.ref)
                z_ppo, ang_ppo = self.ppo_controller.response(t-warmup_time-takeoff_time,self.state,self.ref)
                # print("pid_acc: ",z_acc,"pid_ang: ",ang_vel)
                # print("ppo_acc: ",z_ppo,"ppo_ang: ",ang_ppo, "\n")

            # elif t<takeoff_time+ warmup_time+task1_time+task2_time:
            #     #HOVER
            #     if task2_flag==0:
            #         print("********* TASK PPO********")
            #         task2_flag = 1
            #         _ref = self.ref.pos

            #     z_acc, ang_vel = self.bc_controller.response(t-warmup_time, self.state, self.ref)
            #     print('z_cmd', z_acc, 'ang', ang_vel, t)
            #     ang_vel[0] = ang_vel[0]/3
            #     ang_vel[1] = ang_vel[1]/3
            #     # z_acc[1]
            #     z_pid, ang_pid = self.pid_controller.response(t-warmup_time,self.state,self.ref)
            #     print('pid z cmd', z_pid, 'pid ang', ang_pid)


            # # #########################################################
                      
            else:
                if land_flag==0:
                    self.last_state = self.state.pos
                    land_time = t
                    print("********* LAND **********")
                    land_flag = 1

                self.set_landing_ref(t-land_time,landing_height,landing_rate)
                z_acc, ang_vel = self.pid_controller.response(t-warmup_time,self.state,self.ref)
                z_ppo, ang_ppo = self.bc_controller.response(t-warmup_time,self.state,self.ref)
                
                land_buffer.appendleft(self.state.pos[-1])
                land_buffer.pop()
                if np.mean(land_buffer) < 0.06:
                    "***** Flight done! *********"
                    land_flag=2

            self.pose_positions.append(np.copy(self.pose_pos))
            self.pose_orientations.append(self.state.rot.as_euler('ZYX', degrees=True))
            self.cf_positions.append(self.cf.position())
            self.ref_positions.append(self.ref.pos)
            self.ref_orientation.append(self.ref.rot.as_euler('ZYX',degrees=True))
            self.ts.append(t)
            self.thrust_cmds.append(z_acc)
            self.ang_vel_cmds.append(ang_vel * 180 / (2*np.pi))

            self.ppo_acc.append(z_ppo)
            self.ppo_ang.append(ang_ppo*180/(2*np.pi))

            # print("pid_acc: ",z_acc,"pid_ang: ",ang_vel)
            # print("ppo_acc: ",z_ppo,"ppo_ang: ",ang_ppo, "\n")

            if land_flag==2:
                z_acc,ang_vel=0.,np.zeros(3)
            if self.debug:
                z_acc = 0.0
                # z_acc = 0.3*np.sin(t) + 0.3
                ang_vel = np.zeros(3)
                print("cf", self.cf.position(), "pose", self.pose_pos, "time",t)

            # ang_vel[0] = np.copy(ang_ppo[0]*0.35)
            self._send2cfClient(self.cf, z_acc, ang_vel)

            timeHelper.sleepForRate(sleepRate)

            if land_flag==2:
                break

    def write_to_log(self):
        LOG_DIR = Path().home() / 'Drones' / 'crazyswarm_new' / 'logs'

        self.pose_positions = np.array(self.pose_positions)
        self.pose_orientations = np.array(self.pose_orientations)
        self.cf_positions = np.array(self.cf_positions)
        self.ref_orientation = np.array(self.ref_orientation)
        self.ref_positions = np.array(self.ref_positions)
        self.ts = np.array(self.ts)
        self.thrust_cmds = np.array(self.thrust_cmds)
        self.ang_vel_cmds = np.array(self.ang_vel_cmds)

        self.ppo_acc = np.array(self.ppo_acc)
        self.ppo_ang = np.array(self.ppo_ang)

        np.savez(LOG_DIR / self.logfile, 
            pose_positions=self.pose_positions,
            pose_orientations=self.pose_orientations,
            cf_positions=self.cf_positions,
            ref_positions = self.ref_positions,
            ref_orientation = self.ref_orientation,
            ang_vel_cmds=self.ang_vel_cmds,
            ts=self.ts,
            thrust_cmds=self.thrust_cmds,
            ppo_ang = self.ppo_ang,
            ppo_acc = self.ppo_acc,
        )

    def main_loop_sim(self,):
        i = 0
        t = 0.     
        state = State_struct()   
        p = np.array([-0.01638478, -0.0758579, -0.01579895])
        r = R.from_euler('ZYX', np.array([-1.10916878,  0.63578264,  0.15242415]), degrees=True)
        state.pos = p
        state.rot = r
        ref = State_struct()
        # self.cf.rb.pos = p
        # self.cf.rb.quat = np.hstack((r.as_quat()[-1], r.as_quat()[0:3]))

        print("PID RESPONSE", self.pid_controller.response(0.0, state, ref))

        while t<30.0:

            self.set_hover_ref(t)
            self.cf.step_angvel_cf(i*self.dt, self.dt, self.bc_controller, ref=self.ref)
            quadsim_state = self.cf.rb.state()
            
            state.pos = quadsim_state.pos
            state.rot = quadsim_state.rot
            
            self.cf.vis.set_state(quadsim_state.pos,quadsim_state.rot)
            time.sleep(self.dt/2)
            t = i*self.dt
            i += 1

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--quadsim', action='store_true')
    parser.add_argument('--config', action='store', type=str, default="cf_config.yaml")
    parser.add_argument('--logfile', action='store', type=str, default='log.npz')
    parser.add_argument('--debug', action='store', type=bool, default=False)
    g = EasyDict(vars(parser.parse_args()))

    x = ctrlCF("cf2",sim=g.quadsim,config_file=g.config,log_file=g.logfile, debug=g.debug)

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
    

