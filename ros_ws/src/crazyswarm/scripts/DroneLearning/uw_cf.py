#!/usr/bin/env python
import signal
import argparse
from easydict import EasyDict
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
import copy

from cf_utils.rigid_body import State_struct
from ref_traj import Trajectories

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

class ctrlCF():
    
    def __init__(self, cfName,sim=False,config_file="experiments/cf_config.yaml", log_file='log.npz', debug=False):
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
            self.dt = 0.003


        with open(config_file,"r") as f:
            self.config = yaml.full_load(f)

        self.set_logging_arrays()
        self.set_tasks()

    ######### E STOPS ################
    def shutdown_callback(self,):
        print("Shutting down ROS!")

    def emergency_handler(self,signum,frame):
        '''
        Intercepts Ctrl+C from the User and stops the motor of the drone.
        '''
        print("User Emergency Stop!")
        self.swarm.allcfs.emergency()
        self.write_to_log()
        exit()
    
    def BB_failsafe(self,):
        '''
        Bounding Box Fail Safe

        If the drone goes outside the bounding box dimensions(as provided in the config file),
        The motors are stopped. The Bounding box is made by taking the `self.init_pos` as the center
        of the bounding box.
        '''
        pos = self.state.pos
        w_bound = self.config["E_BB_width"]
        h_bound = self.config["E_BB_height"]
        if abs(pos[0] - self.init_pos[0]) > w_bound/2 or abs(pos[1] - self.init_pos[1]) > w_bound/2 or pos[2]>h_bound:
            print('Out of Bounding Box EMERGENCY STOP!!')
            self.swarm.allcfs.emergency()
            self.write_to_log()
            exit()
    ##################################

    ##### MAIN LOOP Init functions ########
    def set_logging_arrays(self,):
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

    def set_tasks(self,):
        self.tasks = []
        if self.config["tasks"] is not None:
            self.tasks = self.config["tasks"]

        self.flag= {
                    "takeoff":0,
                    "land":0,
                    "warmup":0,
                    "tasks":[0]*len(self.tasks)}

        self.task_num = -1

    def init_loop_params(self,):
        '''
        Initializing all the parameters required in the loop.

        self.trajs -> instance of Trajectory class which has multiple trajectory functions. 
        
        self.warmup_time -> Wierd issue about kalman filter taking almost 10 secs to warm up. 
                            Setting an offset time to make sure the pose data to the drone 
                            is without any major lag

        self.land_buffer -> Basically stores the last 5 Z positions while landing. When mean 
                            of the Z position is less than a threshold, landing is complete 
                            and drone stops
        '''
        # All tasks are done taking the offset position (point after takeoff) as the origin
        self.trajs = Trajectories(self.init_pos)

        # Making a landing position buffer
        self.land_buffer = deque([0.]*5)
        
        self.takeoff_time = self.config["takeoff_height"]/self.config["takeoff_rate"]
        self.warmup_time = self.config["kalman_warmup"]
        
        if self.debug:
            self.takeoff_time = 100000

        self.land_start_timer = 0. # for landing
        self.tasks_time = 0.  # setting time limit for each task
        self.prev_task_time = 0. # elapsed time just before the task, so tha the task can start with 0. sec 
    #####################################

    ##### State estimation ###########
    def state_callback(self, data):
        '''
        A ros subscriber to `/cfX/pose` message. 
        Numerically calculates the translational velocity of the drone
        '''
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
    #####################################


    def _send2cfClient(self,cf,z_acc,ang_vel):
        '''
        Hacky Way!
        Takes the Z accelerations and angular velocities from the controllers and sends
        these values to the firmware via crazyswarm's `cmdFullState`

        Note: self.ref.pos is sent not for any controls purpose. It is a hacky way of 
              sending reference positions to the firmware for logging.
        '''
        pos = self.ref.pos 
        vel = [0,0,0]
        acc = [0,0,z_acc]
        yaw = 0
        omega = ang_vel.tolist()
        cf.cmdFullState(pos,vel,acc,yaw,omega)

    def land(self,):
        pass

    
    ####### Log all the data at the end #########
    def write_to_log(self):

        if not self.isSim:
            LOG_DIR = Path().home() / 'sda4/drones' / 'crazyswarm' / 'logs'
    
            self.pose_positions = np.array(self.pose_positions)
            self.pose_orientations = np.array(self.pose_orientations)
            self.cf_positions = np.array(self.cf_positions)
            self.ref_orientation = np.array(self.ref_orientation)
            self.ref_positions = np.array(self.ref_positions)
            self.ts = np.array(self.ts)
            self.thrust_cmds = np.array(self.thrust_cmds)
            self.ang_vel_cmds = np.array(self.ang_vel_cmds)
    
            # self.ppo_acc = np.array(self.ppo_acc)
            # self.ppo_ang = np.array(self.ppo_ang)
    
            np.savez(LOG_DIR / self.logfile, 
                pose_positions=self.pose_positions,
                pose_orientations=self.pose_orientations,
                cf_positions=self.cf_positions,
                ref_positions = self.ref_positions,
                ref_orientation = self.ref_orientation,
                ang_vel_cmds=self.ang_vel_cmds,
                ts=self.ts,
                thrust_cmds=self.thrust_cmds,
                # ppo_ang = self.ppo_ang,
                # ppo_acc = self.ppo_acc,
            )
        else:
            LOG_DIR = Path().home() / 'sda4/drones' / 'crazyswarm' / 'sim_logs'
    
            self.pose_positions = np.array(self.pose_positions)
            self.pose_orientations = np.array(self.pose_orientations)
            self.ref_orientation = np.array(self.ref_orientation)
            self.ref_positions = np.array(self.ref_positions)
            self.ts = np.array(self.ts)
            self.thrust_cmds = np.array(self.thrust_cmds)
            self.ang_vel_cmds = np.array(self.ang_vel_cmds)
    
            # self.ppo_acc = np.array(self.ppo_acc)
            # self.ppo_ang = np.array(self.ppo_ang)
    
            np.savez(LOG_DIR / self.logfile, 
                pose_positions=self.pose_positions,
                pose_orientations=self.pose_orientations,
                ref_positions = self.ref_positions,
                ref_orientation = self.ref_orientation,
                ang_vel_cmds=self.ang_vel_cmds,
                ts=self.ts,
                thrust_cmds=self.thrust_cmds,)
    
    def set_refs_from_tasks(self,t,offset_pos):
        '''
        Iterates over the tasks by keeping tabs on the timer. 

        self.tasks is intialized in the `set_tasks()` function. It is basically an array
        of tasks mentioned in the configuration file.

        self.task_num is the index of self.tasks currently being completed by the drone.
        NOTE: take off and landing are not considered in this task
        '''
        # Iterating over the tasks
        if t >= self.takeoff_time+self.tasks_time+self.warmup_time:
            self.task_num+=1

        ###########################################################################
        if t<self.warmup_time:
            if self.warmup_time-t<3 and self.flag["warmup"]==0:
                self.flag["warmup"]=1
                print("Taking off in 3 seconds ..... ")

        ###### Take off Function
        elif t<self.takeoff_time+self.warmup_time:
            if self.flag["takeoff"]==0:
                print("********* TAKEOFF **********")
                self.flag["takeoff"] = 1
            self.ref = self.trajs.set_takeoff_ref(t-self.warmup_time,self.config["takeoff_height"],self.config["takeoff_rate"])

        ###### Tasks
        # Switching to the tasks and getting the reference trajectory positions
        
        elif self.task_num < len(self.tasks):
            if self.flag["tasks"][self.task_num] == 0:
                self.prev_task_time=t
                print( "*****"+self.tasks[self.task_num]["description"]+"*****")

                self.flag["tasks"][self.task_num] = 1 
                self.tasks_time+=self.tasks[self.task_num]["time"]
                
            if t<self.takeoff_time + self.warmup_time + self.tasks_time:
                self.ref = getattr(self.trajs,self.tasks[self.task_num]["ref"])(t-self.prev_task_time)
                self.ref.pos+=offset_pos           
        
        ###### Landing
        else:
            if self.flag["land"]==0:
                self.trajs.last_state = self.state.pos
                print("********* LAND **********")
                self.flag["land"] = 1
            
            self.ref = self.trajs.set_landing_ref(t-self.land_start_timer,self.config["landing_height"],self.config["landing_rate"])                
            self.land_buffer.appendleft(self.state.pos[-1])
            self.land_buffer.pop()
            if np.mean(self.land_buffer) < 0.06:
                print("***** Flight done! ******")
                self.flag['land']=2

    def main_loop_cf(self,):
        timeHelper = self.swarm.timeHelper
        signal.signal(signal.SIGINT, self.emergency_handler)  # Handling Ctrl+C

        try:
            while not self.initialized:
                pass
        except KeyboardInterrupt:
            exit()
        
        # All tasks are done taking the offset position (point after takeoff) as the origin
        self.init_pos = np.copy(self.state.pos)
        self.init_loop_params()
        offset_pos = self.init_pos+np.array([0.,0.,self.config["takeoff_height"]])
        
        t = 0.0
        startTime = timeHelper.time()
        while not rospy.is_shutdown():
            if not self.debug:
                self.BB_failsafe() # Bounding Box Failsafe

            t = timeHelper.time() - startTime
            
            # Setting the refernce trajectory points from the tasks stated in the configuration file
    
            self.set_refs_from_tasks(t,offset_pos)

            ###### Setting the controller for the particular task
            if self.task_num>0 and self.task_num<len(self.tasks):
                controller = getattr(self,self.tasks[self.task_num]["cntrl"])
            else:
                # PID controller for takeoff and landing
                controller = self.pid_controller
                
            # Sending state data to the controller
            z_acc,ang_vel = 0.,np.array([0.,0.,0.])      
            if t>self.warmup_time:
                z_acc,ang_vel = controller.response(t-self.prev_task_time,self.state,self.ref)

            self.pose_positions.append(np.copy(self.pose_pos))
            self.pose_orientations.append(self.state.rot.as_euler('ZYX', degrees=True))
            self.cf_positions.append(self.cf.position())
            self.ref_positions.append(self.ref.pos)
            self.ref_orientation.append(self.ref.rot.as_euler('ZYX',degrees=True))
            self.ts.append(t)
            self.thrust_cmds.append(z_acc)
            self.ang_vel_cmds.append(ang_vel * 180 / (2*np.pi))


            # Setting motors to 0, if landing is complete
            if self.flag["land"]==2:
                z_acc,ang_vel=0.,np.zeros(3)
            if self.flag["land"]==0:
                self.land_start_timer = t

            if self.debug:
                z_acc = 0.0
                ang_vel = np.zeros(3)
                print("cf", self.cf.position(), "pose",self.pose_pos, 'orientation', self.motrack_orientation.as_euler('ZYX', degrees=True), "time",t)

            self._send2cfClient(self.cf, z_acc, ang_vel)

            timeHelper.sleepForRate(sleepRate)

            if self.flag["land"]==2:
                break
    
    
    # Simulation
    def update_sim_states(self,quadsim_state):
        self.state.pos = quadsim_state.pos
        self.state.rot = quadsim_state.rot
        self.state.vel = quadsim_state.vel

    def main_loop_sim(self,):

        # All tasks are done taking the offset position (point after takeoff) as the origin
        self.init_pos = np.array([0.,0.,0.])
        self.init_loop_params()
        offset_pos = self.init_pos+np.array([0.,0.,self.config["takeoff_height"]])

        # Initializing States from the Quadsim Environment
        quadsim_state = self.cf.rb.state()
        self.update_sim_states(quadsim_state)

        i = 0.
        t = 0. 
        while True:

            # Setting the refernce trajectory points from the tasks stated in the configuration file
    
            self.set_refs_from_tasks(t,offset_pos)

            ###### Setting the controller for the particular task
            if self.task_num>0 and self.task_num<len(self.tasks):
                controller = getattr(self,self.tasks[self.task_num]["cntrl"])
            else:
                # PID controller for takeoff and landing
                controller = self.pid_controller

            # Send controller commands to the simulator and simulate the dynamics
            z_acc,ang_vel = 0.,np.array([0.,0.,0.])
            if t>self.warmup_time:
                _,z_acc,ang_vel = self.cf.step_angvel_cf(i*self.dt, self.dt, controller, ref=self.ref)            
            
            # End Flight if landed
            if self.flag["land"]==2:
                z_acc,ang_vel=0.,np.zeros(3)
            if self.flag["land"]==0:
                self.land_start_timer = t

            # Quadsim and State update in the simulation and Visualisation
            quadsim_state = self.cf.rb.state()
            self.update_sim_states(quadsim_state)
            self.cf.vis.set_state(quadsim_state.pos,quadsim_state.rot)
            
            # Simulation timer update
            t = i*self.dt
            i +=1 
            
            # Logging
            self.pose_positions.append(np.copy(self.state.pos))
            self.pose_orientations.append(self.state.rot.as_euler('ZYX', degrees=True))
            self.ref_positions.append(self.ref.pos)
            self.ref_orientation.append(self.ref.rot.as_euler('ZYX',degrees=True))
            self.ts.append(t)
            self.thrust_cmds.append(z_acc)
            self.ang_vel_cmds.append(ang_vel * 180 / (2*np.pi))

            if self.flag["land"]==2:
                break

            # self.ppo_acc.append(z_ppo)
            # self.ppo_ang.append(ang_ppo*180/(2*np.pi))
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--quadsim', action='store_true')
    parser.add_argument('--config', action='store', type=str, default="experiments/cf_config.yaml")
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
    

