import numpy as np
from cf_utils.rigid_body import State_struct
from scipy.spatial.transform import Rotation as R


class Trajectories:
    def __init__(self, init_pos=np.array([0.,0.,0.])):
        self.init_pos = init_pos
        self.last_state = State_struct()

        self.ret = 0
        
    def yaw_rot(self,t):
        rate = 1.2

        ref_pos = np.array([0.,0.,0.])
        ref_vel = np.array([0.,0.,0])
        ref_rot = np.array([min(rate*t, 2*(np.pi/3)),0.,0. ])
        ref = State_struct(pos=ref_pos,vel = ref_vel, rot = R.from_euler("ZYX",ref_rot))

        return ref, self.ret

    def set_hover_ref(self,t):
        ref_pos = np.array([0.,0.0,0.0])
        ref_vel = np.array([0.,0.,0])
        ref = State_struct(pos=ref_pos,vel = ref_vel)
        self.ret = 0
        return ref, self.ret
        
    def set_takeoff_ref(self,t,takeoff_height,takeoff_rate):
        moving_pt = takeoff_rate*t
        ref_pos = self.init_pos+np.array([0.,0.,min(moving_pt,takeoff_height)])
        ref_vel = np.array([0.,0.,0])
        ref = State_struct(pos=ref_pos,vel = ref_vel)
        self.ret = 0
        return ref, self.ret

    def set_landing_ref(self,t,landing_height,landing_rate):
        ref_pos = self.last_state.pos
        ref_rot= self.last_state.rot
        ref_pos[-1] = max(self.last_state.pos[-1] - landing_rate*t,landing_height)
        ref_vel = np.array([0.,0.,0])
        ref = State_struct(pos=ref_pos,vel = ref_vel, rot=ref_rot)
        self.ret = 0
        return ref, self.ret

    def set_circle_ref(self,t):
        radius = 0.5
        center = np.array([-radius,0.,0.])
        ref_pos = np.array([radius*np.cos(t*0.6),radius*np.sin(t*0.6),0.0]) + center
        ref_vel = np.array([0.,0.,0])
        ref = State_struct(pos=ref_pos,vel = ref_vel)
        self.ret = 0
        return ref, self.ret

    def goto(self,t):
        rate = 0.3
        final_pts = np.array([0.5,0.5,0.5])
        ref_pos = np.array([rate*t,rate*t,rate*t])
        # self.ret = 0
        if np.linalg.norm((final_pts-ref_pos))<0.01 or self.ret==1:
            ref_pos = final_pts
            self.ret = 1

        ref_vel = np.array([0.,0.,0])
        ref = State_struct(pos=ref_pos,vel = ref_vel)
        
        return ref, self.ret
    
    def zigzag_traj(self, t, T=1, D=1):
        if (t // T) % 2 == 0:
            x = D / T * (t % T)
        else:
            x = D - (D / T * (t % T))
        
        ref_pos = np.array([x,0,0])
        ref = State_struct(pos = ref_pos)

        return ref, ref_pos        

        