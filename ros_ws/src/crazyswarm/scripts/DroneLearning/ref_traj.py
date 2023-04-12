import numpy as np
from cf_utils.rigid_body import State_struct
from scipy.spatial.transform import Rotation as R
from quadsim.learning.refs.random_zigzag import RandomZigzag
import copy

class Trajectories:
    def __init__(self, init_pos=np.array([0.,0.,0.])):
        self.init_pos = init_pos
        self.last_state = State_struct()
        self.curr_state = State_struct()

        self.ret = 0

        self.random_zigzag_obj = RandomZigzag(max_D=np.array([1,0.5,0]), seed=2023)
    
    # ESSENTIAL FUNCTIONS
    # Cubic Polynomial trajectory for Goto
    # Take off and landing use Goto functions 
    
    def make_K_matrix_goto(self, t):
        K = np.array([[12 * t**2, 6 * t, 2],
                    [4 * t**3, 3 * t**2, 2 * t],
                    [t**4, t**3, t**2]])

        return K
    
    def _goto_init(self, final_point, rate = 0.2):
        self.final_point = final_point
        dist = np.linalg.norm((self.last_state.pos - final_point))
        T = dist / rate
        K = self.make_K_matrix_goto(T)

        Kinv = np.linalg.inv(K)

        Bx = np.array([0., -self.last_state.vel[0], final_point[0] - self.last_state.vel[0] * T - self.last_state.pos[0]])
        By = np.array([0., -self.last_state.vel[1], final_point[1] - self.last_state.vel[1] * T - self.last_state.pos[1]])
        Bz = np.array([0., -self.last_state.vel[2], final_point[2] - self.last_state.vel[2] * T - self.last_state.pos[2]])
        self.coeff_x = Kinv.dot(Bx.T)
        self.coeff_y = Kinv.dot(By.T)
        self.coeff_z = Kinv.dot(Bz.T)

        
    def goto(self, t):
        K = self.make_K_matrix_goto(t)
        ref_state_x = K.dot(self.coeff_x)
        ref_state_y = K.dot(self.coeff_y)
        ref_state_z = K.dot(self.coeff_z)
        ref_state_x[2] += self.last_state.vel[0] * t + self.last_state.pos[0]
        ref_state_y[2] += self.last_state.vel[1] * t + self.last_state.pos[1]
        ref_state_z[2] += self.last_state.vel[2] * t + self.last_state.pos[2]

        ref_state_x[1] += self.last_state.vel[0]
        ref_state_y[1] += self.last_state.vel[1]
        ref_state_z[1] += self.last_state.vel[2]

        dist = np.linalg.norm((self.curr_state.pos - self.final_point))
        diff_pos = self.curr_state.pos - self.final_point
        
        if (np.linalg.norm(diff_pos[0:2]) > 0.1 or abs(diff_pos[-1]) > 0.01) and self.ret == 0: 
            ref_pos = [ref_state_x[2], ref_state_y[2], ref_state_z[2]]
            ref_vel = [ref_state_x[1], ref_state_y[1], ref_state_z[1]]
            ref_acc = [ref_state_x[0], ref_state_y[0], ref_state_z[0]]
            ref_jerk = np.array([24 * self.coeff_x[0] * t + 6 * self.coeff_x[1], 24 * self.coeff_y[0] * t + 6 * self.coeff_y[1], 24 * self.coeff_z[0] * t + 6 * self.coeff_z[1]])
            ref_snap = np.array([48 * self.coeff_x[0], 48 * self.coeff_y[0], 48 * self.coeff_z[0]])
            ref = State_struct(ref_pos, ref_vel, ref_acc, ref_jerk, ref_snap)
        else :
            self.ret = 1
            ref = State_struct(pos = copy.deepcopy(self.final_point))
        
        return ref, self.ret
    
    def set_hover_ref(self, t):
        ref_pos = np.array([0., 0.0, 0.0])
        ref_vel = np.array([0., 0., 0])
        ref = State_struct(pos=ref_pos, vel=ref_vel)
        self.ret = 0
        return ref, self.ret
    

    def set_landing_ref(self, t, landing_height, landing_rate):
        ref_pos = self.last_state.pos
        ref_rot = self.last_state.rot
        ref_pos[-1] = max(self.last_state.pos[-1] - landing_rate * t, landing_height)
        ref_vel = np.array([0., 0., 0])
        ref = State_struct(pos=ref_pos, vel=ref_vel, rot=ref_rot)
        self.ret = 0
        return ref, self.ret
    
    # EXEPRIMENT RUN trajectories
    def yaw_rot(self,t):
        rate = 1.2

        ref_pos = np.array([0., 0., 0.])
        ref_vel = np.array([0., 0., 0])
        ref_rot = np.array([min(rate * t, 2 * (np.pi / 3)), 0., 0. ])
        ref = State_struct(pos=ref_pos,vel=ref_vel, rot=R.from_euler("ZYX", ref_rot))

        return ref, self.ret

    def set_circle_ref(self, t):
        radius = 0.5
        center = np.array([-radius, 0., 0.])
        ref_pos = np.array([radius * np.cos(t * 0.6), radius * np.sin(t * 0.6), 0.0]) + center
        ref_vel = np.array([0., 0., 0])
        ref = State_struct(pos=ref_pos, vel=ref_vel)
        self.ret = 0
        return ref, self.ret
   
    def zigzag_traj(self, t, T=1, D=1):
        
        if (t // T) % 2 == 0:
            x = D / T * (t % T)
            # xdot = D / T
        else:
            x = D - (D / T * (t % T))
        
        ref_pos = np.array([x, 0, 0])
        ref = State_struct(pos=ref_pos)

        return ref, ref_pos   

    def zigzag_guanya(self, t):
        p = 2. # period
        t_ = t
        x = 2 * np.abs(t_ / p - np.floor(t_ / p + 0.5))

        ref_pos = np.array([x, 0, 0])
        ref = State_struct(pos=ref_pos)

        return ref, ref_pos

    def random_zigzag(self, t):
        ref_pos = self.random_zigzag_obj.pos(t)
        ref_vel = self.random_zigzag_obj.vel(t)
        ref = State_struct(pos=ref_pos, vel=ref_vel)

        return ref, self.ret




    # LEGACY CODES
    def DONT_USE_set_takeoff_ref(self, t, takeoff_height, takeoff_rate):
        moving_pt = takeoff_rate * t
        ref_pos = self.init_pos + np.array([0., 0., min(moving_pt, takeoff_height)])
        ref_vel = np.array([0., 0., 0])
        ref = State_struct(pos=ref_pos, vel = ref_vel)
        self.ret = 0
        return ref, self.ret

    def DONT_USE_set_takeoff_ref_flat(self, t, takeoff_height, takeoff_rate):
        T = takeoff_height / takeoff_rate
        K = np.array([[12 * T**2, 6 * T, 2],
                      [4 * T**3, 3 * T**2, 2 * T],
                      [T**4, T**3, T**2]])
        Kinv = np.linalg.inv(K)
        coeffs = Kinv.dot(np.array([0., 0., takeoff_height]).T)

        moving_pt =  coeffs[0] * t ** 4 + coeffs[1] * t ** 3 + coeffs[2] * t ** 2
        ref_pos = self.init_pos + np.array([0., 0., min(moving_pt, takeoff_height)])
        if moving_pt < takeoff_height :
            ref_vel  = np.array([0., 0.,  4 * coeffs[0] * t ** 3 + 3 * coeffs[1] * t ** 2 + 2 * coeffs[2] * t])
            ref_acc  = np.array([0., 0., 12 * coeffs[0] * t ** 2 + 6 * coeffs[1] * t      + 2 * coeffs[2]])
            ref_jerk = np.array([0., 0., 24 * coeffs[0] * t      + 6 * coeffs[1]])
            ref_snap = np.array([0., 0., 48 * coeffs[0]])
        else:
            ref_vel = np.zeros(3)
            ref_acc = np.zeros(3)
            ref_jerk = np.zeros(3)
            ref_snap = np.zeros(3)

        ref = State_struct(pos=ref_pos, 
                           vel=ref_vel,
                           acc=ref_acc,
                           jerk=ref_jerk,
                           snap=ref_snap)
        self.ret = 0
        return ref, self.ret

    def DONT_USE_goto_legacy(self, t, final_point):
        rate = 0.3
        final_pts = np.array([0.5, 0.5, 0.5])
        ref_pos = np.array([rate * t, rate * t, rate * t])
        # self.ret = 0
        if np.linalg.norm((final_pts - ref_pos)) < 0.01 or self.ret == 1:
            ref_pos = final_pts
            self.ret = 1

        ref_vel = np.array([0., 0., 0])
        ref = State_struct(pos=ref_pos, vel=ref_vel)
        
        return ref, self.ret