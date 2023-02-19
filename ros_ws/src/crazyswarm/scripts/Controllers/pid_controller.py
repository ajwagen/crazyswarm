import numpy as np

from scipy.spatial.transform import Rotation as R

# from quadsim.control import Controller
def add2npQueue(array, new):
    array[0:-1] = array[1:]
    array[-1] = new
    return array

class PIDController():
  def __init__(self,isSim):
    super().__init__()
    # self.model = model

    self.isSim = isSim

    self.mass = 0.032
    self.g = 9.8

    # self.kp_pos = 6.0
    # self.kd_pos = 4.0
    # self.ki_pos = 0.0
    # self.kp_rot =   1.0
    # self.yaw_gain = 0.3
    # self.kp_ang =   1e-8

    # self.kp_pos = 2.5
    # self.kd_pos = 3.5
    # self.ki_pos = 0.7
    # self.kp_rot =   70.0/16
    # self.yaw_gain = 130.0/16
    # self.kp_ang =   16

    self.kp_pos = 6.0
    self.kd_pos = 4.0
    self.ki_pos = 1.2
    self.kp_rot =   90.0/16
    self.yaw_gain = 220.0/16
    self.kp_ang =   16

    self.prev_t = None
    self.pos_err_int = np.zeros(3)
    self.I = np.array([[3.144988,4.753588,4.640540],
                       [4.753588,3.151127,4.541223],
                       [4.640540,4.541223,7.058874]])*1e-5

    self.p_err_buffer = np.zeros((50,3))
    self.dt_buffer = np.zeros((50,3))
    self.pos_err_int = 0
  def response(self, t, state, ref ):
    """
        Given a time t and state state, return body z force and torque (body-frame).

        State is defined in rigid_body.py and includes pos, vel, rot, and ang.

        The reference is available using self.ref (defined in flatref.py)
        and contains pos, vel, acc, jerk, snap, yaw, yawvel, and yawacc,
        which are all functions of time.

        self.model contains quadrotor model parameters such as mass, inertia, and gravity.
        See models.py.

        TODO Implement a basic quadrotor controller.
        E.g. you may want to compute position error using something like state.pos - self.ref.pos(t).

        You can test your controller by running main.py.
    """
    if self.prev_t is None:
      dt = 0
    else:
      dt = t - self.prev_t
    self.prev_t = t

    pos = state.pos
    vel = state.vel
    rot = state.rot
    
    # pos = state[0:3]
    # vel = state[3:6]
    # rot = state[6:10]

    p_err = pos - ref.pos
    # print(rot)
    # r = R.from_quat(rot)

    # self.p_err_buffer = add2npQueue(self.p_err_buffer, p_err)
    # self.dt_buffer = add2npQueue(self.dt_buffer, dt)

    # self.pos_err_int = np.sum(self.p_err_buffer*self.dt_buffer)
    self.pos_err_int+=p_err*dt

    acc_des = (np.array([0, 0, self.g]) - self.kp_pos*(p_err) - self.kd_pos*(vel) - self.ki_pos*self.pos_err_int + ref.acc)

    u_des = rot.as_matrix().T.dot(acc_des)

    acc_des = np.linalg.norm(u_des)

    rot_err = np.cross(u_des / acc_des, np.array([0, 0, 1]))

    yaw, _, _ = rot.as_euler('zyx')
    yaw_des = 0.0  # self.ref.yaw(t)

    omega_des = -self.kp_rot * rot_err
    omega_des[2] = -self.yaw_gain*(yaw-yaw_des)

    return acc_des, omega_des