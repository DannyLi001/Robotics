#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseArray
import numpy as np
import math
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from rclpy.time import Duration
import math
from math import copysign, fabs, sqrt, pi, sin, cos, asin, acos, atan2, exp, log
import os


# define parameters

# square length of square
SQ_LEN = 1.7

# landmark square length
LM_LEN = 2.4

# how many states for the car
STATE_SIZE = 3

# how many states for the landmarks
LM_SIZE = 2

# time 
DT = 0.1

# Noise for prediction
RT = np.diag([100 * 2.58e-3, 1500 * 1.57e-04, 100 * 2.06e-3]) 
# RT = np.diag([0.0, 0.0, 0.0]) 

# Noise for calculating K
QT = np.diag([10000 * 2.53e-05, 1500 * 0.000117])


# frame id and observation mapping
frame_id_to_observation_id = {}
observation_id_to_frame_id = {}



class PIDcontroller(Node):
    def __init__(self, Kp, Ki, Kd):
        super().__init__('PID_Controller_NodePub')
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0,0.0,0.0])
        self.lastError = np.array([0.0,0.0,0.0])
        self.timestep = 0.1
        self.maximumValue = 0.02
        self.publisher_ = self.create_publisher(Twist, '/twist', 10)

    def setTarget(self, target):
        """
        Set the target pose.
        """
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.target = np.array(target)

    def getError(self, currentState, targetState):
        """
        Return the difference between two states.
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result

    def setMaximumUpdate(self, mv):
        """
        Set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState):
        """
        Calculate the update value on the state based on the error between current state and target state with PID.
        """
        e = self.getError(currentState, self.target)
        P = self.Kp * e
        self.I += self.Ki * e * self.timestep
        D = self.Kd * (e - self.lastError)
        result = P + self.I + D
        self.lastError = e

        # Scale down the twist if its norm is more than the maximum value
        resultNorm = np.linalg.norm(result)
        if resultNorm > self.maximumValue:
            result = (result / resultNorm) * self.maximumValue
            self.I = 0.0

        return result



# detect april tags
class APDetection(Node):
    def __init__(self):
        super().__init__('robot_state_estimator')
        self.subscription = self.create_subscription(
            PoseArray,
            '/april_poses',
            self.april_pose_callback,
            10)
        self.subscription

        self.ap_detected = False

        # store the ap frame ids
        self.realtime_frame_ids = set()

        # map from frame id to the corresponding message
        self.realtime_frame_id_to_msg = {}

    

    def april_pose_callback(self, msg):
        # Log the number of poses in the PoseArray
        self.ap_detected = False
        self.get_logger().info(f"Received PoseArray with {len(msg.poses)} poses")
        if len(msg.poses) < 1:
            return

        pose_ids = msg.header.frame_id.split(',')[:-1]

        self.ap_detected = True
        for idx in range(len(pose_ids)):
            self.realtime_frame_ids.add(pose_ids[idx])
            self.realtime_frame_id_to_msg[pose_ids[idx]] = msg.poses[idx]

        
        """
        How to retrive

        self.realtime_frame_id_to_msg[pose_id].position.x
        self.realtime_frame_id_to_msg[pose_id].position.y
        self.realtime_frame_id_to_msg[pose_id].position.z

        self.realtime_frame_id_to_msg[pose_id].orientation.x
        self.realtime_frame_id_to_msg[pose_id].orientation.y
        self.realtime_frame_id_to_msg[pose_id].orientation.z
        self.realtime_frame_id_to_msg[pose_id].orientation.w
        """

    def reset(self):
        self.ap_detected = False
        self.realtime_frame_ids = set()
        self.realtime_frame_id_to_msg = {}



def divide_line_into_three_equal_parts(point1, point2):
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    point_a = p1 + (p2 - p1) / 3
    point_b = p1 + 2 * (p2 - p1) / 3
    
    return tuple(point_a), tuple(point_b)


def generate_lm_coordinates():
    lm_len = LM_LEN
    sq_len = SQ_LEN

    left_bottom = ( 0 - ((lm_len - sq_len)/2) ,  0 - ((lm_len - sq_len)/2) )
    right_bottom = ( sq_len + ((lm_len - sq_len)/2) ,  0 - ((lm_len - sq_len)/2) )
    left_top = ( 0 - ((lm_len - sq_len)/2) , sq_len + ((lm_len - sq_len)/2) )
    right_top = ( sq_len + ((lm_len - sq_len)/2) , sq_len + ((lm_len - sq_len)/2) )

    top_point1, top_point2 = divide_line_into_three_equal_parts(left_top, right_top)
    left_point1, left_point2 = divide_line_into_three_equal_parts(left_top, left_bottom)
    bottom_point1, bottom_point2 = divide_line_into_three_equal_parts(left_bottom, right_bottom)
    right_point1, right_point2 = divide_line_into_three_equal_parts(right_top, right_bottom)

    return top_point1, top_point2, left_point1, left_point2, bottom_point1, bottom_point2, right_point1, right_point2


def generate_square_waypoints(run_times):
    arr = np.array([[0, 0, 0], [SQ_LEN, 0, np.pi/2], [SQ_LEN, SQ_LEN, np.pi], [0, SQ_LEN, -1 * np.pi/2]])
    arr = np.vstack([arr] * run_times)
    arr = np.vstack([arr, arr[0]])
    return arr



def calculate_regular_octagon_vertices(L):
    d = L / (2 + np.sqrt(2))
    
    vertices = [
        (d, 0),
        (L - d, 0),
        (L, d),
        (L, L - d),
        (L - d, L),
        (d, L),
        (0, L - d),
        (0, d)
    ]
    
    return vertices



def generate_octagon_waypoints(run_times):
    octagon_vertics = calculate_regular_octagon_vertices(SQ_LEN)
    angles = np.reshape( np.array([np.pi/2, np.pi/2, np.pi, np.pi, -1 * np.pi/2, -1 * np.pi/2, 0, 0]) , (-1, 1))
    octagon_vertics =  np.hstack([octagon_vertics, angles])
    octagon_vertics = np.vstack([octagon_vertics] * run_times)
    octagon_vertics = np.vstack([octagon_vertics, octagon_vertics[0]])
    return octagon_vertics

    

# car control
def genTwistMsg(desired_twist):
    """
    Convert the twist to twist msg.
    """
    twist_msg = Twist()
    twist_msg.linear.x = desired_twist[0] 
    twist_msg.linear.y = desired_twist[1] 
    twist_msg.linear.z = 0.0
    twist_msg.angular.x = 0.0
    twist_msg.angular.y = 0.0
    twist_msg.angular.z = desired_twist[2]
    return twist_msg


def coord(twist, current_state):
    J = np.array([[np.cos(current_state[2]), np.sin(current_state[2]), 0.0],
                  [-np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
                  [0.0,0.0,1.0]])
    return np.dot(J, twist)






# ========== EKF SLAM FUNCTIONS ==========
# Refered PythonRobotics at this part.
# https://atsushisakai.github.io/PythonRobotics/modules/slam/ekf_slam/ekf_slam.html#google_vignette


# x: whole state vector
# u control command (v_x, v_y, w_z)
# return Fx and Gt used for Sigma_t
# Since we add another dimension of control command, we need to change the jacobian matrix
def jacob_motion(x, u):
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
        (STATE_SIZE, LM_SIZE * calc_n_lm(x))))) 
 
    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0]) - DT * u[1, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0]) - DT * u[1, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)
    
    G = np.eye(STATE_SIZE) + (Fx.T @ jF @ Fx)
    return G, Fx,



# given all the state vector, calculate how many landmarks are there
def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n



# x: car's (x, y, theta)
# u: control command: (v_x, v_y, w_z)
# return the estimated position after the control command
def motion_model(x, u):

    # we do not need to change state vector
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])


    # B matrix describes to control command can change the car's position
    B = np.array([[DT * math.cos(x[2, 0]), -DT * math.sin(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), DT * math.cos(x[2, 0]), 0],
                  [0.0, 0.0, DT]])

    x = np.dot(F, x) + np.dot(B, u)

    return x




# x: car's position
# z: observation
# return estimated lm position based on car's position and observation
def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))
    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])
    return zp



# x: whole state vector
# ind: observation id
# return the corresponding landmark position
def get_landmark_position_from_state(x, ind):
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]
    return lm



# lm: estimated landmark position
# xEst: whole state vector
# PEst: covariance matrix
# z: observation of lm
# LMid: observation id
def calc_innovation(lm, xEst, PEst, z, LMid):
    
    # according to lm estimation, what's position difference between car and lm
    delta = lm - xEst[0:2]  
    q = np.dot(delta.T , delta)[0, 0] 

    # according to lm estimation, what's angle difference between car and lm
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0] 
    
    # put position and angle difference together
    zp = np.array([[math.sqrt(q), angle_mod(z_angle)]]) 

    # what's the difference between lm observation and lm estimation
    # this is the same as z - zbar
    y = (z - zp).T

    # normalize angle
    y[1] = angle_mod(y[1]) 

    # calculate H matrix
    H = jacob_h(q, delta, xEst, LMid + 1)
    
    return y, H


# this function is used for calculate H matrix
# q and delta: related to position difference between lm estimation and lm observation
# x: while state vector
# i: observation id: !! starting from 0
# return H matrix
def jacob_h(q, delta, x, i):         
    sq = math.sqrt(q)

    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

    G = G / q
    
    nLM = calc_n_lm(x)
    
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

    F = np.vstack((F1, F2))

    H = np.dot(G, F)

    return H



# normalize angle
def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    Examples
    --------
    >>> angle_mod(-4.0)
    2.28318531

    >>> angle_mod([-4.0])
    np.array(2.28318531)

    >>> angle_mod([-150.0, 190.0, 350], degree=True)
    array([-150., -170.,  -10.])

    >>> angle_mod(-60.0, zero_2_2pi=True, degree=True)
    array([300.])

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


# from camera reading generate observation array
def process_camera_reading(msg):
    dx = -1 * msg.position.x
    dy = msg.position.z
    dist = np.hypot(dx, dy)
    angle = math.atan2(dx, dy)
    return np.array([dist, angle])




def ekf_slam(xEst, PEst, u, detected_frame_id_and_msgs): 
    global frame_id_to_observation_id 
    global observation_id_to_frame_id
    
    # Predict
    G, Fx = jacob_motion(xEst[0 : STATE_SIZE], u)
    xEst[0 : STATE_SIZE] = motion_model(xEst[0 : STATE_SIZE], u)
    PEst[0 : STATE_SIZE, 0 : STATE_SIZE] = (G.T @ PEst[0 : STATE_SIZE, 0 : STATE_SIZE] @ G) + (Fx.T @ RT @ Fx)
    initP = np.eye(2)
   
    # Update
    for frame_id in detected_frame_id_and_msgs:
        # print(f"observed frame id: {frame_id}")
        
        # process observations
        processed_observation = process_camera_reading(detected_frame_id_and_msgs[frame_id])
            
        # handle newly detected id
        if frame_id not in frame_id_to_observation_id:
            # print("Detected New LM")

            # generate observation id
            new_observation_id = len(frame_id_to_observation_id)

            # complete the mapping between frame id and observation id
            frame_id_to_observation_id[frame_id] = new_observation_id
            observation_id_to_frame_id[new_observation_id] = frame_id
            assert len(frame_id_to_observation_id) == len(observation_id_to_frame_id)
            
            # increase the size of xEst and PEST
            xAug = np.vstack( (xEst, calc_landmark_position(xEst, processed_observation)) )   
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
            xEst = xAug
            PEst = PAug
        
        # get the landmark estimation from xEst
        lm = get_landmark_position_from_state(xEst, frame_id_to_observation_id[frame_id])

        # calculate Jacobian H
        y, H = calc_innovation(lm, xEst, PEst, processed_observation, frame_id_to_observation_id[frame_id])
        
        # calculate gain
        K = PEst @ H.T @ np.linalg.inv( (H @ PEst @ H.T) + QT )
        
        # update xEst
        xEst = xEst + (K @ y)

        # update PEst
        PEst = ( np.eye(len(xEst)) - (  K @ H  ) ) @ PEst
    
    # normalize angle
    xEst[2] = angle_mod(xEst[2])

    return xEst, PEst




def main(args=None):
    
    time.sleep(3)

    folder_path = "octagon2"

    # generate way points
    waypoint = generate_octagon_waypoints(2)
    
    
    rclpy.init(args=args)

    # ap detection 
    apdetector = APDetection()

    # init pid controller
    pid = PIDcontroller(0.02,0.005,0.005)

    # init xEst and PEst
    xEst = np.zeros((STATE_SIZE, 1)) 
    PEst = np.eye(STATE_SIZE)    

    # init current_state
    current_state = np.array([0.0,0.0,0.0])
    
    # current_state_history
    current_state_history = None

    for wp in waypoint:

        print("move to way point", wp)
        time.sleep(0.02)
        
        # set wp as the target point
        pid.setTarget(wp)
        
        update_value = pid.update(current_state)
        twist_information = genTwistMsg(coord(update_value, current_state))
        pid.publisher_.publish(twist_information)
        time.sleep(0.05)

        # update the current state
        current_state += update_value
        print(current_state)
        print("=======================================")
        counter = 0
        
        while(np.linalg.norm(pid.getError(current_state, wp)) > 0.15): 
            
            update_value = pid.update(current_state)
            twist_information = genTwistMsg(coord(update_value, current_state))
            pid.publisher_.publish(twist_information)
            time.sleep(0.05)

            # detect ap-tags
            rclpy.spin_once(apdetector)

            if apdetector.ap_detected:
                # do slam here
                control_command =  np.array([twist_information.linear.x, twist_information.linear.y, twist_information.angular.z]) / DT
                control_command = np.reshape(control_command, (-1, 1))
                xEst, PEst = ekf_slam(xEst, PEst, control_command, apdetector.realtime_frame_id_to_msg)
                current_state = xEst[0:STATE_SIZE, :]
                current_state = current_state.flatten()
            else:
                current_state += update_value


            # record current_state_history
            if current_state_history is None:
                current_state_history = np.copy(current_state)
                current_state_history = np.reshape(current_state_history, (-1, STATE_SIZE))
            else:
                current_state_history = np.vstack( (current_state_history, current_state) )


            apdetector.reset()

            if counter % 20 == 0:

                print("\n=====================")
                print(f"update_value: {update_value}")
                print(f"current_state: {current_state}")
                print(f"coord result: {coord(update_value, current_state)}")
                print(f"twist x: {twist_information.linear.x}")
                print(f"twist y: {twist_information.linear.y}")
                print(f"twist z: {twist_information.angular.z}")
                print(f"Error Norm: { np.linalg.norm(pid.getError(current_state, wp)) }")

                for frame_id in frame_id_to_observation_id:
                    print(frame_id)
                    print(xEst[STATE_SIZE + LM_SIZE * frame_id_to_observation_id[frame_id] : STATE_SIZE + LM_SIZE * (frame_id_to_observation_id[frame_id] + 1), :  ])
                    print()

                print("=====================\n")
                print("\n*********************************************************************\n")
                pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))
                # time.sleep(2)

            counter += 1
    # stop the car and exit
    pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))

    rclpy.shutdown()


    print(f"xEst\n{xEst}\n")
    print(f"PEst\n{PEst}\n")


    # save matrices
    np.savetxt(os.path.join(folder_path, 'xEst.txt'), xEst, fmt='%f') 
    np.savetxt(os.path.join(folder_path, 'PEst.txt'), PEst, fmt='%f') 
    np.savetxt(os.path.join(folder_path, 'current_state_history.txt'), current_state_history, fmt='%f') 

    with open(os.path.join(folder_path, "frame_id_to_observation_id.txt"), 'w') as file:
        for key, value in frame_id_to_observation_id.items():
            file.write(f"{key}: {value}\n")

    with open(os.path.join(folder_path, "observation_id_to_frame_id.txt"), 'w') as file:
        for key, value in observation_id_to_frame_id.items():
            file.write(f"{key}: {value}\n")


    
if __name__ == "__main__":
    main()