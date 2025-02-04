from path_planning_algorithm import VisibilityRoadMap, VoronoiRoadMap, Dijkstra, generate_nodes_on_line
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseArray
import math
from tf2_ros import Buffer, TransformBroadcaster, TransformListener


def normalize_angle(degrees):
    """Normalize an angle to the range [-180, 180] degrees."""
    normalized = (degrees + 180) % 360 - 180
    # To handle the special case when the result is -180 (we want +180 instead)
    if normalized == -180:
        return 180
    return normalized


def create_transform_matrix(x, y, theta):
    T = np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])
    return T


def invert_transform_2d(rotation_matrix, translation_vector):
    rotation_inverse = rotation_matrix.T
    translation_inverse = -rotation_inverse @ translation_vector
    inverse_transform = np.array([
        [rotation_inverse[0, 0], rotation_inverse[0, 1], translation_inverse[0]],
        [rotation_inverse[1, 0], rotation_inverse[1, 1], translation_inverse[1]],
        [0, 0, 1]
    ])
    return inverse_transform



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



class CarPoseEsimation:

    def __init__(self):

        self.ap_poss = {
        "marker_0": (1.2, 1, -90), 
        "marker_1": (1, 1.2, -180), 
        "marker_2": (1.2, 1.4, 90), 
        "marker_3": (1.4, 1.2, 0), 
        "marker_4": (0, 0.8, 0), 
        "marker_5": (0, 1.6, 0),
        "marker_6": (0.8, 2.4, -90),
        "marker_7": (1.6, 2.4, -90),
        "marker_8": (2.4, 1.6, -180),
        "marker_9": (2.4, 0.8, -180),
        "marker_10": (1.6, 0, 90),
        "marker_11": (0.8, 0, 90),       
        }

    def get_car_world_coordinate_mat(self, pose_info, ap_pos):

        position = pose_info.position
        orientation = pose_info.orientation
        r = R.from_quat( [orientation.x, orientation.y, orientation.z, orientation.w] )
        euler_angles = r.as_euler('xyz', degrees=True)
        pitch, raw_yaw_deg, roll = euler_angles
        yaw_deg = normalize_angle( raw_yaw_deg - 90 )
        yaw_deg = normalize_angle( -1 * yaw_deg - 180 )
        yaw_rad = np.deg2rad(yaw_deg)
        
        # april-tag's x in camera coordinate system
        ap_cam_x = position.x

        # april-tag's y in world coordinate system
        ap_cam_y = position.z

        T_tag2world = create_transform_matrix(ap_pos[0], ap_pos[1], np.deg2rad(ap_pos[2]))

        T_tag2cam = create_transform_matrix(ap_cam_x, ap_cam_y, yaw_rad)

        T_cam2tag = invert_transform_2d(T_tag2cam[:2,:2], [T_tag2cam[0][2],T_tag2cam[1][2]])

        T_cam2world = np.matmul(T_tag2world, T_cam2tag)

        angle_rad = np.pi / 2 + np.deg2rad(ap_pos[2]) - yaw_rad
        angle_deg = normalize_angle(np.rad2deg(angle_rad))
        angle_rad = np.deg2rad(angle_deg)


        return np.matmul(T_cam2world,[[0],[0],[1]])[0].item(), np.matmul(T_cam2world,[[0],[0],[1]])[1].item(), angle_deg, angle_rad



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





def main(args=None):

    modes = ["min_distance", "max_safety"]
    mode = modes[1]

    # generate way points
    if mode == "min_distance":
        vrm = VisibilityRoadMap()
        graph = vrm.planning()
        dijkstra = Dijkstra(graph, vrm.start, vrm.goal)
        path, _ = dijkstra.planning()
        n = 5

        # add more waypoints to make sure it's always on the path
        detailed_path = []
        for idx, (nodei, nodej) in enumerate( zip(path[:-1] , path[1:]) ):
            detailed_path.extend(generate_nodes_on_line(nodei, nodej, n)[:-1])
        detailed_path.append(path[-1])

        waypoints = np.array( [[d.x, d.y] for d in detailed_path] )
    
        angles = []
        for i in range(len(waypoints)):
            temp = np.deg2rad( normalize_angle(90 + i * 20) )
            angles.append(temp)
        angles = np.reshape(angles, (-1, 1))
        waypoints = np.hstack([waypoints, angles])
        
        np.savetxt("min_distance_waypoints.txt", waypoints, fmt='%f')

        print(waypoints)

        filename = "min_distance_current_state_history.txt"

    else:
        vrm = VoronoiRoadMap()
        graph = vrm.planning()
        dijkstra = Dijkstra(graph, vrm.start, vrm.goal)
        path, _ = dijkstra.planning()
        downsamples = path[1:-1][::2]
        path = [ path[0] ] + downsamples + [path[-1]]
        waypoints = np.array( [[d.x, d.y] for d in path] )

        angle_step = 180 / len(waypoints)
        angles = []
        for i in range(len(waypoints)):
            temp = np.deg2rad( normalize_angle(90 + i * angle_step) )
            angles.append(temp)
        angles = np.reshape(angles, (-1, 1))
        waypoints = np.hstack([waypoints, angles])

        filename = "max_safety_current_state_history.txt"
        
        np.savetxt("max_safety_waypoints.txt", waypoints,  fmt='%f')

        print(waypoints)

    
    rclpy.init(args=args)

    # ap detection 
    apdetector = APDetection()

    # init pid controller
    pid = PIDcontroller(0.02,0.005,0.005)

    # initialize current state
    current_state = waypoints[0]

    # current_state_history
    current_state_history = None

    # define car pose estimator
    car_pose_estimator = CarPoseEsimation()


    for wp in waypoints:

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
        
        while(np.linalg.norm(pid.getError(current_state, wp)) > 0.2): 
            
            update_value = pid.update(current_state)
            twist_information = genTwistMsg(coord(update_value, current_state))
            pid.publisher_.publish(twist_information)
            time.sleep(0.05)

            # detect ap-tags
            rclpy.spin_once(apdetector)

            if apdetector.ap_detected:
                # do car estimation here
                for frame_id in apdetector.realtime_frame_id_to_msg:
                    ap_pos = car_pose_estimator.ap_poss[frame_id]
                    pose_info = apdetector.realtime_frame_id_to_msg[frame_id]
                    car_w_x, car_w_y, car_w_omega_deg, car_w_omega_rad = car_pose_estimator.get_car_world_coordinate_mat(pose_info, ap_pos)
                    break
                current_state = np.array([car_w_x, car_w_y, car_w_omega_rad])
            else:
                current_state += update_value


            # record current_state_history
            if current_state_history is None:
                current_state_history = np.copy(current_state)
                current_state_history = np.reshape(current_state_history, (-1, 3))
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
                print("=====================\n")
                print("\n*********************************************************************\n")
                pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))

            counter += 1
    # stop the car and exit
    pid.publisher_.publish(genTwistMsg(np.array([0.0,0.0,0.0])))

    rclpy.shutdown()

    # save matrices
   
    np.savetxt(filename, current_state_history, fmt='%f') 




def static_test(args=None):

    rclpy.init(args=args)

    # ap detection 
    apdetector = APDetection()

    # car position estimation
    car_pose_estimator = CarPoseEsimation()


    while True:
        rclpy.spin_once(apdetector)
        
        if len(apdetector.realtime_frame_id_to_msg) == 0:
            print("No AP detected")
        else:
            for frame_id in apdetector.realtime_frame_id_to_msg:
                ap_pos = car_pose_estimator.ap_poss[frame_id]
                pose_info = apdetector.realtime_frame_id_to_msg[frame_id]
                car_w_x, car_w_y, car_w_omega_deg, car_w_omega_rad = car_pose_estimator.get_car_world_coordinate_mat(pose_info, ap_pos)
                print(f"car_w_x: {car_w_x}")
                print(f"car_w_y: {car_w_y}")
                print(f"car_w_omega_deg: {car_w_omega_deg}")
                print(f"car_w_omega_rad: {car_w_omega_rad}")
                break

        print() ; print()


        apdetector.reset()



    
if __name__ == "__main__":
    main()