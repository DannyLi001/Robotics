
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseArray
import numpy as np
import math
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from scipy.spatial.transform import Rotation as R



def normalize_angle(degrees):
    """Normalize an angle to the range [-180, 180] degrees."""
    normalized = (degrees + 180) % 360 - 180
    # To handle the special case when the result is -180 (we want +180 instead)
    if normalized == -180:
        return 180
    return normalized


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

        
        for frame_id in self.realtime_frame_id_to_msg:
            print(frame_id)
            print(f"x value: {self.realtime_frame_id_to_msg[frame_id].position.x}" )
            print(f"z value: {self.realtime_frame_id_to_msg[frame_id].position.z}" )

            orientation = self.realtime_frame_id_to_msg[frame_id].orientation

            r = R.from_quat( [orientation.x, orientation.y, orientation.z, orientation.w] )
            euler_angles = r.as_euler('xyz', degrees=True)
            pitch, raw_yaw_deg, roll = euler_angles
            yaw_deg = normalize_angle( raw_yaw_deg - 90 )
            yaw_deg = normalize_angle( -1 * yaw_deg - 180 )

            yaw_rad = np.deg2rad(yaw_deg)

            print(f"yaw_rad: {yaw_rad}")
            print(f"yaw_deg: {yaw_deg}")

            break
        
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






def main(args=None):

    rclpy.init(args=args)

    # ap detection 
    apdetector = APDetection()

    while True:
        rclpy.spin_once(apdetector)
        apdetector.reset()



    
if __name__ == "__main__":
    main()