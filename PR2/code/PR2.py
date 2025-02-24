import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def load_sensor_data(dataset):
    sensor_data = {}
    
    with np.load(f"./data/Encoders{dataset}.npz") as data:
        sensor_data["encoder_counts"] = data["counts"]  # 4 x n encoder counts
        sensor_data["encoder_stamps"] = data["time_stamps"]  # encoder time stamps
    
    with np.load(f"./data/Hokuyo{dataset}.npz") as data:
        sensor_data["lidar_angle_min"] = data["angle_min"]  # start angle of the scan [rad]
        sensor_data["lidar_angle_max"] = data["angle_max"]  # end angle of the scan [rad]
        sensor_data["lidar_angle_increment"] = data["angle_increment"]  # angular distance between measurements [rad]
        sensor_data["lidar_range_min"] = data["range_min"]  # minimum range value [m]
        sensor_data["lidar_range_max"] = data["range_max"]  # maximum range value [m]
        sensor_data["lidar_ranges"] = data["ranges"]  # range data [m]
        sensor_data["lidar_stamps"] = data["time_stamps"]  # acquisition times of the lidar scans
    
    with np.load(f"./data/Imu{dataset}.npz") as data:
        sensor_data["imu_angular_velocity"] = data["angular_velocity"]  # angular velocity in rad/sec
        sensor_data["imu_linear_acceleration"] = data["linear_acceleration"]  # accelerations in gs
        sensor_data["imu_stamps"] = data["time_stamps"]  # acquisition times of the imu measurements
    
    with np.load(f"./data/Kinect{dataset}.npz") as data:
        sensor_data["disp_stamps"] = data["disparity_time_stamps"]  # acquisition times of the disparity images
        sensor_data["rgb_stamps"] = data["rgb_time_stamps"]  # acquisition times of the rgb images
    
    return sensor_data

def synchronize_data(data1, stamp1, data2, stamp2):
    if stamp1.size < stamp2.size:
        indices = find_closest_indices(stamp1, stamp2)
        return data1, stamp1, data2[:, indices], stamp2[indices]
    else:
        indices = find_closest_indices(stamp2, stamp1)
        return data1[indices], stamp1[indices], data2, stamp2

def find_closest_indices(stamp1, stamp2):
    indices = []
    for t1 in stamp1:
        idx = np.searchsorted(stamp2, t1)
        if idx == 0:
            best_idx = 0
        elif idx == len(stamp2):
            best_idx = len(stamp2) - 1
        else:
            left_diff = t1 - stamp2[idx-1]
            right_diff = stamp2[idx] - t1
            best_idx = idx-1 if left_diff < right_diff else idx
        indices.append(best_idx)
    return np.array(indices)

class OdometryTracker:
    def __init__(self):
        # Constants
        self.ticks_to_meters = 0.0022  # Meters per encoder tick
        self.dt = 1/40.0  # 40Hz sampling rate
        self.wheelbase = 0.5  # Assume 0.5m (adjust based on actual robot)
        
        # Initial pose (x, y, theta)
        self.pose = np.array([0.0, 0.0, 0.0])  # Start at identity pose
        
        # Trajectory history
        self.trajectory = [self.pose.copy()]
    
    def update_pose(self, encoder_counts, imu_yaw_rate):
        """
        Update robot pose using encoder and IMU data
        :param encoder_counts: [FR, FL, RR, RL] encoder ticks
        :param imu_yaw_rate: Yaw rate from IMU (rad/s)
        """
        # Parse encoder counts
        fr, fl, rr, rl = encoder_counts
        
        # Calculate distances for left and right sides
        right_ticks = (fr + rr) / 2.0
        left_ticks = (fl + rl) / 2.0
        
        right_distance = right_ticks * self.ticks_to_meters
        left_distance = left_ticks * self.ticks_to_meters
        
        # Calculate linear velocity (average of both sides)
        linear_velocity = (right_distance + left_distance) / (2.0 * self.dt)
        
        # Use IMU yaw rate for angular velocity
        angular_velocity = imu_yaw_rate  # Already in rad/s
        
        # Update orientation
        delta_theta = angular_velocity * self.dt
        self.pose[2] += delta_theta
        
        # Update position
        delta_x = linear_velocity * np.cos(self.trajectory[-1][2]) * self.dt
        delta_y = linear_velocity * np.sin(self.trajectory[-1][2]) * self.dt
        self.pose[0] += delta_x
        self.pose[1] += delta_y
        
        # Save new pose
        self.trajectory.append(self.pose.copy())
    
    def plot_trajectory(self):
        """Plot the accumulated trajectory"""
        trajectory = np.array(self.trajectory)
        plt.figure(figsize=(10, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], label='Odometry Path')
        plt.scatter(0, 0, c='red', marker='*', s=200, label='Start')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Robot Trajectory from Encoder+IMU Odometry')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()




if __name__ == '__main__':

    dataset = 20
    data = load_sensor_data(dataset)

    tracker = OdometryTracker()

    # Simulate some motion data
    syn_enc, syn_enc_ts, syn_imu, syn_imu_ts = synchronize_data(data['encoder_counts'], data['encoder_stamps'], 
                                                                data['imu_angular_velocity'], data['imu_stamps'])

    # Process all data
    for enc_counts, yaw_rate in zip(syn_enc.T, syn_imu[-1].T):
        tracker.update_pose(enc_counts, yaw_rate)

    # Plot results
    tracker.plot_trajectory()