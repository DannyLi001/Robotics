import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


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


def icp(source, target, initial_pose, max_iterations=50, tolerance=5e-7):
    """Perform ICP to align target to source given an initial pose."""
    current_pose = initial_pose.copy()
    prev_error = 0
    for _ in range(max_iterations):
        # Transform target points with current pose
        homogeneous_target = np.hstack((target, np.ones((target.shape[0], 1))))
        transformed = (current_pose @ homogeneous_target.T).T[:, :3]
        
        # Find nearest neighbors in target
        tree = cKDTree(source)
        distances, indices = tree.query(transformed)
        correspondences = source[indices]
        
        # Compute mean squared error
        current_error = np.mean(distances ** 2)
        if abs(prev_error - current_error) < tolerance:
            # print(abs(prev_error - current_error))
            break
        prev_error = current_error
        
        # Compute optimal R and t using SVD
        A = transformed
        B = correspondences
        A_centroid = np.mean(A, axis=0)
        B_centroid = np.mean(B, axis=0)
        H = (A - A_centroid).T @ (B - B_centroid)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Correct reflection if necessary
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = B_centroid - R @ A_centroid
        
        # Update current pose
        delta_pose = np.eye(4)
        delta_pose[:3, :3] = R
        delta_pose[:3, 3] = t
        current_pose = delta_pose @ current_pose
    
    return current_pose, prev_error


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

def lidar_to_pointcloud(ranges, angles, valid_range=(0.1, 30.0)):
    valid = (ranges > valid_range[0]) & (ranges < valid_range[1])
    x = ranges[valid] * np.cos(angles[valid])
    y = ranges[valid] * np.sin(angles[valid])
    return np.column_stack((x, y, np.zeros_like(x)))

def preprocess_point_cloud(pc, voxel_size=0.1):
    """Downsample point cloud using voxel grid filtering"""
    if not pc.any(): return pc
    
    # Voxel grid downsampling
    voxel_grid = {}
    for point in pc:
        voxel = tuple((point // voxel_size).astype(int))
        voxel_grid.setdefault(voxel, []).append(point)
    
    return np.array([np.mean(points, axis=0) for points in voxel_grid.values()])

def run_scan_matching(data, tracker):
    # Generate LiDAR point clouds
    angles = np.linspace(data['lidar_angle_min'], data['lidar_angle_max'], 
                        data['lidar_ranges'].shape[0])
    point_clouds = [preprocess_point_cloud(lidar_to_pointcloud(ranges, angles)) 
                   for ranges in data['lidar_ranges'].T]

    # Initialize corrected trajectory
    corrected_trajectory = [np.eye(4)]
    lidar_timestamps = data['lidar_stamps']

    for i in range(1, len(point_clouds)):
        # Get consecutive scans
        source_pc = point_clouds[i-1]
        target_pc = point_clouds[i]
        if len(source_pc) < 100 or len(target_pc) < 100: continue

        # Get temporal information
        t_prev = lidar_timestamps[i-1]
        t_curr = lidar_timestamps[i]
        
        # Get odometry-based relative transformation
        T_prev = get_odometry_pose(tracker.trajectory, t_prev, data['encoder_stamps'])
        T_curr = get_odometry_pose(tracker.trajectory, t_curr, data['encoder_stamps'])
        T_odom = np.linalg.inv(T_prev) @ T_curr

        # Perform ICP
        T_icp, _ = icp(source_pc, target_pc, T_odom, max_iterations=30)
        
        # Update trajectory
        corrected_trajectory.append(corrected_trajectory[-1] @ T_icp)

    return corrected_trajectory

def get_odometry_pose(trajectory, lidar_ts, encoder_tss):
    """Get interpolated odometry pose at specified timestamp"""
    times = encoder_tss
    poses = trajectory
    
    idx = np.searchsorted(times, lidar_ts)
    if idx == 0 or idx == len(times):
        return pose_to_transform(poses[idx])
    
    # Linear interpolation
    alpha = (lidar_ts - times[idx-1]) / (times[idx] - times[idx-1])
    interp_pose = poses[idx-1] + alpha * (poses[idx] - poses[idx-1])
    return pose_to_transform(interp_pose)

def pose_to_transform(pose):
    """Convert [x, y, theta] to 4x4 transformation matrix"""
    x, y, theta = pose
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, x],
        [np.sin(theta), np.cos(theta), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def plot_results(odom_traj, icp_traj):
    plt.figure(figsize=(12, 6))
    
    # Plot odometry trajectory
    odom_xy = np.array([p[:2] for p in odom_traj])
    plt.plot(odom_xy[:,0], odom_xy[:,1], 'b-', label='Wheel Odometry')
    
    # Plot ICP-corrected trajectory
    icp_xy = np.array([transform_to_pose(T)[:2] for T in icp_traj])
    plt.plot(icp_xy[:,0], icp_xy[:,1], 'r-', label='ICP-Corrected')
    
    plt.title('Trajectory Comparison')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def transform_to_pose(T):
    """Convert 4x4 matrix to [x, y, theta]"""
    theta = np.arctan2(T[1,0], T[0,0])
    return np.array([T[0,3], T[1,3], theta])

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
    
    # Perform scan matching
    icp_trajectory = run_scan_matching(data, tracker)
    
    plot_results(tracker.trajectory, icp_trajectory)