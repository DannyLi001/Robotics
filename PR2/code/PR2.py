import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt


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
    for i in range(max_iterations):
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

def lidar_to_pointcloud(ranges, angles):
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    return np.column_stack((x, y, np.zeros_like(x)))

def run_scan_matching(data, tracker):
    # Generate LiDAR point clouds
    angles = np.linspace(data['lidar_angle_min'], data['lidar_angle_max'], 
                        data['lidar_ranges'].shape[0])
    point_clouds = [lidar_to_pointcloud(ranges, angles)
                   for ranges in data['lidar_ranges'].T]

    point_clouds = np.array(point_clouds)

    # Initialize corrected trajectory
    corrected_trajectory = [np.eye(4)]
    lidar_timestamps = data['lidar_stamps']

    for i in range(1, len(point_clouds)):
        # Get consecutive scans
        source_pc = point_clouds[i-1]
        target_pc = point_clouds[i]
        

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







class OccupancyGrid:
    def __init__(self, resolution=0.1, size=100):
        self.resolution = resolution  # meters per cell
        self.size = size              # grid size in meters
        self.origin = np.array([size/resolution//2, size/resolution//2])  # center point
        self.grid = np.zeros((int(size/resolution), int(size/resolution)), dtype=np.float32)
        self.log_odds_free = -0.4
        self.log_odds_occ = 0.6
        self.max_log_odds = 100
        self.min_log_odds = -100

    def world_to_grid(self, point):
        return ((point / self.resolution) + self.origin).astype(int)

    def update(self, scan, pose):
        # Convert pose to transformation matrix
        x, y, theta = pose
        T = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1]
        ])
        
        # Transform LiDAR points to world frame
        homogeneous_scan = np.vstack((scan.T, np.ones(scan.shape[0])))
        world_scan = (T @ homogeneous_scan).T[:, :2]
        
        # Filter valid points
        valid = np.linalg.norm(world_scan, axis=1) < 30  # 30m max range
        world_scan = world_scan[valid]
        
        # Get robot position in grid coordinates
        robot_grid = self.world_to_grid(np.array([x, y]))
        
        # Update grid using Bresenham's algorithm
        for point in world_scan:
            end = self.world_to_grid(point)
            line = self.bresenham(robot_grid, end)
            
            # Update free cells
            for cell in line[:-1]:
                if 0 <= cell[0] < self.grid.shape[0] and 0 <= cell[1] < self.grid.shape[1]:
                    self.grid[cell[0], cell[1]] = np.clip(
                        self.grid[cell[0], cell[1]] + self.log_odds_free,
                        self.min_log_odds, self.max_log_odds
                    )
            
            # Update occupied cell
            if 0 <= end[0] < self.grid.shape[0] and 0 <= end[1] < self.grid.shape[1]:
                self.grid[end[0], end[1]] = np.clip(
                    self.grid[end[0], end[1]] + self.log_odds_occ,
                    self.min_log_odds, self.max_log_odds
                )

    def bresenham(self, start, end):
        """Bresenham's line algorithm for 2D grid"""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steep = dy > dx
        
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = 0
        ystep = 1 if y0 < y1 else -1
        y = y0
        line = []
        
        for x in range(x0, x1 + 1):
            coord = (y, x) if steep else (x, y)
            line.append(coord)
            error += dy
            if 2*error >= dx:
                y += ystep
                error -= dx
        return line

class TextureMapper:
    def __init__(self, occupancy_grid):
        self.grid = np.zeros((occupancy_grid.grid.shape[0], 
                            occupancy_grid.grid.shape[1], 3), dtype=np.uint8)
        self.resolution = occupancy_grid.resolution
        self.origin = occupancy_grid.origin
        
    def add_rgbd_frame(self, rgb, depth, pose, intrinsics):
        # Create point cloud from depth image
        depth = depth.astype(float) / 1000  # convert mm to meters
        u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        z = depth
        x = (u - intrinsics['cx']) * z / intrinsics['fx']
        y = (v - intrinsics['cy']) * z / intrinsics['fy']
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        
        # Transform to world coordinates
        T = pose_to_matrix(pose)
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        world_points = (T @ homogeneous_points.T).T[:, :3]
        
        # Filter floor points (z < 0.2m)
        floor_mask = (world_points[:, 2] < 0.2)
        floor_points = world_points[floor_mask]
        colors = rgb.reshape(-1, 3)[floor_mask]
        
        # Convert to grid coordinates
        grid_coords = ((floor_points[:, :2] / self.resolution) + self.origin[:2]).astype(int)
        
        # Update texture map
        for (x, y), color in zip(grid_coords, colors):
            if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                self.grid[x, y] = color

def process_mapping(data, trajectory):
    # Initialize maps
    og = OccupancyGrid(resolution=0.1, size=100)
    texture_map = TextureMapper(og)
    
    # Process first LiDAR scan
    angles = np.linspace(data['lidar_angle_min'], data['lidar_angle_max'],
                        data['lidar_ranges'].shape[0])
    ranges = data['lidar_ranges'][:, 0]
    valid = (ranges > 0.1) & (ranges < 30)  # valid range 0.1-30 meters
    scan = np.vstack((ranges[valid] * np.cos(angles[valid]),
                     ranges[valid] * np.sin(angles[valid]))).T
    
    # Update with initial pose (assuming identity)
    og.update(scan, np.array([0, 0, 0]))
    
    # Visualize initial map
    plt.figure()
    plt.imshow(1 - 1/(1+np.exp(og.grid)), cmap='gray', origin='lower')
    plt.title('Initial Occupancy Map')
    plt.show()
    
    # Process all scans
    for i in range(len(trajectory)):
        # Update occupancy grid
        ranges = data['lidar_ranges'][:, i]
        valid = (ranges > 0.1) & (ranges < 30)
        scan = np.vstack((ranges[valid] * np.cos(angles[valid]),
                         ranges[valid] * np.sin(angles[valid]))).T
        og.update(scan, trajectory[i])
        
        # Update texture map with Kinect data
        if i < len(data['rgb_stamps']):
            # Find closest RGBD frame
            idx = np.argmin(np.abs(data['rgb_stamps'][i] - data['lidar_stamps']))
            rgb = cv2.cvtColor(data['rgb_images'][idx], cv2.COLOR_BGR2RGB)
            depth = data['depth_images'][idx]
            texture_map.add_rgbd_frame(rgb, depth, trajectory[i], 
                                     intrinsics={'fx': 525, 'fy': 525,
                                                'cx': 319.5, 'cy': 239.5})
    
    return og, texture_map

def postprocess_maps(occupancy_grid, texture_map):
    # Apply thresholding to occupancy grid
    prob_map = 1 / (1 + np.exp(-occupancy_grid.grid))
    occ_map = (prob_map > 0.65).astype(np.uint8) * 255
    free_map = (prob_map < 0.35).astype(np.uint8) * 255
    
    # Apply median filter to texture map
    texture_map.grid = cv2.medianBlur(texture_map.grid, 3)
    
    return occ_map, texture_map.grid









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

    # Process mapping
    occupancy_grid, texture_map = process_mapping(data, icp_trajectory)
    occ_map, color_map = postprocess_maps(occupancy_grid, texture_map)
    
    # Visualize final maps
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(occ_map, cmap='gray', origin='lower')
    ax[0].set_title('Occupancy Map')
    ax[1].imshow(color_map, origin='lower')
    ax[1].set_title('Texture Map')
    plt.show()
