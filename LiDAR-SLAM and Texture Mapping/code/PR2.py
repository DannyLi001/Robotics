import cv2
import os
# import gtsam
# from gtsam import noiseModel
# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pr2_utils import bresenham2D
from scipy.spatial import cKDTree
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


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
    

    def load_image(args):
        """Helper function for parallel loading"""
        img_path, is_rgb = args
        if is_rgb:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return img

    def load_images_optimized(directory, is_rgb=False, num_workers=8):
        """Load images in parallel with sorted order"""
        # Get sorted file paths using scandir (faster than listdir)
        with os.scandir(directory) as entries:
            files = sorted([entry.path for entry in entries if entry.name.endswith('.png')], 
                        key=lambda x: os.path.basename(x))
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            args = [(f, is_rgb) for f in files]
            images = list(executor.map(load_image, args))
        
        return np.array(images)

    # Load data (adjust num_workers based on your CPU cores)
    sensor_data["disp_images"] = load_images_optimized(f"./data/dataRGBD/Disparity{dataset}", num_workers=8)
    sensor_data["rgb_images"] = load_images_optimized(f"./data/dataRGBD/RGB{dataset}", is_rgb=True, num_workers=8)

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

    for i in tqdm(range(1, len(point_clouds)), desc="icp trajectory"):
        # Get consecutive scans
        source_pc = point_clouds[i-1]
        target_pc = point_clouds[i]
        
        # Get temporal information
        t_prev = lidar_timestamps[i-1]
        t_curr = lidar_timestamps[i]
        
        # Get odometry-based relative transformation
        T_prev = get_odometry_pose(tracker.trajectory, t_prev, data['encoder_stamps'])  # pre_T
        T_curr = get_odometry_pose(tracker.trajectory, t_curr, data['encoder_stamps'])  # cur_T
        T_odom = np.linalg.inv(T_prev) @ T_curr # cur_T_pre

        # Perform ICP
        T_icp, _ = icp(source_pc, target_pc, T_odom, max_iterations=30) # cur_T_pre
        
        # Update trajectory
        corrected_trajectory.append(corrected_trajectory[-1] @ T_icp)   # cur_T

    # return np.array([transform_to_pose(T) for T in corrected_trajectory])
    return corrected_trajectory, point_clouds

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
    plt.plot(icp_traj[:,0], icp_traj[:,1], 'r-', label='ICP-Corrected')
    
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
    def __init__(self, resolution=0.1, size=60):
        self.resolution = resolution  # meters per cell
        self.size = size              # grid size in meters
        self.origin = np.array([size/resolution//2, size/resolution//2])  # center point
        self.grid = np.zeros((int(size/resolution), int(size/resolution)), dtype=np.float64)
        self.log_odds_free = 0.6
        self.log_odds_occ = -0.4
        self.max_log_odds = 100
        self.min_log_odds = -100

    def world_to_grid(self, point):
        col = int(point[0]/self.resolution + self.origin[0])
        row = int(point[1]/self.resolution + self.origin[1])
        return (row, col)  

    def update(self, scan, pose):
        # Convert pose to transformation matrix
        pose = np.array([transform_to_pose(pose)])
        x, y, theta = pose.flatten()    # robot position and angle in world
        T = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1]
        ])
        
        # Transform LiDAR points to world frame
        homogeneous_scan = np.vstack((scan.T, np.ones(scan.shape[0])))
        world_scan = (T @ homogeneous_scan).T[:, :2]    # scan in world
        
        # Filter valid points
        valid = np.linalg.norm(world_scan, axis=1) < 30  # 30m max range
        world_scan = world_scan[valid]
        
        # Get robot position in grid coordinates
        robot_grid = self.world_to_grid(np.array([x, y]))   # robot position in grid
        
        # Update grid using Bresenham's algorithm
        for point in world_scan:
            end = self.world_to_grid(point) # scan in grid
            if robot_grid == end:
                continue
            line = (bresenham2D(robot_grid[0], robot_grid[1], end[0], end[1]).T).astype(np.int32)
            
            # Filter valid indices
            valid_mask = (0 <= line[:, 0]) & (line[:, 0] < self.grid.shape[0]) & \
                        (0 <= line[:, 1]) & (line[:, 1] < self.grid.shape[1])

            valid_cells = line[valid_mask]

            # Update free cells using vectorized operations
            self.grid[valid_cells[:, 0], valid_cells[:, 1]] = np.clip(
                self.grid[valid_cells[:, 0], valid_cells[:, 1]] + self.log_odds_free,
                self.min_log_odds, self.max_log_odds
            )
            
            # Update occupied cell
            if 0 <= end[0] < self.grid.shape[0] and 0 <= end[1] < self.grid.shape[1]:
                self.grid[end[0], end[1]] = np.clip(
                    self.grid[end[0], end[1]] + self.log_odds_occ,
                    self.min_log_odds, self.max_log_odds
                )

class TextureMapper:
    def __init__(self, occupancy_grid):
        self.grid = np.zeros((occupancy_grid.grid.shape[0], 
                            occupancy_grid.grid.shape[1], 3), dtype=np.uint8)
        self.resolution = occupancy_grid.resolution
        self.origin = occupancy_grid.origin
        self.count = np.zeros((occupancy_grid.grid.shape[0], 
                             occupancy_grid.grid.shape[1]), dtype=np.int32)



def process_mapping(data, trajectory):
    # ================== 1. Precomputations ==================
    # Depth camera extrinsics (relative to robot center)
    depth_cam_pos = np.array([0.18, 0.005, 0.36])  # x, y, z
    depth_cam_rot = np.array([0, 0.36, 0.021])     # roll, pitch, yaw
    dept_T_rob = create_transform_matrix(depth_cam_pos, depth_cam_rot)    # dept_T_rob

    # Create RGB frame to trajectory index mapping
    lidar_stamps = data['lidar_stamps']
    rgb_stamps = data['rgb_stamps']
    traj_indices = np.searchsorted(lidar_stamps, rgb_stamps)
    traj_indices = np.clip(traj_indices, 0, len(trajectory)-1)
    rgb_frame_map = defaultdict(list)
    for rgb_idx, traj_idx in enumerate(traj_indices):   # key: traj_idx, value: rbg_idx
        rgb_frame_map[traj_idx].append(rgb_idx)

    # ================== 2. Initialize Maps ==================
    og = OccupancyGrid(resolution=0.1, size=60)
    texture_map = TextureMapper(og)
    angles = np.linspace(data['lidar_angle_min'], data['lidar_angle_max'],
                        data['lidar_ranges'].shape[0])

    # Precompute valid ranges mask once
    all_ranges = data['lidar_ranges']
    valid_mask = (all_ranges > 0.1) & (all_ranges < 30)  

    # ================== 3. Main Processing Loop ==================
    for i in tqdm(range(len(trajectory)), desc="Building Map"):
        # ========== 3.1 Update Occupancy Grid ==========
        ranges = all_ranges[:, i]
        valid = valid_mask[:, i]
        
        # Vectorized scan generation
        cos_angles = np.cos(angles[valid])
        sin_angles = np.sin(angles[valid])
        scan = np.column_stack((ranges[valid] * cos_angles,
                                ranges[valid] * sin_angles))
        og.update(scan, trajectory[i])

        # ========== 3.2 Process Kinect Data ==========
        if i in rgb_frame_map:
            dept_T = trajectory[i] @ dept_T_rob
            for rgb_idx in rgb_frame_map[i]:
                # Get synchronized data
                disp_img = data['disp_images'][rgb_idx]
                rgb_img = cv2.cvtColor(data['rgb_images'][rgb_idx], cv2.COLOR_BGR2RGB)

                # from writeup, compute correspondence
                height, width = disp_img.shape

                dd = np.array(-0.00304 * disp_img + 3.31)
                depth = 1.03 / dd

                mesh = np.meshgrid(np.arange(0, height), np.arange(0, width), indexing='ij')  
                i_idxs = mesh[0].flatten()
                j_idxs = mesh[1].flatten()

                rgb_i = np.array((526.37 * i_idxs + 19276 - 7877.07 * dd.flatten()) / 585.051)  # force int for indexing
                rgb_j = np.array((526.37 * j_idxs + 16662) / 585.051)

                rgb_i = np.round(rgb_i).astype(np.int32)
                rgb_j = np.round(rgb_j).astype(np.int32)

                # some may be out of bounds, just clip them
                rgb_i = np.clip(rgb_i, 0, height - 1)
                rgb_j = np.clip(rgb_j, 0, width - 1)

                colors = rgb_img[rgb_i, rgb_j].reshape((height, width, 3))
                colors_bgr = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR).reshape(-1, 3)

                # cv2.imshow("color", colors)
                
                uv1 = np.vstack([j_idxs, i_idxs, np.ones_like(i_idxs)])
                K = np.array([[585.05, 0, 242.94],
                            [0, 585.05, 315.84],
                            [0, 0, 1]])

                # project images to 3d points
                points = depth.flatten() * (np.linalg.inv(K) @ uv1)

                oRr = np.array([[0, -1, 0],
                                [0, 0, -1],
                                [1, 0, 0]])
                # we want rRo because we have points in optical frame and want to move them to the regular frame.
                points = oRr.T @ points

                homogeneous_points = np.vstack((points, np.ones(points.shape[1])))
                points_world = (dept_T @ homogeneous_points)[:3, :].T

                floor_mask = points_world[:, 2] < 0.1
                floor_points = points_world[floor_mask]
                floor_colors = colors_bgr[floor_mask]
                floor_colors = colors.reshape(-1, 3)[floor_mask]
                

                # Convert to grid coordinates
                map_origin = texture_map.origin
                grid_x = np.round((floor_points[:, 0] / og.resolution + map_origin[0])).astype(int)
                grid_y = np.round((floor_points[:, 1] / og.resolution + map_origin[1])).astype(int)

                # Update texture grid (vectorized)
                valid = (grid_x >= 0) & (grid_x < texture_map.grid.shape[1]) & \
                        (grid_y >= 0) & (grid_y < texture_map.grid.shape[0])
                
                # Use advanced indexing for efficient assignment
                texture_map.grid[grid_y[valid], grid_x[valid]] = floor_colors[valid]

                # cv2.imshow("1", texture_map.grid)

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(points.T)
                # pcd.colors = o3d.utility.Vector3dVector(colors_rgb.reshape(-1, 3) / 255)  # open3d expects color channels 0-1, opencv uses uint8 0-255

                # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)  # visualize the camera regular frame for reference.

                # o3d.visualization.draw_geometries([pcd, origin])  # display the pointcloud and origin

    # ================== 4. Visualization ==================
    plt.figure(figsize=(12, 8))
    plt.imshow(1 - 1/(1+np.exp(og.grid)), cmap='gray', origin='lower')
    plt.title('Optimized Occupancy Map')
    plt.colorbar(label='Occupancy Probability')
    plt.show()

    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(texture_map.grid, cv2.COLOR_BGR2RGB), origin='lower')
    plt.title('Texture Map')
    plt.show()
    
    return og, texture_map

def create_transform_matrix(position, euler_angles):
    """Create 4x4 transformation matrix from Euler angles (RPY) and position"""
    roll, pitch, yaw = euler_angles
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    R = R_z @ R_y @ R_x
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def optimize_with_gtsam(icp_trajectory, point_clouds):
    # Convert ICP trajectory to initial estimates in GTSAM's Pose2 format
    initial_estimates = gtsam.Values()
    for i, T in enumerate(icp_trajectory):
        x, y, theta = transform_to_pose(T)
        initial_estimates.insert(i, gtsam.Pose2(x, y, theta))
    
    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    
    # Add prior on the first pose to anchor the graph
    prior_noise = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # Tune these values
    first_pose = initial_estimates.atPose2(0)
    graph.add(gtsam.PriorFactorPose2(0, first_pose, prior_noise))
    
    # Add odometry factors between consecutive poses
    odometry_noise = noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # Tune these values
    for i in range(1, len(icp_trajectory)):
        # Calculate relative pose from ICP result
        T_prev = icp_trajectory[i-1]
        T_curr = icp_trajectory[i]
        T_rel = np.linalg.inv(T_prev) @ T_curr
        dx, dy, dtheta = transform_to_pose(T_rel)
        # Add between factor to the graph
        graph.add(gtsam.BetweenFactorPose2(i-1, i, gtsam.Pose2(dx, dy, dtheta), odometry_noise))
    
    # Fixed-interval loop closure
    loop_closure_noise = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # Tighter noise for loop closures
    for i in tqdm(range(0, len(icp_trajectory) - 10, 10), desc="Fixed-interval loop closures"):
        j = i + 10
        if j >= len(icp_trajectory):
            continue
        # Get corresponding LiDAR scans
        source_pc = point_clouds[i]
        target_pc = point_clouds[j]
        # Initial guess from current trajectory
        T_i = icp_trajectory[i]
        T_j = icp_trajectory[j]
        initial_pose = np.linalg.inv(T_i) @ T_j
        # Perform ICP
        T_icp, error = icp(source_pc, target_pc, initial_pose, max_iterations=50)
        if error < 0.1:  # Threshold to determine valid loop closure
            dx, dy, dtheta = transform_to_pose(T_icp)
            graph.add(gtsam.BetweenFactorPose2(i, j, gtsam.Pose2(dx, dy, dtheta), loop_closure_noise))
    
    # Proximity-based loop closure
    positions = np.array([transform_to_pose(T)[:2] for T in icp_trajectory])
    tree = cKDTree(positions)
    radius = 0.05  # meters for proximity search
    proximity_noise = noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    
    for i in tqdm(range(len(icp_trajectory)), desc="Proximity loop closures"):
        # Find nearby poses within radius
        neighbors = tree.query_ball_point(positions[i], radius)[:5]
        for j in neighbors:
            if j <= i:  # Avoid duplicate checks and self
                continue
            # Get LiDAR scans
            source_pc = point_clouds[i]
            target_pc = point_clouds[j]
            # Initial guess from current trajectory
            T_i = icp_trajectory[i]
            T_j = icp_trajectory[j]
            initial_pose = np.linalg.inv(T_i) @ T_j
            # Perform ICP
            T_icp, error = icp(source_pc, target_pc, initial_pose, max_iterations=50)
            # Check if match is physically plausible
            if error < 0.1 and np.linalg.norm(T_icp[:2,3]) < 1.5:  # Thresholds for translation
                dx, dy, dtheta = transform_to_pose(T_icp)
                graph.add(gtsam.BetweenFactorPose2(i, j, gtsam.Pose2(dx, dy, dtheta), proximity_noise))
    
    # Optimize the factor graph
    parameters = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, parameters)
    result = optimizer.optimize()
    
    # Convert optimized poses back to 4x4 transformation matrices
    optimized_trajectory = []
    for i in range(len(icp_trajectory)):
        pose = result.atPose2(i)
        T = np.array([
            [np.cos(pose.theta()), -np.sin(pose.theta()), 0, pose.x()],
            [np.sin(pose.theta()),  np.cos(pose.theta()), 0, pose.y()],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        optimized_trajectory.append(T)
    
    return optimized_trajectory



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
    icp_trajectory, point_clouds = run_scan_matching(data, tracker)   # list of cur_T transformation matrix
    
    plot_results(tracker.trajectory, np.array([transform_to_pose(T) for T in icp_trajectory]))

    # Optimize trajectory using GTSAM with loop closures
    # optimized_trajectory = optimize_with_gtsam(icp_trajectory, point_clouds)

    # Plot results
    # plot_results(tracker.trajectory, np.array([transform_to_pose(T) for T in optimized_trajectory]))

    # Process mapping with optimized trajectory
    # occ_grid_gt, tex_map_gt = process_mapping(data, optimized_trajectory)



    # Process mapping
    occupancy_grid, texture_map = process_mapping(data, icp_trajectory)

    valid_occ = occupancy_grid.grid.reshape(-1) > 0
    # valid_occ = occ_grid_gt.grid.reshape(-1) > 0
    valid_tex = (texture_map.grid.reshape(-1, 3) > 0)[:, 0]
    # valid_tex = (tex_map_gt.grid.reshape(-1, 3) > 0)[:, 0]
    intersect = valid_occ != valid_tex

    final = texture_map.grid.reshape(-1, 3)
    final[intersect] = np.array([255, 255, 255])
    final = final.reshape(600, 600, 3)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), origin='lower')
    plt.title('Combined Map')
    plt.show()

    pass