import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from scipy.spatial import cKDTree

def rotation_matrix_z(yaw):
    """Generate a 4x4 rotation matrix around the z-axis."""
    R = np.eye(4)
    R[:3, :3] = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    return R

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

if __name__ == "__main__":
    obj_name = 'drill'  # 'drill' or 'liq_container'
    num_pc = 4  # number of point clouds
    source_pc = read_canonical_model(obj_name)
    
    yaw_angles = np.linspace(0, 2 * np.pi, num=36)  # 10-degree steps
    for i in range(num_pc):
        print(f"Processing point cloud {i+1}")
        target_pc = load_pc(obj_name, i)
        
        best_pose = np.eye(4)
        best_error = float('inf')

        target_centroid = np.mean(target_pc, axis=0)
        # Translation matrix to center the source at the origin
        T = np.eye(4)
        T[:3, 3] = -target_centroid
        
        # visualize_icp_result(source_pc, target_pc, T)
        for yaw in yaw_angles:
            print(f"current yaw: {yaw:.4f}")
            # Initialize pose: center source at origin, then apply yaw rotation
            initial_pose = rotation_matrix_z(yaw) @ T  # Combine rotation and translation
            
            # Run ICP
            pose, error = icp(source_pc, target_pc, initial_pose)
            
            # Update best pose if error is lower
            if error < best_error:
                best_error = error
                best_pose = pose.copy()
        
        # Visualize the best result
        visualize_icp_result(source_pc, target_pc, best_pose)