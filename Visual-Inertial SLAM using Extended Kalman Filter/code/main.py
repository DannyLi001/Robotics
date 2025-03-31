import numpy as np
import cv2
from pr3_utils import *
from scipy.sparse import lil_matrix
from tqdm import tqdm
from scipy.linalg import block_diag

def compute_jacobian(T_cam, q_cam, K):
    q = q_cam.flatten()
    dpi_dq = np.array([
        [1/q[2], 0,     -q[0]/q[2]**2, 0],
        [0,      1/q[2], -q[1]/q[2]**2, 0],
        [0,      0,      0,             0],
        [0,      0,      -q[3]/q[2]**2, 1/q[2]]
    ])  # 投影函数导数（4x4）

    P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # 提取非齐次坐标（3x4）
    H = K @ dpi_dq[:3, :] @ T_cam @ P.T  # 雅可比矩阵（3x3）
    return H

opt_T_cam = np.array([
    [0, -1, 0, 0],  # IMU的Y轴（左）对应光学X轴的负方向（右）
    [0, 0, -1, 0],  # IMU的Z轴（上）对应光学Y轴的负方向（下）
    [1, 0, 0, 0],
    [0, 0, 0, 1]# IMU的X轴（前）对应光学Z轴的正方向（前）
])

def triangulate(left_uv, right_uv, imu_T, extL_T_imu, extR_T_imu, K_l, K_r):
    # 构造左右相机的投影矩阵 P = K * [R | t]
    left_T = opt_T_cam @ extL_T_imu @ imu_T  # 世界到左相机的变换
    right_T = opt_T_cam @ extR_T_imu @ imu_T  # 世界到右相机的变换
    P_left = left_T[:3, :]
    P_right = right_T[:3, :]

    # 构建线性方程组 Ax=0
    A = []
    u_l, v_l, _ = np.linalg.inv(K_l) @ np.append(left_uv,1)
    u_r, v_r, _ = np.linalg.inv(K_r) @ np.append(right_uv,1)
    A.append(u_l * P_left[2] - P_left[0])
    A.append(v_l * P_left[2] - P_left[1])
    A.append(u_r * P_right[2] - P_right[0])
    A.append(v_r * P_right[2] - P_right[1])
    A = np.array(A)
    
    # SVD求解最小二乘解
    _, _, V = np.linalg.svd(A)
    m_homo = V[-1]  # 最小奇异值对应的解
    m = m_homo[:3] / m_homo[3]
    
    return m

if __name__ == '__main__':

    # Load the measurements
    dataset = "dataset00"
    filename = f"./data/{dataset}/{dataset}.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(
        filename)
    
    imu_T_extL = extL_T_imu
    imu_T_extR = extR_T_imu
    extL_T_imu = inversePose(extL_T_imu)
    extR_T_imu = inversePose(extR_T_imu)
    mu_pose = np.eye(4)  # 初始位姿 SE(3) 4x4
    Sigma_pose = np.eye(6)*1e-6  # 位姿协方差 6x6

    # 噪声参数
    W = np.diag([0.02, 0.02, 0.02, 0.01, 0.01, 0.01])  # 过程噪声 6x6

    traj_mu = [mu_pose.copy()]
    traj_cov = [Sigma_pose.copy()]

    u_t = np.hstack((v_t, w_t)) # Tx6

    # 主循环
    for i in tqdm(range(len(timestamps)-1)):
        # (a) IMU预测步骤
        tau = timestamps[i+1] - timestamps[i]

        F = pose2adpose(axangle2pose(-u_t[i]*tau))  # df/du Ad(SE(3)) 6x6

        # 更新均值和协方差
        mu_pose = mu_pose @ axangle2pose(u_t[i]*tau)    # IMU在t时的世界坐标 SE(4) 4x4
        Sigma_pose = F @ Sigma_pose @ F.T + W*tau   # IMU在t时的位置covariance 6x6
        traj_mu.append(mu_pose.copy())
        traj_cov.append(Sigma_pose.copy())
        
    traj_mu = np.array(traj_mu)       # Tx4x4
    traj_cov = np.array(traj_cov)             # Tx6x6
    visualize_trajectory_2d(traj_mu, show_ori=False)






    # Part (b): Landmark Mapping via EKF Update
    num_landmarks = features.shape[1]
    landmark_means = np.zeros(num_landmarks*3)  # 3M
    invalid_landmarks = np.zeros_like(landmark_means, dtype=bool)
    valid_landmarks = np.ones_like(landmark_means, dtype=bool)

    # 参数配置
    min_observed_frames = 3    # 最小有效帧数
    min_disparity = 3          # 最小视差（像素）
    max_reprojection_error = 5 # 最大重投影误差（像素）
    min_depth, max_depth = 0.5, 50.0  # 深度范围
    W, H = 640, 480           # 图像尺寸（假设）
    border_thresh = 10         # 图像边界阈值
    
    def project_point_to_camera(point_world, opt_T, K):
        point_hom = np.append(point_world, 1.0)
        point_cam_hom = opt_T @ point_hom
        point_cam = point_cam_hom[:3] / point_cam_hom[2]
        return (K @ point_cam)[:2]
    
    for j in range(num_landmarks):
        start = 3*j
        end = start+3
        valid_frames = np.where(features[0, j, :] != -1)[0]
        
        if len(valid_frames) == 0:
            invalid_landmarks[start:end] = True
            valid_landmarks[start:end] = False
            continue
        
        # 条件1：有效帧数不足
        if len(valid_frames) < min_observed_frames:
            invalid_landmarks[start:end] = True
            valid_landmarks[start:end] = False
            continue
        
        t = valid_frames[0]
        left_uv = features[:2, j, t]  # 左相机像素坐标
        right_uv = features[2:, j, t]  # 右相机像素坐标
        imu_T = inversePose(traj_mu[t])      # 当前IMU位姿
        landmark_means[start:end] = triangulate(
                left_uv, right_uv, imu_T, extL_T_imu, extR_T_imu, K_l, K_r)
        
        # 条件2：三维距离检查
        dist = np.linalg.norm(landmark_means[start:end] - traj_mu[t, :3, 3])
        
        # 条件3：视差检查
        disparity = left_uv[0] - right_uv[0]
        
        # 条件4：重投影误差和深度检查
        left_T = opt_T_cam @ extL_T_imu @ imu_T  # 世界到左相机的变换
        right_T = opt_T_cam @ extR_T_imu @ imu_T # 世界到右相机的变换
        uv_left_proj = project_point_to_camera(landmark_means[start:end], left_T, K_l)
        uv_right_proj = project_point_to_camera(landmark_means[start:end], right_T, K_r)
        error_left = np.linalg.norm(uv_left_proj - left_uv)
        error_right = np.linalg.norm(uv_right_proj - right_uv)
        total_error = error_left + error_right
        
        # 左相机坐标系下的深度
        P_cam_left = left_T @ np.append(landmark_means[start:end], 1.0)
        depth = P_cam_left[2]
        
        # 条件5：图像边界检查
        out_of_bounds = (
            left_uv[0] < border_thresh or 
            left_uv[0] > W - border_thresh or
            left_uv[1] < border_thresh or 
            left_uv[1] > H - border_thresh
        )
        
        # 综合过滤条件
        if (
            dist < 0.5 or dist > 50 or
            abs(disparity) < min_disparity or
            total_error > max_reprojection_error or
            depth < min_depth or depth > max_depth or
            out_of_bounds or
            disparity < 0 
            ):
            invalid_landmarks[start:end] = True
            valid_landmarks[start:end] = False
        
    valid_landmarks_reshaped = valid_landmarks.reshape(-1, 3)
    landmark_validity = np.all(valid_landmarks_reshaped, axis=1)  # [M,] 布尔数组
    valid_indices = np.where(landmark_validity)[0]  # 有效地标的原始索引

    # 随机采样（最多1000个）
    max_landmarks = 1000
    if len(valid_indices) > max_landmarks:
        np.random.seed(0)
        selected_indices = np.random.choice(
            valid_indices, 
            size=max_landmarks, 
            replace=False
        )
    else:
        selected_indices = valid_indices  # 不足1000个时全部保留

    # 构建新的有效性标记（仅保留被选中的地标）
    new_valid_mask = np.zeros(num_landmarks*3, dtype=bool)
    for idx in selected_indices:
        start = idx * 3
        end = start + 3
        new_valid_mask[start:end] = True

    # 更新有效性标记
    valid_landmarks = new_valid_mask
    invalid_landmarks = ~new_valid_mask
    
    # plt.scatter(landmark_means[:].reshape(-1,3)[:,0],landmark_means[:].reshape(-1,3)[:,1],s=1)
    plt.scatter(landmark_means[valid_landmarks].reshape(-1,3)[:,0],landmark_means[valid_landmarks].reshape(-1,3)[:,1],s=1)
    plt.plot(traj_mu[:,0,3],traj_mu[:,1,3],'r-')
    plt.show()
    
    
    
    
    
    
    
    
    features = features[:, valid_landmarks.reshape(-1,3)[:,0], :]
    valid_landmark_means = landmark_means[valid_landmarks]
    landmark_covs = lil_matrix((features.shape[1]*3, features.shape[1]*3))    # 3Mx3M
    landmark_covs.setdiag(1e3)
    V = np.eye(4)*1e6  # 4x4
    traj_update = []
    cov_update = []
    
    for t in tqdm(range(len(timestamps)-1)):
        mu_pose  = traj_mu[t]
        Sigma_pose  = traj_cov[t]
        
        active_mask = (features[0, :, t] != -1)
        active_indices = np.where(active_mask)[0]
        
        if len(active_indices) > 0:
            z_left = features[:2, active_indices, t].T    # (M,2)
            z_right = features[2:, active_indices, t].T   # (M,2)
            active_means = valid_landmark_means.reshape(-1,3)[active_indices]  # (M,3)
            
            # 构造噪声矩阵（4M x 4M块对角）
            V_extended = block_diag(*[V for _ in range(len(active_indices))])
            
            # 计算残差和雅可比
            r_total = []
            H_landmarks = np.zeros((4*len(active_indices), 3*len(active_indices)))
            for i, idx in enumerate(active_indices):
                start = idx*3
                end = start+3
                m_j = valid_landmark_means[start:end]
                
                # 左相机投影
                left_T = opt_T_cam @ extL_T_imu @ inversePose(mu_pose)
                q_cam_left = left_T @ np.append(m_j, 1.0)
                z_pred_left = (K_l @ (q_cam_left[:3]/q_cam_left[2]))[:2]
                
                # 右相机投影
                right_T = opt_T_cam @ extR_T_imu @ inversePose(mu_pose)
                q_cam_right = right_T @ np.append(m_j, 1.0)
                z_pred_right = (K_r @ (q_cam_right[:3]/q_cam_right[2]))[:2]
                
                # 残差
                r = np.hstack([z_left[i] - z_pred_left, 
                             z_right[i] - z_pred_right])  # (4,)
                r_total.append(r)
                
                # 雅可比（地标）
                H_left = compute_jacobian(left_T, q_cam_left, K_l)[:2, :]
                H_right = compute_jacobian(right_T, q_cam_right, K_r)[:2, :]
                H_landmarks[4*i:4*i+4, 3*i:3*i+3] = np.vstack([H_left, H_right])
            
            # 堆叠为全局矩阵
            r_total = np.hstack(r_total)          # (4M,)
            
            # 卡尔曼增益
            label = np.zeros(features.shape[1]*3, dtype=bool)
            for j in range(3):
                label[active_indices*3+j] = True
            Sigma_landmarks = landmark_covs[label][:,label]
            S = H_landmarks @ Sigma_landmarks @ H_landmarks.T + V_extended
            K = Sigma_landmarks @ H_landmarks.T @ np.linalg.inv(S)
            
            # 更新地标
            delta = (K @ r_total).reshape(-1,3)
            valid_landmark_means.reshape(-1,3)[active_indices] += delta
            
            # 更新协方差
            I = np.eye(3*len(active_indices))
            Sigma_landmarks_new = (I - K @ H_landmarks) @ Sigma_landmarks
            landmark_covs[label][:,label] = Sigma_landmarks_new
        
        
        # --- 位姿更新（任务4）---
        if len(active_indices) > 0:
            H_pose = []
            r_pose = []
            for i, idx in enumerate(active_indices):
                m_j = valid_landmark_means.reshape(-1,3)[idx]
                
                def compute_jacobian_pose(T_imu, m_j, K, cam_T_imu):
                    # 将地标转换为齐次坐标
                    m_hom = np.append(m_j, 1.0)  # (4,)
                    
                    # 计算世界到相机的变换: T_cam = T_cam_to_imu @ T_imu^{-1}
                    imu_T = np.linalg.inv(T_imu)
                    T_cam = opt_T_cam @ cam_T_imu @ imu_T  # (4x4)
                    
                    # 将地标变换到相机坐标系
                    p_cam_hom = T_cam @ m_hom  # (4,)
                    q = p_cam_hom
                    p_cam = p_cam_hom[:3] / p_cam_hom[3]  # (x, y, z)
                    x, y, z = p_cam[0], p_cam[1], p_cam[2]
                    
                    dpi_dq = np.array([
                        [1/q[2], 0,     -q[0]/q[2]**2, 0],
                        [0,      1/q[2], -q[1]/q[2]**2, 0],
                        [0,      0,      0,             0],
                        [0,      0,      -q[3]/q[2]**2, 1/q[2]]
                    ])  # 投影函数导数（4x4）
                    
                    # 计算相机坐标对位姿的导数 ∂p_cam/∂ξ (左扰动模型)
                    dpcam_dxi = np.array([
                        [1, 0, 0, 0, z, -y],
                        [0, 1, 0, -z, 0, x],
                        [0, 0, 1, y, -x, 0],
                        [0, 0, 0, 0, 0, 0]
                    ])  # (3x6)
                    
                    H = K[:2] @ dpi_dq[:3] @ opt_T_cam @ dpcam_dxi  # (2x6)
                    
                    return H
                
                # 计算位姿雅可比
                H_left = compute_jacobian_pose(mu_pose, m_j, K_l, extL_T_imu)
                H_right = compute_jacobian_pose(mu_pose, m_j, K_r, extR_T_imu)
                H_pose.append(np.vstack([H_left, H_right]))  # (4,6)
                
                # 计算残差（同上）
                # ...（与地标更新部分共享残差计算）...
                r_pose.append(r_total[4*i:4*(i+1)])
            
            H_pose = np.vstack(H_pose)  # (4M,6)
            r_pose = np.hstack(r_pose)  # (4M,)
            
            # 卡尔曼增益
            S_pose = H_pose @ Sigma_pose @ H_pose.T + V_extended
            K_pose = Sigma_pose @ H_pose.T @ np.linalg.inv(S_pose)
            
            # 更新位姿
            delta_xi = K_pose @ r_pose
            mu_pose = axangle2pose(delta_xi) @ mu_pose
            
            # 更新协方差
            I = np.eye(6)
            Sigma_pose = (I - K_pose @ H_pose) @ Sigma_pose 
        
        # 保存当前状态
        traj_update.append(mu_pose.copy())
        cov_update.append(Sigma_pose.copy())
        
    traj_update = np.array(traj_update)
    
    # 可视化最终结果
    plt.figure(figsize=(5,5))
    plt.scatter(valid_landmark_means.reshape(-1,3)[:,0], 
                valid_landmark_means.reshape(-1,3)[:,1], 
                s=1, c='r', label='Landmarks')
    plt.plot(traj_mu[:,0,3], traj_mu[:,1,3], 'r-', label='Trajectory')
    plt.plot(traj_update[:,0,3], traj_update[:,1,3], 'b-', label='Trajectory Updated')
    plt.legend()
    plt.show()
        