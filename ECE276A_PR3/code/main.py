import cv2
import numpy as np
from pr3_utils import *
from scipy.sparse import lil_matrix
from scipy.linalg import inv
from tqdm import tqdm
def compute_jacobian(T_cam, q_cam, K):
    """
    计算雅可比矩阵 H_j = K * dπ/dq * T_cam * P^T
    - T_cam: 世界坐标系到相机坐标系的变换矩阵（4x4）
    - q_cam: 地标在相机坐标系下的齐次坐标（4x1）
    - K: 相机内参矩阵（3x3）
    """
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


R_imu_to_optical = np.array([
    [0, -1, 0],  # IMU的Y轴（左）对应光学X轴的负方向（右）
    [0, 0, -1],  # IMU的Z轴（上）对应光学Y轴的负方向（下）
    [1, 0, 0]    # IMU的X轴（前）对应光学Z轴的正方向（前）
])

def correct_extrinsic(ext):
    corrected = np.eye(4)
    corrected[:3, :3] = R_imu_to_optical @ ext[:3, :3]  # 校正旋转部分
    corrected[:3, 3] = R_imu_to_optical @ ext[:3, 3]    # 校正平移部分
    return corrected


def triangulate(left_uv, right_uv, T_imu, extL, extR, K_l, K_r):
    extL = correct_extrinsic(inversePose(extL))
    extR = correct_extrinsic(inversePose(extR))
    
    # 构造左右相机的投影矩阵 P = K * [R | t]
    T_left = extL @ T_imu  # 世界到左相机的变换
    T_right = extR @ T_imu  # 世界到右相机的变换
    P_left = K_l @ T_left[:3, :]
    P_right = K_r @ T_right[:3, :]

    # 构建线性方程组 Ax=0
    A = []
    u_l, v_l = left_uv
    u_r, v_r = right_uv
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
    dataset = "dataset01"
    filename = f"./data/{dataset}/{dataset}.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(
        filename)

    mu_pose = np.eye(4)  # 初始位姿 SE(3) 4x4
    Sigma_pose = np.eye(6)*1e-6  # 位姿协方差 6x6

    # 噪声参数
    W = np.diag([0.1**2, 0.1**2, 0.1**2, 0.05**2, 0.05**2, 0.05**2])  # 过程噪声 6x6

    trajectory = [mu_pose.copy()]
    cov_log = [Sigma_pose.copy()]

    u_t = np.hstack((v_t, w_t)) # Tx6

    # 主循环
    for i in tqdm(range(len(timestamps)-1)):
        # (a) IMU预测步骤
        tau = timestamps[i+1] - timestamps[i]

        F = pose2adpose(axangle2pose(-u_t[i]*tau))  # df/du Ad(SE(3)) 6x6

        # 更新均值和协方差
        mu_pose = mu_pose @ axangle2pose(u_t[i]*tau)    # IMU在t时的世界坐标 SE(4) 4x4
        Sigma_pose = F @ Sigma_pose @ F.T + W*tau   # IMU在t时的位置covariance 6x6
        trajectory.append(mu_pose.copy())
        cov_log.append(Sigma_pose.copy())
    trajectory = np.array(trajectory)       # Tx4x4
    cov_log = np.array(cov_log)             # Tx6x6
    # visualize_trajectory_2d(trajectory, show_ori=False)
    # plt.show()

    # Part (b): Landmark Mapping via EKF Update
    num_landmarks = features.shape[1]
    landmark_means = np.zeros(num_landmarks*3)  # 3M
    landmark_covs = lil_matrix((num_landmarks*3, num_landmarks*3))    # 3Mx3M
    V = np.eye(4)  # 4x4
    invalid_landmarks = np.zeros_like(landmark_means, dtype=bool)
    valid_landmarks = np.ones_like(landmark_means, dtype=bool)

    for j in range(num_landmarks):
        start = 3*j
        end = start+3
        valid_frames = np.where(features[0, j, :] != -1)[0]
        landmark_covs[start:end,start:end] = np.diag([10, 10, 10]) 
        if len(valid_frames) == 0:
            continue
        t = valid_frames[0]
        left_uv = features[:2, j, t]  # 左相机像素坐标
        right_uv = features[2:, j, t]  # 右相机像素坐标
        T_imu = inversePose(trajectory[t])      # 当前IMU位姿
        landmark_means[start:end] = triangulate(
                left_uv, right_uv, T_imu, extL_T_imu, extR_T_imu, K_l, K_r)
        
        dist = np.linalg.norm(landmark_means[start:end-1] - trajectory[t,:2,3])
        if dist < 0.5 or dist > 50 or left_uv[1] < 300: 
            invalid_landmarks[start:end] = True
            valid_landmarks[start:end] = False
    # plt.scatter(landmark_means[valid_landmarks].reshape(-1,3)[:,0],landmark_means[valid_landmarks].reshape(-1,3)[:,1],s=1)
    # plt.plot(trajectory[:,0,3],trajectory[:,1,3],'r-')
    # plt.show()
    
    features = features[:, valid_landmarks.reshape(-1,3)[:,0], :]
    valid_landmark_means = landmark_means[valid_landmarks]
    landmark_covs = lil_matrix((features.shape[1]*3, features.shape[1]*3))    # 3Mx3M
    landmark_covs.setdiag(10)
    # features[:,invalid_landmarks,:] = -1
    for t in tqdm(range(len(timestamps)-1)):
        T_imu = trajectory[t]
        T_cam_left = correct_extrinsic(inversePose(extL_T_imu)) @ inversePose(T_imu)  # 左相机变换
        T_cam_right = correct_extrinsic(inversePose(extR_T_imu)) @ inversePose(T_imu) # 右相机变换
        
        for j in range(features.shape[1]):
            z_observed = features[:, j, t]
            if np.all(z_observed == -1):
                continue  # 无观测
            z_observed_left = features[:2, j, t]  # 左相机观测
            z_observed_right = features[2:, j, t] # 右相机观测
            
            # --- 预测左右相机的观测值 ---
            start = j*3
            end = start + 3
            m_j = valid_landmark_means[start:end]
            m_homo = np.append(m_j, 1.0)
            
            # 左相机投影
            q_cam_left = T_cam_left @ m_homo
            q_proj_left = projection(q_cam_left.reshape(1,4))[0]
            z_pred_left = K_l @ q_proj_left[:3]
            
            # 右相机投影
            q_cam_right = T_cam_right @ m_homo
            q_proj_right = projection(q_cam_right.reshape(1,4))[0]
            z_pred_right = K_r @ q_proj_right[:3]
            
            # --- 合并残差 ---
            residual_left = z_observed_left - z_pred_left[:2]
            residual_right = z_observed_right - z_pred_right[:2]
            residual = np.hstack([residual_left, residual_right])  # 4x1
            
            # --- 合并雅可比矩阵 ---
            H_left = compute_jacobian(T_cam_left, q_cam_left, K_l)  # 3x3
            H_right = compute_jacobian(T_cam_right, q_cam_right, K_r)  # 3x3
            H_j = np.vstack([H_left[:2, :], H_right[:2, :]])  # 4x3
            
            # --- 卡尔曼增益与协方差更新（噪声矩阵扩展为4x4）---
            Sigma_j = landmark_covs[start:end,start:end].toarray()
            S = H_j @ Sigma_j @ H_j.T + V
            K = Sigma_j @ H_j.T @ np.linalg.inv(S)
            
            # --- 更新地标状态 ---
            valid_landmark_means[start:end] += (K @ residual).flatten()
            I_KH = np.eye(3) - K @ H_j
            landmark_covs[start:end,start:end] = lil_matrix(I_KH @ Sigma_j)
        
    plt.scatter(valid_landmark_means.reshape(-1,3)[:,0],valid_landmark_means.reshape(-1,3)[:,1],label="Updated Landmarks",s=1,c='red')
    plt.scatter(landmark_means[valid_landmarks].reshape(-1,3)[:,0],landmark_means[valid_landmarks].reshape(-1,3)[:,1],label="Initial Landmarks",s=1,c='green')
    plt.plot(trajectory[:,0,3],trajectory[:,1,3],'b-')
    plt.legend()
    plt.show()
            
    num_landmarks = features.shape[1]
    # imu_pose: 使用6维轴角表示 (v, ω)
    joint_state = np.zeros(6 + num_landmarks * 3)
    joint_state[:6] = (trajectory[0])  # 初始位姿

    # 联合协方差矩阵 (6+3M)x(6+3M)
    joint_cov = lil_matrix((6 + num_landmarks*3, 6 + num_landmarks*3))
    joint_cov[:6, :6] = Sigma_pose  # IMU初始协方差
    for j in range(num_landmarks):
        start = 6 + j*3
        end = start + 3
        joint_cov[start:end, start:end] = np.eye(3)*10  # 地标初始协方差

    # 观测噪声矩阵 (4x4)
    V_obs = np.eye(4) * 0.1
        
    
    
    
    # You may use the function below to visualize the robot pose over time
    visualize_trajectory_2d(trajectory, show_ori=True)
