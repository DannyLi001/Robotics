import torch
import numpy as np
import pytorch_kinematics as pk
from scipy.spatial.transform import Rotation as R

class IKSolver:
    def __init__(self, urdf_path: str, end_link_name: str, ee_offset: float = 0.05):
        """
        Initializes the Inverse Kinematics solver.

        Args:
            urdf_path (str): Path to the URDF file.
            end_link_name (str): Name of the end-effector link in the URDF.
            ee_offset (float): A static offset from the end-effector link along the z-axis.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.end_link_name = end_link_name
        self.ee_offset = ee_offset

        # Build the kinematic chain from the URDF
        urdf_str = open(urdf_path, 'r').read()
        self.pk_chain = pk.build_serial_chain_from_urdf(
            urdf_str,
            end_link_name=self.end_link_name,
        ).to(device=self.device)

        # Apply the static end-effector offset if provided
        if self.ee_offset != 0.0:
            offset_link = pk.frame.Link(
                name=self.end_link_name,
                offset=pk.Transform3d(rot=None, pos=torch.tensor([0.0, 0.0, self.ee_offset], device=self.device), device=self.device)
            )
            self.pk_chain.add_link(parent=self.end_link_name, link=offset_link)
            self.end_link_name = offset_link.name

        self.lim = torch.tensor(self.pk_chain.get_joint_limits(), device=self.device)

        # Setup the Pseudo-Inverse IK solver
        self.pik = pk.PseudoInverseIK(
            self.pk_chain,
            num_retries=1,
            joint_limits=self.lim.T,
            max_iterations=200,
            early_stopping_any_converged=True
        )
        
    def solve(self, target_pose: np.ndarray, current_qpos: np.ndarray, is_delta_pose: bool = False):
        """
        Solves the Inverse Kinematics problem.

        Args:
            target_pose (np.ndarray): Target pose.
                - If is_delta_pose is False: Absolute pose [x, y, z, rx, ry, rz] (Euler angles).
                - If is_delta_pose is True: Delta pose [dx, dy, dz, drx, dry, drz] (Euler angles).
            current_qpos (np.ndarray): Current joint positions.
            is_delta_pose (bool): True if the input pose is a delta pose, False for an absolute pose.
        
        Returns:
            torch.Tensor: Solved joint positions or None if IK fails.
        """
        # Convert numpy arrays to torch tensors
        target_pose_tensor = torch.from_numpy(target_pose).float().to(self.device).unsqueeze(0)
        q0_tensor = torch.from_numpy(current_qpos).float().to(self.device).unsqueeze(0)

        # Convert euler angles to quaternion for pytorch-kinematics
        euler_angles = target_pose_tensor[:, 3:]
        rotation = R.from_euler('xyz', euler_angles.squeeze().cpu().numpy(), degrees=False)
        target_quat = torch.from_numpy(rotation.as_quat()).float().to(self.device)
        target_quat = target_quat[[3, 0, 1, 2]].unsqueeze(0) # Reorder to [w, x, y, z]

        if not is_delta_pose:
            # 1. Get current EE pose from forward kinematics
            current_pose_matrix = self.pk_chain.forward_kinematics(q0_tensor).get_matrix()
            current_pos = current_pose_matrix[:, :3, 3]
            current_rot_quat = pk.frame.RotationMatrix(current_pose_matrix[:, :3, :3]).as_quat()

            # 2. Compute delta pose
            delta_pos = target_pose_tensor[:, :3] - current_pos
            delta_rot_quat = pk.frame.RotationMatrix.from_quaternion(target_quat) * pk.frame.RotationMatrix.from_quaternion(current_rot_quat).inv()
            
            # Convert delta quaternion to delta Euler angles
            delta_rot_euler = torch.from_numpy(R.from_quat(delta_rot_quat.as_quat().squeeze().cpu().numpy()[[1, 2, 3, 0]]).as_euler('xyz', degrees=False)).float().to(self.device)
            
            # Combine to form delta pose tensor
            target_pose_tensor = torch.cat([delta_pos, delta_rot_euler.unsqueeze(0)], dim=1)
        
        # Now, target_pose_tensor is a delta pose in both cases, ready for the IK solver.
        
        # Solve the IK
        solved_q_values = self.pik.solve(target_pose_tensor, q_initial=q0_tensor)
        
        if solved_q_values.success:
            return solved_q_values.solution.squeeze(0)
        else:
            print("IK failed to find a solution.")
            return None

if __name__ == '__main__':
    ik_solver = IKSolver(
        urdf_path="./assets/urdfs/fr3_franka_hand.urdf", 
        end_link_name="fr3_hand_tcp",
        ee_offset=0.05
    )
    pass