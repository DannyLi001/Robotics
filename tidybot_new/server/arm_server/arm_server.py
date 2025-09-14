import numpy as np
import time
import queue
from scipy.spatial.transform import Rotation
from multiprocessing.managers import BaseManager
from .arm_controller import FrankaArm, RobotMode
from ...config.config import get_arm_constants, get_rpc_constants
from ...utils.ik_solver import IKSolver

POLICY_CONTROL_PERIOD = 0.1

# Load arm and RPC configuration
ARM_CFG = get_arm_constants()
RPC_CFG = get_rpc_constants()

# Extract RPC settings to match homer's constants
ARM_RPC_HOST = RPC_CFG.get("ARM_RPC_HOST", "127.0.0.1")
ARM_RPC_PORT = int(RPC_CFG.get("ARM_RPC_PORT", 10001))
authkey_str = RPC_CFG.get("RPC_AUTHKEY", "secrete password")
RPC_AUTHKEY = authkey_str.encode() if isinstance(authkey_str, str) else authkey_str

class Arm:
    def __init__(self, control_mode):
        if control_mode is RobotMode.JOINT_ANGLE:
            print("Using joint angle control")
        elif control_mode is RobotMode.CARTESIAN_POS:
            print("Using cartesian pose control")
        elif control_mode is RobotMode.JOINT_VEL:
            print("Using joint velocity control")
        elif control_mode is RobotMode.CARTESIAN_VEL:
            print("Using cartesian velocity control")
        else:
            raise ValueError(f"Invalid joint control mode: {control_mode}")

        arm_ip = ARM_CFG.get("ip", "172.16.0.2")
        self.control_mode = control_mode
        self.arm = FrankaArm(arm_ip, control_mode=self.control_mode)
        self.command_queue = queue.Queue(1)
        
        # Set the command queue for the arm controller
        self.arm.set_command_queue(self.command_queue)
        
        if self.control_mode is RobotMode.JOINT_ANGLE:
            self.ik_solver = IKSolver(urdf_path=self.arm.urdf_path, end_link_name=self.arm.ee_link_name, ee_offset=0.0)
        

    def reset(self):
        # Initialize Franka arm        
        if self.arm.cyclic_running:
            time.sleep(0.75)
            self.arm.stop_cyclic()
        self.arm.reset()
        # Start control loop thread
        self.arm.init_cyclic()
        while not self.arm.cyclic_running:
            time.sleep(0.01)
        print("Franka arm reset complete")

    def execute_action(self, action):
        if action['gripper_pos'] is None:
            action['gripper_pos'] = 0.0
        if action['grasp'] is None:
            action['grasp'] = True

        if self.control_mode is RobotMode.JOINT_ANGLE:
            if 'arm_qpos' in action:
                qpos = action['arm_qpos']
            elif 'arm_pos' in action and 'arm_euler' in action:
                current_qpos = self.get_qpos()
                target_pose = np.concatenate([action['arm_pos'], action['arm_euler']])
                qpos = self.ik_solver.solve(target_pose, is_delta_pose=False, current_pose=current_qpos)
            elif 'arm_delta_pos' in action and 'arm_delta_euler' in action:
                target_pose = np.concatenate([action['arm_delta_pos'], action['arm_delta_euler']])
                qpos = self.ik_solver.solve(target_pose, is_delta_pose=True)
            else:
                raise ValueError("Action must contain either 'arm_qpos' or both 'arm_pos' and 'arm_euler' in JOINT_ANGLE mode")
            print(f"Executing action: {qpos}")
            self.command_queue.put((qpos, action['gripper_pos'].item(), action['grasp']))

        elif self.control_mode is RobotMode.CARTESIAN_POS:
            if 'arm_pos' in action and 'arm_euler' in action:
                quat = Rotation.from_euler("xyz", action['arm_euler']).as_quat()
                pose = np.concatenate([action['arm_pos'], quat])
            elif 'arm_delta_pos' in action and 'arm_delta_euler' in action:
                current_pose = self.arm.get_ee_pose()
                target_pos = current_pose[:3] + action['arm_delta_pos']
                target_euler = current_pose[3:] + action['arm_delta_euler']
                quat = Rotation.from_euler("xyz", target_euler).as_quat()
                pose = np.concatenate([target_pos, quat])
            else:
                raise ValueError("Action must contain either 'arm_pos' and 'arm_euler' or 'arm_delta_pos' and 'arm_delta_euler' in CARTESIAN_POS mode")
            self.command_queue.put((pose, action['gripper_pos'].item(), action['grasp']))

        elif self.control_mode is RobotMode.JOINT_VEL:
            if 'arm_qvel' in action:
                qvel = action['arm_qvel']
            else:
                raise ValueError("Action must contain 'arm_qvel' in JOINT_VEL mode")
            self.command_queue.put((qvel, action['gripper_pos'].item(), action['grasp']))

        elif self.control_mode is RobotMode.CARTESIAN_VEL:
            if 'arm_twist' in action:
                twist = action['arm_twist']
            else:
                raise ValueError("Action must contain 'arm_twist' in CARTESIAN_VEL mode")
            self.command_queue.put((twist, action['gripper_pos'].item(), action['grasp']))
            
        else:
            raise ValueError(f"Invalid joint control mode: {self.control_mode}")



    def get_qpos(self) -> np.ndarray:
        return self.arm.get_qpos()

    def get_state(self) -> dict:
        return self.arm.get_state()

    def close(self):
        if self.arm.cyclic_running:
            time.sleep(0.75) 
            self.arm.stop_cyclic()


class ArmManager(BaseManager):
    pass

ArmManager.register('Arm', Arm)

if __name__ == '__main__':
    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Franka arm manager server started at {ARM_RPC_HOST}:{ARM_RPC_PORT}')
    server.serve_forever()