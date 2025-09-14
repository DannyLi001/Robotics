import time
import numpy as np
import queue
import threading
from scipy.spatial.transform import Rotation
from ...config.config import get_arm_constants
from enum import Enum

try:
    from franky import Robot, Gripper, JointMotion, JointState, JointVelocityMotion, ControlException, RelativeDynamicsFactor, \
                       Affine, CartesianMotion, CartesianVelocityMotion, RobotVelocity, Twist, Duration
except ImportError:
    print("Warning: franky library not found. Please install franky for Franka arm control.")
    # Define dummy classes for linting purposes
    class Robot: pass
    class Gripper: pass
    class JointMotion: pass
    class JointState: pass
    class JointVelocityMotion: pass
    class ControlException: pass
    class RelativeDynamicsFactor: pass
    class Affine: pass
    class CartesianMotion: pass
    class CartesianVelocityMotion: pass
    class RobotVelocity: pass
    class Twist: pass
    class Duration: pass


class RobotMode(Enum):
    JOINT_ANGLE = 1
    JOINT_VEL = 2
    CARTESIAN_POS = 3
    CARTESIAN_VEL = 4


POLICY_CONTROL_PERIOD = 0.1

class FrankaArm:
    def __init__(self, arm_ip: str, control_mode: RobotMode = RobotMode.JOINT_ANGLE):
        cfg = get_arm_constants()
        
        # Basic setup
        self.robot = Robot(arm_ip)
        self.gripper = Gripper(arm_ip)
        
        # Load config
        self.home_qpos = cfg["home_qpos"]
        # self.home_pose = cfg["home_pose"]
        self.urdf_path = cfg["urdf_path"]
        self.ee_link_name = cfg["ee_link_name"]
        self.joint_limits = [(float(a), float(b)) for a, b in cfg["joint_limits"]]
        self.joint_velocity_limit = np.array(cfg["joint_velocity_limit"], dtype=np.float32)
        self.joint_acceleration_limit = np.array(cfg["joint_acceleration_limit"], dtype=np.float32)
        self.joint_jerk_limit = np.array(cfg["joint_jerk_limit"], dtype=np.float32)
        self.velocity_factor = cfg["velocity_factor"]
        self.acceleration_factor = cfg["acceleration_factor"]
        self.jerk_factor = cfg["jerk_factor"]
        self.control_mode = control_mode

        # Setup robot
        # print(self.robot.joint_velocity_limit)
        # print(self.robot.joint_acceleration_limit)
        # print(self.robot.joint_jerk_limit)
        self.robot.recover_from_errors()
        self.robot.joint_velocity_limit.set(self.joint_velocity_limit)
        self.robot.joint_acceleration_limit.set(self.joint_acceleration_limit)
        self.robot.joint_jerk_limit.set(self.joint_jerk_limit)
        self.robot.relative_dynamics_factor = RelativeDynamicsFactor(velocity=self.velocity_factor, acceleration=self.acceleration_factor, jerk=self.jerk_factor)

        # Cyclic thread setup
        self.cyclic_thread = None
        self.kill_the_thread = False
        self.cyclic_running = False

        # Action input setup (will be set by external server if used)
        self.command_queue = None
        self.target_qpos = None
        self.target_gripper_pos = None
        self.last_command_time = None
        self.executing_motion = False
        self.last_gripper_pos = np.array([0.0, 0.0])

    def execute_ee_pose(self, target):
        if self.control_mode is RobotMode.JOINT_ANGLE:
            motion = JointMotion(JointState(target))
            try:
                self.robot.move(motion, asynchronous=True)
                print(f"Moving to joint positions: {np.round(target, 3)}")
            except ControlException as e:
                print(f"Control error: {e}")
                self.robot.recover_from_errors()

        elif self.control_mode is RobotMode.CARTESIAN_POS:
            rotation = Rotation.from_quat(target[-4:])
            affine_pose = Affine(target[:3], rotation.as_quat())
            motion = CartesianMotion(affine_pose)
            try:
                self.robot.move(motion, asynchronous=True)
                print(f"Moving to cartesian pose: {np.round(target[:3], 3)}")
            except ControlException as e:
                print(f"Control error: {e}")
                self.robot.recover_from_errors()

        elif self.control_mode is RobotMode.JOINT_VEL:
            motion = JointVelocityMotion(target, duration=Duration(1000 * POLICY_CONTROL_PERIOD))
            try:
                self.robot.move(motion, asynchronous=True)
                print(f"Moving with joint velocities: {np.round(target, 3)}")
            except ControlException as e:
                print(f"Control error: {e}")
                self.robot.recover_from_errors()

        elif self.control_mode is RobotMode.CARTESIAN_VEL:
            target_twist = Twist(target)
            motion = CartesianVelocityMotion(RobotVelocity(target_twist, elbow_velocity=-0.2))
            try:
                self.robot.move(motion, asynchronous=True)
                print(f"Moving with Cartesian velocity: {target_twist}")
            except ControlException as e:
                print(f"Control error: {e}")
                self.robot.recover_from_errors()
        else:
            raise ValueError(f"Invalid joint control mode: {self.control_mode}") 

    def execute_gripper(self, width: float, grasp: bool = False):
        if grasp:
            self.gripper.grasp(0.0, speed=0.05, force=50.0)
        else:
            self.gripper.move(width, speed=0.05)
        print(f"Gripper width: {self.gripper.width}")
    
    def reset(self):
        """Reset to home position."""
        print("Resetting to home position...")
        self.robot.recover_from_errors()
        
        motion = JointMotion(JointState(self.home_qpos))
        self.robot.move(motion, asynchronous=False) 
        self.gripper.grasp(0.0, speed=0.05, force=50.0)
        print("Reset complete.")
    
    def get_qpos(self) -> np.ndarray:
        """Get current joint positions."""
        return self.robot.current_joint_state.position

    def get_qvel(self) -> np.ndarray:
        """Get current joint velocities."""
        return self.robot.current_joint_state.velocity

    def get_ee_pose(self) -> np.ndarray:
        """
        Get current end-effector pose as [position, euler_angles].
        
        Returns:
            np.ndarray: A 6D array containing [x, y, z, roll, pitch, yaw] in radians.
        """
        cartesian_state = self.robot.current_cartesian_state
        ee_pose_matrix = cartesian_state.pose.end_effector_pose.matrix
        
        position = ee_pose_matrix[:3, 3]
        rotation = Rotation.from_matrix(ee_pose_matrix[:3, :3])
        
        euler_angles = rotation.as_euler('xyz', degrees=False)
    
        return np.concatenate([position, euler_angles])
    
    def get_state(self) -> dict:
        # Get current state
        cartesian_state = self.robot.current_cartesian_state
        ee_pose_matrix = cartesian_state.pose.end_effector_pose.matrix
        position = ee_pose_matrix[:3, 3]
        rotation = Rotation.from_matrix(ee_pose_matrix[:3, :3])
        quat = rotation.as_quat()  # [x,y,z,w]
        
        # Get gripper width
        gripper_width = self.gripper.width
        gripper_pos = np.array([gripper_width/2, gripper_width/2])
        gripper_vel = (gripper_pos - self.last_gripper_pos) / POLICY_CONTROL_PERIOD
        self.last_gripper_pos = gripper_pos.copy()
        
        state = {
            'arm_qpos': self.get_qpos(),
            'arm_qvel': self.get_qvel(),
            'arm_pos': position,
            'arm_quat': quat,
            'gripper_pos': gripper_pos,
            'gripper_vel': gripper_vel,
        }
        return state
            
    def _stop(self):
        self.robot.stop()

    def set_command_queue(self, command_queue):
        """Set the command queue for external control."""
        self.command_queue = command_queue

    def init_cyclic(self):
        """Initialize cyclic control mode for Franka arm."""
        assert not self.cyclic_running, 'Cyclic thread is already running'

        print('Franka arm is ready for cyclic control')

        # Start cyclic thread
        self.kill_the_thread = False
        self.cyclic_thread = threading.Thread(target=self._run_cyclic, daemon=True)
        self.cyclic_thread.start()

    def _run_cyclic(self):
        """Simplified cyclic loop: only process command queue."""
        self.cyclic_running = True
        print('Franka arm cyclic control started')
        
        while not self.kill_the_thread:
            t_start = time.time()
            if self.command_queue is not None and not self.command_queue.empty():
                try:
                    # Handle both 2-tuple and 3-tuple formats for backward compatibility
                    command = self.command_queue.get_nowait()
                    target, gripper_pos, grasp = command
                    
                    self.last_command_time = time.time()
                    if self.control_mode is RobotMode.JOINT_ANGLE:
                        print(f"Received command: qpos={np.round(target, 3) if target is not None else None}, gripper={gripper_pos}, grasp={grasp}")
                    elif self.control_mode is RobotMode.CARTESIAN_POS:
                        print(f"Received command: pose={np.round(target, 3) if target is not None else None}, gripper={gripper_pos}, grasp={grasp}")
                    elif self.control_mode is RobotMode.JOINT_VEL:
                        print(f"Received command: qvel={np.round(target, 3) if target is not None else None}, gripper={gripper_pos}, grasp={grasp}")
                    elif self.control_mode is RobotMode.CARTESIAN_VEL:
                        print(f"Received command: twist={np.round(target, 3) if target is not None else None}, gripper={gripper_pos}, grasp={grasp}")
                    else:
                        print(f"Invalid joint control mode: {self.control_mode}")
                        
                    # Execute joint motion
                    if target is not None:
                        self.execute_ee_pose(target)
                    
                    # Execute gripper command
                    if grasp is True:
                        self.execute_gripper(0.0, grasp=True)
                    elif gripper_pos is not None:
                        self.execute_gripper(width=gripper_pos, grasp=False)
                        
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f"Command execution error: {e}")
                    self.robot.recover_from_errors()

            elapsed = time.time() - t_start
            remaining = POLICY_CONTROL_PERIOD
            if remaining > 0:
                time.sleep(remaining)
            else:
                print(f'Warning: Step time {1000 * elapsed:.3f} ms in {self.__class__.__name__} run_cyclic')

        self.cyclic_running = False
        print('Franka arm cyclic control stopped')

    def stop_cyclic(self):
        """Stop cyclic control mode for Franka arm."""
        # Kill cyclic thread
        if self.cyclic_running:
            self.kill_the_thread = True
            self.cyclic_thread.join()

        # Stop robot motion
        try:
            self.robot._stop()
        except Exception as e:
            print(f'Error stopping robot: {e}')

        print('Franka arm cyclic control stopped')

if __name__ == '__main__':
    # arm = FrankaArm('172.16.0.2')
    # arm.reset()
    # arm.execute_ee_pose(np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785]))
    # arm.execute_gripper(0.05)
    # arm.get_qpos()
    # arm.get_qvel()
    # arm.get_ee_pose()
    # arm.get_state()
    # arm.set_command_queue(None)
    pass
    # arm.init_cyclic()
    # while not arm.cyclic_running:
    #     time.sleep(0.01)
    # pass
    # arm.stop_cyclic()

