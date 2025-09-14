import os
import yaml
import numpy as np
from typing import Any, Dict, Optional
from dataclasses import dataclass
import pickle

# Global configuration cache
_CONFIG_CACHE: Dict[str, Any] | None = None

################################################################################
# Configuration Data Classes
################################################################################

@dataclass
class CameraConfig:
    """Camera configuration"""
    name: str
    type: str  # "kinova", "realsense", "logitech"
    serial: Optional[str] = None
    frame_width: Optional[int] = 640
    frame_height: Optional[int] = 480
    fps: Optional[int] = 30

################################################################################
# Configuration Loading Functions
################################################################################

def _default_config_path() -> str:
    """Get default config path"""
    return os.environ.get(
        "TIDYBOT_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "config.yaml"),
    )

def _load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = os.path.abspath(_default_config_path())
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {config_path}: {e}")
    
    return _CONFIG_CACHE

def _load_constants_config() -> Dict[str, Any]:
    """Get constants configuration"""
    config = _load_config()
    return config.get("constants", {})

def _load_camera_config() -> Dict[str, Any]:
    """Get camera configuration"""
    config = _load_config()
    return config.get("cameras", {})

def _load_arm_config() -> Dict[str, Any]:
    """Get arm configuration"""
    config = _load_config()
    return config.get("arm", {})

def _load_rpc_config() -> Dict[str, Any]:
    """Get RPC configuration"""
    config = _load_config()
    return config.get("rpc", {})

################################################################################
# Constants - dynamically loaded from YAML
################################################################################

def get_vehicle_constants() -> tuple[np.ndarray, np.ndarray]:
    """Get vehicle constants"""
    constants = _load_constants_config()
    vehicle = constants.get("vehicle", {})
    h_x = np.array(vehicle.get("h_x", [0.0, 0.0, 0.0, 0.0]))
    h_y = np.array(vehicle.get("h_y", [0.0, 0.0, 0.0, 0.0]))
    return h_x, h_y

def get_encoder_magnet_offsets() -> list[float]:
    """Get encoder magnet offsets"""
    constants = _load_constants_config()
    return constants.get("encoder_magnet_offsets", [0.0, 0.0, 0.0, 0.0])

def get_control_constants() -> Dict[str, Any]:
    """Get control constants"""
    constants = _load_constants_config()
    return {
        "control_freq": constants.get("control_freq", 250),
        "control_period": constants.get("control_period", 0.004),
        "num_casters": constants.get("num_casters", 4)
    }

def get_camera_constants() -> Dict[str, Any]:
    """Get camera constants"""
    camera_hw = _load_camera_config()
    left_cam = camera_hw.get("left", {})
    right_cam = camera_hw.get("right", {})
    
    return {
        "LEFT_CAMERA_SERIAL": left_cam.get("serial", ""),
        "RIGHT_CAMERA_SERIAL": right_cam.get("serial", ""),
        "LEFT_CAMERA_FRAME_WIDTH": left_cam.get("frame_width", 640),
        "LEFT_CAMERA_FRAME_HEIGHT": left_cam.get("frame_height", 480),
        "RIGHT_CAMERA_FRAME_WIDTH": right_cam.get("frame_width", 640),
        "RIGHT_CAMERA_FRAME_HEIGHT": right_cam.get("frame_height", 480),
        "LEFT_CAMERA_FPS": left_cam.get("fps", 30),
        "RIGHT_CAMERA_FPS": right_cam.get("fps", 30),
    }

def get_camera_classes() -> Dict[str, CameraConfig]:
    """Get camera configurations as CameraConfig instances"""
    camera_hw = _load_camera_config()
    camera_classes = {}
    
    for cam_name, cam_info in camera_hw.items():
        camera_classes[cam_name] = CameraConfig(
            name=cam_info.get("name", cam_name),
            type=cam_info.get("type", "unknown"),
            serial=cam_info.get("serial"),
            frame_width=cam_info.get("frame_width", 640),
            frame_height=cam_info.get("frame_height", 480),
            fps=cam_info.get("fps", 30)
        )
    
    return camera_classes

def get_rpc_constants() -> Dict[str, Any]:
    """Get RPC constants"""
    config = _load_rpc_config()
    rpc = config.get("rpc", {})
    
    base_rpc = rpc.get("base", {})
    arm_rpc = rpc.get("arm", {})
    server_rpc = rpc.get("server", {})
    
    return {
        "BASE_RPC_HOST": base_rpc.get("host", "localhost"),
        "BASE_RPC_PORT": base_rpc.get("port", 10000),
        "ARM_RPC_HOST": arm_rpc.get("host", "localhost"),
        "ARM_RPC_PORT": arm_rpc.get("port", 10001),
        "RPC_AUTHKEY": base_rpc.get("authkey", "secret password").encode(),
        "SERVER_HOST": server_rpc.get("host", "localhost"),
        "SERVER_PORT": server_rpc.get("port", 5555),
    }

def get_arm_constants() -> Dict[str, Any]:
    """Get arm constants"""
    arm_cfg = _load_arm_config()
    return {
        "ip": arm_cfg.get("ip", "172.16.0.2"),
        "urdf_path": arm_cfg.get("urdf_path", "./assets/franka.urdf"),
        "ee_link_name": arm_cfg.get("ee_link_name", "panda_hand_tcp"),
        "policy_frequency": arm_cfg.get("policy_frequency", 20),
        "observe": arm_cfg.get("observe", False),
        "joint_limits": arm_cfg.get("joint_limits", []),
        "joint_velocity_limit": arm_cfg.get("joint_velocity_limit", [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
        "joint_acceleration_limit": arm_cfg.get("joint_acceleration_limit", [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]),
        "joint_jerk_limit": arm_cfg.get("joint_jerk_limit", [7500, 3750, 5000, 6250, 7500, 10000, 10000]),
        "velocity_factor": arm_cfg.get("velocity_factor", 1.0),
        "acceleration_factor": arm_cfg.get("acceleration_factor", 1.0),
        "jerk_factor": arm_cfg.get("jerk_factor", 1.0),
        "workspace_limits": arm_cfg.get("workspace_limits", {}),
        "home_qpos": arm_cfg.get("home_qpos", [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]),
    }

# print(get_vehicle_constants())
# print(get_encoder_magnet_offsets())
# print(get_control_constants())
# print(get_camera_constants())
# print(get_camera_classes())
# print(get_rpc_constants())
# print(get_arm_constants())

################################################################################
# Configuration Loading Functions
################################################################################

def get_calibration_matrices(calib_dir) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load calibration matrices from pickle files"""
    intrinsics = {}
    extrinsics = {}

    calib_dir = os.path.abspath(calib_dir)
    if not os.path.exists(calib_dir):
        print(f"Calibration directory does not exist: {calib_dir}")
        return intrinsics, extrinsics

    for filename in os.listdir(calib_dir):
        if not filename.endswith('.pkl'):
            continue

        filepath = os.path.join(calib_dir, filename)
        try:
            if '_intrinsics.pkl' in filename:
                view = filename.replace('_intrinsics.pkl', '')
                with open(filepath, 'rb') as f:
                    mat = pickle.load(f)['matrix']
                    intrinsics[view] = mat.tolist() if isinstance(mat, np.ndarray) else mat
            elif '_extrinsics.pkl' in filename:
                view = filename.replace('_extrinsics.pkl', '')
                with open(filepath, 'rb') as f:
                    mat = pickle.load(f)
                    extrinsics[view] = mat.tolist() if isinstance(mat, np.ndarray) else mat
        except Exception as e:
            print(f"Error loading calibration file {filepath}: {e}")

    return intrinsics, extrinsics

################################################################################
# Convenience Functions
################################################################################

if __name__ == "__main__":
    pass
    # print(get_camera_constants())