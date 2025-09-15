import os
import yaml
import numpy as np
from typing import Any, Dict
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
    type: str  # "kinova", "realsense", "logitech"
    serial: str
    frame_width: int
    frame_height: int
    fps: int

@dataclass
class BaseConfig:
    """Base configuration"""
    h_x: list[float]
    h_y: list[float]
    encoder_magnet_offsets: list[float]
    control_freq: int
    control_period: float
    num_casters: int

@dataclass
class ArmConfig:
    """Arm configuration"""
    type: str
    ip: str
    urdf_path: str
    ee_link_name: str
    
    # Joint configuration
    home_qpos: list[float]
    joint_velocity_limit: list[float]
    joint_acceleration_limit: list[float]
    joint_jerk_limit: list[int]
    
    # franky control factors
    velocity_factor: float
    acceleration_factor: float
    jerk_factor: float
    control_mode: int
    
    # Joint limits [min, max] for each joint
    joint_limits: list[list[float]]
    
@dataclass
class RPCConfig:
    """RPC configuration"""
    host: str
    port: int
    authkey: str
    
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

    config_path = _default_config_path()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {config_path}: {e}")
    
    return _CONFIG_CACHE

def _load_base_config() -> Dict[str, Any]:
    """Get constants configuration"""
    config = _load_config()
    return config.get("base", {})

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
def get_base_class() -> Dict[str, BaseConfig]:
    """Get base constants"""
    base_const = _load_base_config()
    base = BaseConfig(
        h_x=base_const.get("h_x", [0.0, 0.0, 0.0, 0.0]),
        h_y=base_const.get("h_y", [0.0, 0.0, 0.0, 0.0]),
        encoder_magnet_offsets=base_const.get("encoder_magnet_offsets", [0.0, 0.0, 0.0, 0.0]),
        control_freq=base_const.get("control_freq", 250),
        control_period=base_const.get("control_period", 0.004),
        num_casters=base_const.get("num_casters", 4)
    )
    return {"base": base}

def get_camera_classes() -> Dict[str, CameraConfig]:
    """Get camera configurations as CameraConfig instances"""
    camera_hw = _load_camera_config()
    camera_classes = {}
    for cam_name, cam_info in camera_hw.items():
        camera_classes[cam_name] = CameraConfig(
            type=cam_info.get("type", "unknown"),
            serial=cam_info.get("serial", ""),
            frame_width=cam_info.get("frame_width", 640),
            frame_height=cam_info.get("frame_height", 480),
            fps=cam_info.get("fps", 30)
        )
    
    return camera_classes

def get_rpc_classes() -> Dict[str, RPCConfig]:
    """Get RPC constants"""
    rpc_config = _load_rpc_config()
    
    rpc_classes = {}
    for rpc_name, rpc_info in rpc_config.items():
        rpc_classes[rpc_name] = RPCConfig(
            host=rpc_info.get("host", "localhost"),
            port=rpc_info.get("port", 2000),
            authkey=rpc_info.get("authkey", "secret password")
        )
    
    return rpc_classes

def get_arm_classes() -> Dict[str, Any]:
    """Get arm constants"""
    arm_config = _load_arm_config()
    
    arm_classes = {}
    for arm_name, arm_info in arm_config.items():
        arm_classes[arm_name] = ArmConfig(
            type=arm_info.get("type", "franka"),
            ip=arm_info.get("ip", "172.16.0.2"),
            urdf_path=arm_info.get("urdf_path", "./assets/franka.urdf"),
            ee_link_name=arm_info.get("ee_link_name", "panda_hand_tcp"),
            home_qpos=arm_info.get("home_qpos", [0.0, 0.0, 0.0, -1.6, 0.0, 1.6, 0.8]),
            joint_velocity_limit=arm_info.get("joint_velocity_limit", [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
            joint_acceleration_limit=arm_info.get("joint_acceleration_limit", [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]),
            joint_jerk_limit=arm_info.get("joint_jerk_limit", [7500, 3750, 5000, 6250, 7500, 10000, 10000]),
            velocity_factor=arm_info.get("velocity_factor",0.05),
            acceleration_factor=arm_info.get("acceleration_factor", 0.01),
            jerk_factor=arm_info.get("jerk_factor", 0.1),
            control_mode=arm_info.get("control_mode", 1),
            joint_limits=arm_info.get("joint_limits", None),
        )
        
    return arm_classes

# print(get_base_class())
# print(get_camera_classes())
# print(get_rpc_classes())
# print(get_arm_classes())

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

# print(get_calibration_matrices("./config/calib_files"))

################################################################################
# Convenience Functions
################################################################################

if __name__ == "__main__":
    pass
    # print(get_camera_constants())