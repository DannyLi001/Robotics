from ..config.config import get_rpc_classes, get_camera_classes
from ..server.arm_server.arm_server import ArmManager
from ..server.base_server.base_server import BaseManager
import socket
import json
import sys
import os
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

RPC_CFG = get_rpc_classes()

BASE_RPC_HOST = RPC_CFG['base'].host
BASE_RPC_PORT = RPC_CFG['base'].port
authkey_str = RPC_CFG['base'].authkey
BASE_RPC_AUTHKEY = authkey_str.encode() if isinstance(authkey_str, str) else authkey_str

ARM_RPC_HOST = RPC_CFG['arm'].host
ARM_RPC_PORT = RPC_CFG['arm'].port
authkey_str = RPC_CFG['arm'].authkey
ARM_RPC_AUTHKEY = authkey_str.encode() if isinstance(authkey_str, str) else authkey_str

SERVER_RPC_HOST = RPC_CFG['server'].host
SERVER_RPC_PORT = RPC_CFG['server'].port
authkey_str = RPC_CFG['server'].authkey
SERVER_RPC_AUTHKEY = authkey_str.encode() if isinstance(authkey_str, str) else authkey_str

class Robot:
    def __init__(self, base_manager, arm_manager, name="main"):
        self.base = base_manager.Base()
        self.arm = arm_manager.Arm(name)
        
        self.cameras = get_camera_classes()

    def get_state_obs(self) -> dict:
        obs = dict()
        obs.update(self.base.get_state())
        obs.update(self.arm.get_state())
        return obs

    # def get_cam_obs(self):
    #     obs = dict()
    #     for name, cam in self.cameras.items():
    #         if self.use_depth:
    #             rgb_image, depth_image = cam.get_rgbd()
    #             obs.update({f"{name}_image": rgb_image, f"{name}_depth": depth_image})
    #         else:
    #             obs.update({f"{name}_image": cam.get_image()})
    #     return obs

    # def get_obs(self):
    #     obs = self.get_state_obs()
    #     obs.update(self.get_cam_obs())
    #     return obs

    def reset(self):
        print('Resetting base...')
        self.base.reset()

        print('Resetting arm...')
        self.arm.reset()

        print('Robot has been reset')

    def step(self, action):
        self.base.execute_action(action['base'])
        self.arm.execute_action(action['arm'])
        
    def step_arm_only(self, action):
        self.arm.execute_action(action)

    def step_base_only(self, action):
        self.base.execute_action(action)

    def set_gripper(self, width):
        obs = self.get_obs()
        pos = obs['arm_pos']
        euler = R.from_quat(obs['arm_quat']).as_euler('xyz')
        action = {
            'arm_pos': pos,
            'arm_euler': euler,
            'gripper_pos': np.array([width]),
            'grasp': False
        }
        self.arm.execute_action(action)

    def close(self):
        self.base.close()
        self.arm.close()
        # for cam in self.cameras.values():
        #     cam.close()

def start_server(host="0.0.0.0", port=5555) -> socket.socket:
    """Creates, binds, and listens on a socket for incoming connections."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[SERVER] Listening on {host}:{port}")
    return server


def serialize_state(state):
    """Convert numpy arrays to JSON-serializable dict"""
    serializable = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    return serializable

def connect_to_server() -> tuple:
    # RPC server connection for base
    base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=BASE_RPC_AUTHKEY)
    try:
        base_manager.connect()
    except ConnectionRefusedError as e:
        raise Exception(f'Could not connect to base RPC server, is base_server.py running? {e}\n [IMPORTANT]: sudo netplan apply') from e

    # RPC server connection for arm
    arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=ARM_RPC_AUTHKEY)
    try:
        arm_manager.connect()
    except ConnectionRefusedError as e:
        raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e
    return base_manager, arm_manager


def main():
    base_manager, arm_manager = connect_to_server()
    robot = Robot(base_manager, arm_manager)
    # The start_server function now returns a listening socket, not a connection.
    server_sock = start_server(host=SERVER_RPC_HOST, port=SERVER_RPC_PORT)

    try:
        # Outer loop: continuously wait for and accept new client connections
        while True:
            print("[SERVER] Waiting for a new connection...")
            conn, addr = server_sock.accept()
            print(f"[SERVER] Connected by {addr}")

            # Inner loop: handle communication with the current client
            try:
                while True:
                    print("[SERVER] Waiting for command...")
                    data = conn.recv(4096).decode("utf-8").strip()

                    if not data:
                        print("[SERVER] Connection closed by client.")
                        break  # Break inner loop to go back to accepting a new connection

                    try:
                        command = json.loads(data)
                        print(f"[SERVER] Received command: {command}")

                        if "get_pos" in command:
                            state = robot.get_state_obs()
                            state_json = json.dumps(serialize_state(state)) + "\n"
                            conn.sendall(state_json.encode("utf-8"))
                            print("[SERVER] Sent current state.")
                        if "arm" in command:
                            # robot.step_arm_only(command["arm"])
                            print("[SERVER] Executed 'arm' action.")
                        if "base" in command:
                            # robot.step_base_only(command["base"])
                            print("[SERVER] Executed 'base' action.")
                        if "get_pos" not in command and "arm" not in command and "base" not in command:
                            print("[SERVER] No valid command in received data.")

                    except json.JSONDecodeError:
                        print("[SERVER] Invalid JSON received.")

            except ConnectionResetError:
                print("[SERVER] Client disconnected unexpectedly.")
            finally:
                conn.close()
                print("[SERVER] Connection closed.")
    except KeyboardInterrupt:
        print("[SERVER] Stopped by user")
    finally:
        robot.close()
        server_sock.close()


if __name__ == "__main__":
    main()