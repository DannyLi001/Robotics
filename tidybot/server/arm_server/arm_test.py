import time
from multiprocessing.managers import BaseManager
from ...config.config import get_rpc_classes
import numpy as np
from .arm_controller import RobotMode


POLICY_CONTROL_PERIOD = 0.1
ARM_RPC_CFG = get_rpc_classes()['arm']

ARM_RPC_HOST = ARM_RPC_CFG.host
ARM_RPC_PORT = ARM_RPC_CFG.port
authkey_str = ARM_RPC_CFG.authkey
RPC_AUTHKEY = authkey_str.encode() if isinstance(authkey_str, str) else authkey_str


class ArmManager(BaseManager):
    pass

ArmManager.register('Arm')

if __name__ == '__main__':
    arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    arm_manager.connect()
    arm = arm_manager.Arm('main')

    # arm_pos = np.array([343, 449, 330]) / 1000
    # arm_euler = np.deg2rad(np.array([-131, 6.8, 33.9]))
    gripper_pos = np.zeros(1)
    arm_qvel = np.ones(7) * 0.01

    try:
        arm.reset()
        while True:
            time.sleep(POLICY_CONTROL_PERIOD) 
            arm.execute_action({'arm_qvel': arm_qvel})
            # arm.execute_action({'arm_qvel': arm_qvel,'gripper_pos': gripper_pos,'grasp': True})
            # err = np.linalg.norm(arm_pos - arm.get_state()['arm_pos'])
            # print(f"q: {[round(x, 3) for x in arm.get_state()['arm_pos']]}, euler:{[round(x,3) for x in np.rad2deg(arm.get_state()['arm_euler'])]}")
            print(arm.get_state())

    finally:
        arm.close()
