import time
from multiprocessing.managers import BaseManager
from constants import BASE_RPC_HOST, BASE_RPC_PORT, RPC_AUTHKEY
import numpy as np
from constants import POLICY_CONTROL_PERIOD

class WheelManager(BaseManager):
    pass

if __name__ == '__main__':
    WheelManager.register('Base')

    base_manager = WheelManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
    base_manager.connect()
    base = base_manager.Base()
    try:
        base.reset()
        for i in range(50):
            base.execute_action({'base_pose': np.array([-(i / 50) * 0.5, 0.0, 0.0])})
            print(f"base pos: {base.get_state()['base_pose']}")
            time.sleep(POLICY_CONTROL_PERIOD)
    finally:
        base.close()