import time
from multiprocessing.managers import BaseManager
from ...config.config import get_rpc_classes
import numpy as np

BASE_RPC = get_rpc_classes()['base']
BASE_RPC_HOST = BASE_RPC.host
BASE_RPC_PORT = BASE_RPC.port
authkey_str = BASE_RPC.authkey
RPC_AUTHKEY = authkey_str.encode() if isinstance(authkey_str, str) else authkey_str

POLICY_CONTROL_PERIOD = 0.1

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