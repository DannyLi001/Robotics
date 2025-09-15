import time
from multiprocessing.managers import BaseManager
from .base_controller import Vehicle
from ...config.config import get_rpc_classes

BASE_RPC = get_rpc_classes()['base']
BASE_RPC_HOST = BASE_RPC.host
BASE_RPC_PORT = BASE_RPC.port
authkey_str = BASE_RPC.authkey
RPC_AUTHKEY = authkey_str.encode() if isinstance(authkey_str, str) else authkey_str

class Base:
    def __init__(self, max_vel=(0.5, 0.5, 1.57), max_accel=(0.25, 0.25, 0.79)):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.vehicle = None

    def reset(self):
        # Stop low-level control
        if self.vehicle is not None:
            if self.vehicle.control_loop_running:
                self.vehicle.stop_control()

        self.vehicle = Vehicle(max_vel=self.max_vel, max_accel=self.max_accel)

        # Start low-level control
        self.vehicle.start_control()
        while not self.vehicle.control_loop_running:
            time.sleep(0.01)

    def execute_action(self, action):
        if 'base_pose' in action:
            self.vehicle.set_target_position(action['base_pose'])
        elif 'base_vel' in action:
            self.vehicle.set_target_velocity(action['base_vel'])
        else:
            print('Invalid action')

    def get_state(self) -> dict:
        state = {'base_pose': self.vehicle.x, 'base_vel': self.vehicle.dx}
        return state

    def close(self):
        if self.vehicle is not None:
            if self.vehicle.control_loop_running:
                self.vehicle.stop_control()

class BaseManager(BaseManager):
    pass


def main():
    BaseManager.register('Base', Base)
    base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
    server = base_manager.get_server()
    print(f'Base manager server started at {BASE_RPC_HOST}:{BASE_RPC_PORT}')
    server.serve_forever()


if __name__ == '__main__':
    main()
    # pass