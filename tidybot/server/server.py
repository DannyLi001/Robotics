#!/usr/bin/env python3
import socket
import json
import numpy as np

from ..config.config import get_rpc_constants
from ..env.real_env import RealEnv


RPC_CFG = get_rpc_constants()

SERVER_HOST = RPC_CFG.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = RPC_CFG.get("SERVER_PORT", 5555)

def create_env():
    return RealEnv()


def start_server(host="0.0.0.0", port=5555):
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

def main():
    env = create_env()
    env.reset()

    # The start_server function now returns a listening socket, not a connection.
    server_sock = start_server(host=SERVER_HOST, port=SERVER_PORT)

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
                            state = env.get_state_obs()
                            state_json = json.dumps(serialize_state(state)) + "\n"
                            conn.sendall(state_json.encode("utf-8"))
                            print("[SERVER] Sent current state.")
                        if "arm" in command:
                            # env.step_arm_only(command["arm"])
                            print("[SERVER] Executed 'arm' action.")
                        if "base" in command:
                            # env.step_base_only(command["base"])
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
        env.close()
        server_sock.close()


if __name__ == "__main__":
    main()