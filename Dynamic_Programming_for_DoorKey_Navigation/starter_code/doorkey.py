from utils import *
from example import example_use_of_gym_env
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
import numpy as np
from collections import deque
from minigrid.core.world_object import Wall

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)


def get_state_info(env):
    info = {
        "width": env.unwrapped.width,
        "height": env.unwrapped.height,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "door_pos": [],
        "door_open": [],
        "key_pos": None,
        "goal_pos": None
    }
    for i in range(env.unwrapped.height):
        for j in range(env.unwrapped.width):
            cell = env.unwrapped.grid.get(j, i)
            if isinstance(cell, Door):
                info["door_pos"].append((j, i))
                info["door_open"].append(cell.is_open)
            elif isinstance(cell, Key):
                info["key_pos"] = (j, i)
            elif isinstance(cell, Goal):
                info["goal_pos"] = (j, i)
    return info

def create_grid_map(env):
    width = env.unwrapped.width
    height = env.unwrapped.height
    grid = [[None for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            cell = env.unwrapped.grid.get(x, y)
            if isinstance(cell, Wall):
                grid[y][x] = 'wall'
            elif isinstance(cell, Door):
                grid[y][x] = 'door'
            elif isinstance(cell, Key):
                grid[y][x] = 'key'
            elif isinstance(cell, Goal):
                grid[y][x] = 'goal'
            else:
                grid[y][x] = 'empty'
    return grid

def direction_to_vec(direction):
    return [(1, 0), (0, 1), (-1, 0), (0, -1)][direction]

def get_initial_state(info):
    x, y = info["init_agent_pos"]
    dir_vec = info["init_agent_dir"]
    direction = [(1,0), (0,1), (-1,0), (0,-1)].index(tuple(dir_vec.tolist()))
    has_key = False
    door_locked = not info["door_open"][0] if info["door_pos"] else False
    return (x, y, direction, has_key, door_locked)

def transition(state, action, grid, door_pos, key_pos, goal_pos, width, height):
    x, y, dir, has_key, door_locked = state
    new_x, new_y, new_dir, new_has_key, new_door_locked = x, y, dir, has_key, door_locked
    c = 1
    # obstacles in map
    cell = grid[y][x]
    if cell == 'wall':
        pass
    elif action == MF:
        dx, dy = direction_to_vec(dir)  # right 0; down 1; left 2; up 3
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            cell = grid[ny][nx]
            if cell == 'wall':
                pass
            elif cell == 'door':
                if door_locked:
                    pass
                else:
                    new_x, new_y = nx, ny
            else:
                new_x, new_y = nx, ny

    elif action == TL:
        new_dir = (dir - 1) % 4

    elif action == TR:
        new_dir = (dir + 1) % 4

    elif action == PK:
        dx, dy = direction_to_vec(dir)  # right 0; down 1; left 2; up 3
        fx, fy = x + dx, y + dy
        if (fx, fy) == key_pos and not has_key:
            new_has_key = True

    elif action == UD:
        dx, dy = direction_to_vec(dir)  # right 0; down 1; left 2; up 3
        fx, fy = x + dx, y + dy
        if (fx, fy) in door_pos:
            if door_locked and has_key:
                new_door_locked = False

    # doesn't move
    if state == (new_x, new_y, new_dir, new_has_key, new_door_locked):
        c = float('inf')
    terminated = (new_x, new_y) == goal_pos
    return (new_x, new_y, new_dir, new_has_key, new_door_locked), c, terminated

def dynamic_programming(grid, door_pos, key_pos, goal_pos, width, height, gamma=1.0):
    # (pos), dir, have key, door locked
    states = [(x, y, d, k, dl) 
              for x in range(1, width - 1) 
              for y in range(1, height - 1) 
              for d in range(4) 
              for k in [False, True] 
              for dl in [False, True]]
    
    # (max steps = grid area * 4 directions)
    T = width * height * 4  
    
    # Initialize value functions and policies
    V = {t: {s: float('inf') for s in states} for t in range(T+1)}
    policy = {t: {s: None for s in states} for t in range(T)}
    
    # initialize q(x)
    goal_x, goal_y = goal_pos
    for s in states:
        x, y, _, _, _ = s
        V[T][s] = 0.0 if (x, y) == (goal_x, goal_y) else float('inf')

    # Backward Dynamic Programming
    for t in range(T-1, -1, -1):
        for s in states:
            # fix value in goal pos
            x, y, _, _, _ = s
            V[t][s] = 0.0 if (x, y) == (goal_x, goal_y) else V[t][s]
            min_val = V[t][s]
            best_action = policy[t][s]
            
            for action in [MF, TL, TR, PK, UD]:
                # S_t+1 = pi(St, u)
                next_state, cost, done = transition(s, action, grid, door_pos, key_pos, goal_pos, width, height)
                
                # Q(x,u)
                next_val = cost + gamma * V[t+1][next_state]
                
                if next_val < min_val:
                    min_val = next_val
                    best_action = action
            
            V[t][s] = min_val
            policy[t][s] = best_action
        # print()
    return policy, V

def doorkey_problem(env):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """
    info = get_state_info(env)
    grid = create_grid_map(env)
    door_pos = info["door_pos"]
    key_pos = info["key_pos"]
    goal_pos = info["goal_pos"]
    width, height = info["width"], info["height"]
    
    policy, _ = dynamic_programming(grid, door_pos, key_pos, goal_pos, width, height)
    initial_state = get_initial_state(info)
    
    # Execute policy
    current_state = initial_state
    actions = []
    for t in range(width * height * 4):  # Max horizon steps
        if policy[t][current_state] is None:
            break
        actions.append(policy[t][current_state])
        next_state, _, done = transition(current_state, actions[-1], grid, door_pos, key_pos, goal_pos, width, height)
        if done: break
        current_state = next_state
    
    return actions


def partA():
    env_folder = "./envs/known_envs/"
    env_files = ['doorkey-5x5-normal.env', 'doorkey-6x6-normal.env', 'doorkey-8x8-normal.env',
                 'doorkey-6x6-direct.env', 'doorkey-8x8-direct.env',
                 'doorkey-6x6-shortcut.env', 'doorkey-8x8-shortcut.env']
    for env_file in env_files:
        env_path = env_folder + env_file
        env, _ = load_env(env_path)
        seq = doorkey_problem(env)
        draw_gif_from_seq(seq, env, path=f"./gif/{env_file[:-4]}.gif")






# Predefined configurations for random environments
KEY_POSITIONS = [(2,2), (2,3), (1,6)]
GOAL_POSITIONS = [(6,1), (7,3), (6,6)]
DOOR_POSITIONS = [(5,3), (5,7)]

def get_state_info_b(env):
    """Extract environment info with encoded indices for key/goal positions"""
    info = {
        "width": env.unwrapped.width,
        "height": env.unwrapped.height,
        "init_agent_pos": env.unwrapped.agent_pos,
        "init_agent_dir": env.unwrapped.dir_vec,
        "door_open": [],
        "key_pos": None,
        "goal_pos": None
    }
    
    # Extract door statuses
    door_status = []
    for pos in DOOR_POSITIONS:
        door = env.grid.get(*pos)
        door_status.append(door.is_open if door else True)
    info["door_open"] = door_status
    
    # Find key and goal positions
    for i in range(env.height):
        for j in range(env.width):
            cell = env.grid.get(j, i)
            if isinstance(cell, Key):
                info["key_pos"] = (j, i)
            elif isinstance(cell, Goal):
                info["goal_pos"] = (j, i)
    
    # Encode positions to indices
    info["key_idx"] = KEY_POSITIONS.index(info["key_pos"])
    info["goal_idx"] = GOAL_POSITIONS.index(info["goal_pos"])
    
    return info

def transition_b(state, action, grid, width, height):
    """Extended transition function for random environments"""
    x, y, d, has_key, door1_open, door2_open, key_idx, goal_idx = state
    new_state = list(state)
    cost = 1  # Default action cost
    
    key_pos = KEY_POSITIONS[key_idx]
    goal_pos = GOAL_POSITIONS[goal_idx]
    door1_pos, door2_pos = DOOR_POSITIONS

    cell = grid[y][x]
    if cell == 'wall':
        pass
    elif action == MF:
        dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][d]
        nx, ny = x + dx, y + dy
        
        if 0 <= nx < width and 0 <= ny < height:
            cell = grid[ny][nx]
            if cell == 'wall':
                pass
            elif cell == 'door':
                if (nx, ny) == door1_pos and not door1_open:
                    pass  # Blocked by closed door
                elif (nx, ny) == door2_pos and not door2_open:
                    pass
                else:
                    new_state[0], new_state[1] = nx, ny
            else:
                new_state[0], new_state[1] = nx, ny
            

    elif action == TL:
        new_state[2] = (d - 1) % 4
    elif action == TR:
        new_state[2] = (d + 1) % 4
    elif action == PK:
        dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][d]
        nx, ny = x + dx, y + dy
        if (nx, ny) == key_pos and not has_key:
            new_state[3] = True
        
    elif action == UD:
        dx, dy = [(1,0), (0,1), (-1,0), (0,-1)][d]
        fx, fy = x + dx, y + dy
        if (fx, fy) == door1_pos and not door1_open and has_key:
            new_state[4] = True
        elif (fx, fy) == door2_pos and not door2_open and has_key:
            new_state[5] = True

    terminated = (new_state[0], new_state[1]) == goal_pos
    # doesn't move
    if state == tuple(new_state):
        cost = float('inf')
    return tuple(new_state), cost, terminated

def create_grid_map_b():
    width = 10
    height = 10
    grid = [['empty' for _ in range(width)] for _ in range(height)]
    
    for y in range(height):
        for x in range(width):
            if y == 0 or x == 0 or y == height-1 or x == width-1 or x == 5:
                grid[y][x] = 'wall'
            if (x, y) in DOOR_POSITIONS:
                grid[y][x] = 'door'
    return grid
    

def dynamic_programming_b(width, height):
    """Universal policy computation for all random environments"""
    # (pos), dir, have key, door1_open, door2_open, key idx, goal idx
    # KEY_POSITIONS = [(2,2), (2,3), (1,6)]
    # GOAL_POSITIONS = [(6,1), (7,3), (6,6)]
    states = [(x, y, d, k, d1, d2, kidx, gidx)
              for x in range(1, width - 1)
              for y in range(1, height - 1)
              for d in range(4)
              for k in [False, True]
              for d1 in [False, True]
              for d2 in [False, True]
              for kidx in range(3)
              for gidx in range(3)]
    
    # general map (only wall and doors)
    grid = create_grid_map_b()
    
    V = {s: float('inf') for s in states}
    policy = {s: None for s in states}
    
    # initialize q(x)
    for s in states:
        if (s[0], s[1]) == GOAL_POSITIONS[s[7]]:
            V[s] = 0.0
    
    # Value iteration
    while True:
        delta = 0
        for s in states:
        
            if V[s] == 0.0: continue
            
            min_cost = float('inf')
            best_action = None
            
            for action in [MF, TL, TR, PK, UD]:
                # S_t+1 = pi(St, u)
                next_state, cost, done = transition_b(s, action, grid, width, height)
                # Q(x,u)
                total_cost = cost + (0 if done else V.get(next_state, float('inf')))
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_action = action
            
            if V[s] > min_cost:
                delta = max(delta, V[s] - min_cost)
                V[s] = min_cost
                policy[s] = best_action
        
        if delta < 1e-6:
            break
    
    return policy

def doorkey_problem_b(env):
    """Execute precomputed policy for random environment"""
    info = get_state_info_b(env)
    grid = create_grid_map_b()
    initial_state = (
        info["init_agent_pos"][0],
        info["init_agent_pos"][1],
        [(1,0), (0,1), (-1,0), (0,-1)].index(tuple(info["init_agent_dir"])),
        False,
        info["door_open"][0],
        info["door_open"][1],
        info["key_idx"],
        info["goal_idx"]
    )
    
    current_state = initial_state
    actions = []
    visited = set()
    
    while current_state not in visited:
        visited.add(current_state)
        action = UNIVERSAL_POLICY.get(current_state, None)
        if action is None: break
        
        actions.append(action)
        next_state, _, done = transition_b(current_state, action, grid, 10, 10)
        if done: break
        current_state = next_state
    
    return actions


# Precompute policy for all possible configurations
UNIVERSAL_POLICY = dynamic_programming_b(10, 10)

def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)
    seq = doorkey_problem_b(env)
    draw_gif_from_seq(seq, env, path=f"./gif/{os.path.basename(env_path)[:-4]}.gif")


if __name__ == "__main__":
    # example_use_of_gym_env()
    # partA()
    # random.seed(10)
    partB()

