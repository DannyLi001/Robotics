import numpy as np
from pqdict import pqdict
import math

class AStarNode:
  def __init__(self, coord, g=math.inf, h=0, parent=None):
    self.coord = tuple(coord)  
    self.g = g                 
    self.h = h                 
    self.parent = parent       
    self.closed = False        

  def f(self, epsilon=1.0):
    return self.g + epsilon * self.h  

  def __lt__(self, other):
    return self.f() < other.f()
      

class AStarPlanner:
  def __init__(self, boundary, blocks):
    self.boundary = boundary    
    self.blocks = blocks        
    self.resolution = 0.5       

  def _is_collision(self, A, B):
    for block in self.blocks:
      t_min, t_max = 0.0, 1.0
      for i in range(3):
        a, b = A[i], B[i]
        delta = b - a
        slab_min, slab_max = block[i], block[i+3]
        if abs(delta) < 1e-9:
          if a < slab_min or a > slab_max:
            break
          continue
        t0 = (slab_min - a) / delta
        t1 = (slab_max - a) / delta
        t0, t1 = min(t0, t1), max(t0, t1)
        t_min = max(t_min, t0)
        t_max = min(t_max, t1)
        if t_min > t_max:
          break
      else:
        if t_min <= t_max and (t_max >= 0 and t_min <= 1):
          return True
    return False

  def _get_neighbors(self, node):
    neighbors = []
    steps = np.arange(-1, 2) * self.resolution
    for dx in steps:
      for dy in steps:
        for dz in steps:
          if dx == 0 and dy == 0 and dz == 0:
            continue 
          
          neighbor_coord = np.array(node.coord) + np.array([dx, dy, dz])
          
          if (neighbor_coord[0] < self.boundary[0,0] or neighbor_coord[0] > self.boundary[0,3] or
            neighbor_coord[1] < self.boundary[0,1] or neighbor_coord[1] > self.boundary[0,4] or
            neighbor_coord[2] < self.boundary[0,2] or neighbor_coord[2] > self.boundary[0,5]):
            continue
          if self._is_collision(node.coord, neighbor_coord):
            continue
          
          neighbors.append(neighbor_coord)
    return neighbors

  def _heuristic(self, coord, goal):
    return np.linalg.norm(np.array(coord) - np.array(goal))

  def plan(self, start, goal, epsilon=1.0):
    start = np.array(start)
    goal = np.array(goal)
    open_list = pqdict()  
    nodes = {}

    start_node = AStarNode(start, g=0, h=self._heuristic(start, goal))
    nodes[start_node.coord] = start_node
    open_list.additem(start_node.coord, start_node.f(epsilon))

    while open_list:
      current_coord, _ = open_list.popitem()
      current_node = nodes[current_coord]
      current_node.closed = True

      if np.linalg.norm(np.array(current_coord) - goal) < 0.5:
        path = [tuple(goal)]
        while current_node:
          path.append(current_node.coord)
          current_node = current_node.parent
        return np.array(path[::-1])

      for neighbor_coord in self._get_neighbors(current_node):
        neighbor_coord = tuple(neighbor_coord)
        if neighbor_coord in nodes and nodes[neighbor_coord].closed:
          continue

        move_cost = np.linalg.norm(np.array(neighbor_coord) - current_node.coord)
        tentative_g = current_node.g + move_cost

        if neighbor_coord not in nodes or tentative_g < nodes[neighbor_coord].g:
          h = self._heuristic(neighbor_coord, goal)
          neighbor_node = AStarNode(
              coord=neighbor_coord,
              g=tentative_g,
              h=h,
              parent=current_node
          )
          nodes[neighbor_coord] = neighbor_node
          if neighbor_coord in open_list:
            open_list.updateitem(neighbor_coord, neighbor_node.f(epsilon))
          else:
            open_list.additem(neighbor_coord, neighbor_node.f(epsilon))

    return np.array([start])