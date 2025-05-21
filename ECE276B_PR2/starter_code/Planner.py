import numpy as np
import random

class MyPlanner:
  __slots__ = ['boundary', 'blocks']

  def __init__(self, boundary, blocks):
    self.boundary = boundary
    self.blocks = blocks

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

  def plan(self,start,goal):
    path = [start]
    numofdirs = 26
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    dR = np.delete(dR,13,axis=1)
    dR = dR / np.sqrt(np.sum(dR**2,axis=0)) / 2.0
    
    for _ in range(2000):
      mindisttogoal = 1000000
      node = None
      for k in range(numofdirs):
        next = path[-1] + dR[:,k]
        
        # Check if this direction is valid
        if( next[0] < self.boundary[0,0] or next[0] > self.boundary[0,3] or \
            next[1] < self.boundary[0,1] or next[1] > self.boundary[0,4] or \
            next[2] < self.boundary[0,2] or next[2] > self.boundary[0,5] ):
          continue
        
        if self._is_collision(path[-1], next):
          continue
        
        # Update next node
        disttogoal = sum((next - goal)**2)
        if( disttogoal < mindisttogoal):
          mindisttogoal = disttogoal
          node = next
      
      if node is None:
        break
      
      path.append(node)
      
      # Check if done
      if sum((path[-1]-goal)**2) <= 0.1:
        break
      
    return np.array(path)


class RRTNode:
  def __init__(self, coord, parent=None):
    self.coord = np.array(coord) 
    self.parent = parent         

class RRTPlanner:
  def __init__(self, boundary, blocks):
    self.boundary = boundary        
    self.blocks = blocks            
    self.step_size = 0.5            
    self.max_iter = 50000            
    self.goal_sample_rate = 0.05     

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

  def _random_sample(self, goal):
    if random.random() < self.goal_sample_rate:
      return np.array(goal)
    x = random.uniform(self.boundary[0,0], self.boundary[0,3])
    y = random.uniform(self.boundary[0,1], self.boundary[0,4])
    z = random.uniform(self.boundary[0,2], self.boundary[0,5])
    return np.array([x, y, z])

  def _find_nearest(self, tree, sample):
    min_dist = np.inf
    nearest_node = None
    for node in tree:
      dist = np.linalg.norm(node.coord - sample)
      if dist < min_dist:
        min_dist = dist
        nearest_node = node
    return nearest_node

  def _steer(self, from_node, to_point):
    direction = to_point - from_node.coord
    distance = np.linalg.norm(direction)
    if distance <= self.step_size:
      new_coord = to_point
    else:
      new_coord = from_node.coord + direction / distance * self.step_size
    return RRTNode(new_coord, from_node)

  def plan(self, start, goal):
    start_node = RRTNode(start)
    goal_node = RRTNode(goal)
    tree = [start_node]

    for _ in range(self.max_iter):
      sample = self._random_sample(goal)
      nearest_node = self._find_nearest(tree, sample)
      new_node = self._steer(nearest_node, sample)
      if not self._is_collision(nearest_node.coord, new_node.coord):
        tree.append(new_node)
        if np.linalg.norm(new_node.coord - goal) < self.step_size:
          final_node = self._steer(new_node, goal)
          if not self._is_collision(new_node.coord, final_node.coord):
            goal_node.parent = final_node
            tree.append(goal_node)
            path = []
            current = goal_node
            while current is not None:
              path.append(current.coord)
              current = current.parent
            return np.array(path[::-1])
    return np.array([goal])  