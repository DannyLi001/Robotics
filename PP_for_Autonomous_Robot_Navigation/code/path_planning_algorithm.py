import numpy as np
from scipy.spatial import Voronoi
import copy

class Node:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class VisibilityRoadMap:

    def __init__(self):
        
        self.expand_distance = 0.2
        self.sq_len = 2.4
        self.obstacle_len = 0.4

        self.left_up = Node(0, self.sq_len)
        self.left_bottom = Node(0, 0)
        self.right_up = Node(self.sq_len, self.sq_len)
        self.right_bottom = Node(self.sq_len, 0)

        self.obstacle = [Node((self.sq_len - self.obstacle_len) / 2, (self.sq_len + self.obstacle_len) / 2), 
                         Node((self.sq_len + self.obstacle_len) / 2, (self.sq_len + self.obstacle_len) / 2), 
                         Node((self.sq_len + self.obstacle_len) / 2, (self.sq_len - self.obstacle_len) / 2), 
                         Node((self.sq_len - self.obstacle_len) / 2, (self.sq_len - self.obstacle_len) / 2)]
                        
        self.expanded_obstacle = [Node(self.obstacle[0].x - self.expand_distance , self.obstacle[0].y + self.expand_distance),
                                  Node(self.obstacle[1].x + self.expand_distance , self.obstacle[1].y + self.expand_distance),
                                  Node(self.obstacle[2].x + self.expand_distance , self.obstacle[2].y - self.expand_distance),
                                  Node(self.obstacle[3].x - self.expand_distance , self.obstacle[3].y - self.expand_distance)]

        self.start = Node(2.2, 0.2)
        self.goal = Node(0.2, 2.2)


    def planning(self):
        
        all_nodes = [self.left_up, self.left_bottom, self.right_bottom, self.right_up, self.start, self.goal] + self.expanded_obstacle
        visibility_graph = {}
        for node in all_nodes:
            visibility_graph[node] = []

        for idx_i, node_i in enumerate(all_nodes):
            for idx_j, node_j in enumerate(all_nodes):
                
                if idx_i == idx_j:
                    continue

                if node_i in self.expanded_obstacle and node_j in self.expanded_obstacle:
                    continue
                
                if not self.check_line_intersects_square(self.expanded_obstacle, node_i, node_j):
                    visibility_graph[node_i].append(node_j)
        
        return visibility_graph


    def orientation(self, p, q, r):
        """
        Returns the orientation of ordered triplet (p, q, r).
        Returns:
        0 --> p, q and r are collinear
        1 --> Clockwise
        -1 --> Counterclockwise
        """
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0
        return 1 if val > 0 else -1


    def on_segment(self, p, q, r):
        return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))


    def do_lines_intersect(self, p1, q1, p2, q2):

        if (p1.x == p2.x) and (p1.y == p2.y):
            return False
        
        if (p1.x == q2.x) and (p1.y == q2.y):
            return False
        
        if (q1.x == p2.x) and (q1.y == p2.y):
            return False
        
        if (q1.x == q2.x) and (q1.y == q2.y):
            return False


        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and self.on_segment(p1, p2, q1): return True
        if o2 == 0 and self.on_segment(p1, q2, q1): return True
        if o3 == 0 and self.on_segment(p2, p1, q2): return True
        if o4 == 0 and self.on_segment(p2, q1, q2): return True

        return False
    


    def check_line_intersects_square(self, square_nodes, start_node, end_node):
        """
        Checks if line segment from start_node to end_node intersects with the square obstacle
        """
        for i in range(4):
            square_edge_start = square_nodes[i]
            square_edge_end = square_nodes[(i + 1) % 4]
            
            if self.do_lines_intersect(start_node, end_node, square_edge_start, square_edge_end):
                return True
                
        return False
    

class Dijkstra:

    def __init__(self, graph, start, goal):

        self.graph = graph
        self.start = start
        self.goal = goal

    def find_min_distance_node(self, distances, visited):
        min_distance = float('infinity')
        min_node = None        
        for node in distances:
            if node not in visited and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node
        return min_node


    def planning(self):

        graph = self.graph
        start = self.start
        end = self.goal

        distances = {node: float('infinity') for node in graph}
        distances[start] = 0
        predecessors = {node: None for node in graph}
        visited = set()
        
        current_node = start
        while current_node is not None:
            
            visited.add(current_node)
            
            if current_node == end:
                path = []
                while current_node:
                    path.append(current_node)
                    current_node = predecessors[current_node]
                return list(reversed(path)), distances[end]
            
            for neighbor in graph[current_node]:
                if neighbor in visited:
                    continue
                
                distance = np.hypot(current_node.x - neighbor.x, current_node.y - neighbor.y)
                new_distance = distances[current_node] + distance
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
            
            current_node = self.find_min_distance_node(distances, visited)

        return [], float('infinity')



def generate_nodes_on_line(start_node, end_node, n):
    dx = (end_node.x - start_node.x) / (n - 1)
    dy = (end_node.y - start_node.y) / (n - 1)
    nodes = []
    for i in range(n):
        x = start_node.x + dx * i
        y = start_node.y + dy * i
        nodes.append(Node(x, y))
    return nodes




class VoronoiRoadMap:

    def __init__(self):

        # KNN Hyparameters
        self.N_KNN = 5
        self.MAX_EDGE_LEN = 30.0 

        self.sq_len = 2.4
        self.obstacle_len = 0.4

        self.left_up = Node(0, self.sq_len)
        self.left_bottom = Node(0, 0)
        self.right_up = Node(self.sq_len, self.sq_len)
        self.right_bottom = Node(self.sq_len, 0)

        self.obstacle = [Node((self.sq_len - self.obstacle_len) / 2, (self.sq_len + self.obstacle_len) / 2), 
                         Node((self.sq_len + self.obstacle_len) / 2, (self.sq_len + self.obstacle_len) / 2), 
                         Node((self.sq_len + self.obstacle_len) / 2, (self.sq_len - self.obstacle_len) / 2), 
                         Node((self.sq_len - self.obstacle_len) / 2, (self.sq_len - self.obstacle_len) / 2)]
                        
        self.start = Node(2.2, 0.2)
        self.goal = Node(0.2, 2.2)


    def planning(self):

        # get ox and oy
        ox = []
        oy = []

        line_nodes = generate_nodes_on_line(self.left_up, self.right_up, 25)
        ox.extend([d.x for d in line_nodes])
        oy.extend([d.y for d in line_nodes])

        line_nodes = generate_nodes_on_line(self.right_up, self.right_bottom, 25)
        ox.extend([d.x for d in line_nodes])
        oy.extend([d.y for d in line_nodes])

        line_nodes = generate_nodes_on_line(self.right_bottom, self.left_bottom, 25)
        ox.extend([d.x for d in line_nodes])
        oy.extend([d.y for d in line_nodes])

        line_nodes = generate_nodes_on_line(self.left_bottom, self.left_up, 25)
        ox.extend([d.x for d in line_nodes])
        oy.extend([d.y for d in line_nodes])


        temp = copy.deepcopy(self.obstacle)
        temp.append(temp[0])
        for node1, node2 in zip(temp[:-1], temp[1:]):
            line_nodes = generate_nodes_on_line(node1, node2, 5)
            ox.extend([d.x for d in line_nodes])
            oy.extend([d.y for d in line_nodes])
        
        line_nodes = generate_nodes_on_line(node1, node2, 5)
        ox.extend([d.x for d in line_nodes])
        oy.extend([d.y for d in line_nodes])

        sx = self.start.x
        sy = self.start.y
        gx = self.goal.x
        gy = self.goal.y

        sample_x, sample_y = self.voronoi_sampling(sx, sy, gx, gy, ox, oy)    

        np.savetxt("ox.txt", np.array(ox), fmt='%f')
        np.savetxt("oy.txt", np.array(oy), fmt='%f')
        np.savetxt("sample_x.txt", np.array(sample_x), fmt='%f')
        np.savetxt("sample_y.txt", np.array(sample_y), fmt='%f')


        graph = self.knn(sample_x, sample_y)

        # for node in graph:
        #     for neigh in graph[node]:
        #         plt.plot([node.x, neigh.x] ,  [node.y, neigh.y], marker = "*", color = "blue")
        # plt.show()

        return graph


    def knn(self, xs, ys):

        graph = {}

        for idxi, (xi, yi) in enumerate( zip(xs, ys) ):
            
            distances = []

            new_node = Node(xi, yi)

            graph[new_node] = []

            for idxj, (xj, yj) in enumerate( zip(xs, ys) ):

                if idxi == idxj:
                    continue
                
                distances.append(( np.hypot(xi-xj, yi-yj) ,xj, yj ))
            
            sorted_distances = sorted(distances, key=lambda x: x[0])

            closest_n_points = sorted_distances[: self.N_KNN]

            for _, tempx, tempy in closest_n_points:
                graph[new_node].append(Node(tempx, tempy))
        

        return graph

    def voronoi_sampling(self, sx, sy, gx, gy, ox, oy):
        oxy = np.vstack((ox, oy)).T

        # generate voronoi point
        vor = Voronoi(oxy)
        sample_x = [ix for [ix, _] in vor.vertices]
        sample_y = [iy for [_, iy] in vor.vertices]

        sample_x.append(sx)
        sample_y.append(sy)
        sample_x.append(gx)
        sample_y.append(gy)

        return sample_x, sample_y





# ##### Shortest Distance
# vrm = VisibilityRoadMap()
# graph = vrm.planning()
# dijkstra = Dijkstra(graph, vrm.start, vrm.goal)
# path, _ = dijkstra.planning()
# n = 5
# detailed_path = []
# for idx, (nodei, nodej) in enumerate( zip(path[:-1] , path[1:]) ):
#     detailed_path.extend(generate_nodes_on_line(nodei, nodej, n)[:-1])
# detailed_path.append(path[-1])
# waypoints = np.array( [[d.x, d.y] for d in detailed_path] )
# print(waypoints)

# plt.plot(waypoints[:, 0], waypoints[:, -1])
# plt.show()




# vrm = VoronoiRoadMap()
# graph = vrm.planning()
# dijkstra = Dijkstra(graph, vrm.start, vrm.goal)
# path, _ = dijkstra.planning()
# downsamples = path[1:-1][::2]
# path = [ path[0] ] + downsamples + [path[-1]]
# waypoints = np.array( [[d.x, d.y] for d in path] )