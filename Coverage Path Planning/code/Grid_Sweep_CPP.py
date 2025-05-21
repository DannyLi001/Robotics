import numpy as np

class GridSweepCPP:

    def __init__(self, x_len, y_len, vertical_num_grid, horizontal_num_grid):
        
        self.x_len = x_len
        self.y_len = y_len
        self.vertical_num_grid = vertical_num_grid
        self.horizontal_num_grid = horizontal_num_grid

        self.visited = np.zeros((self.vertical_num_grid, self.horizontal_num_grid), dtype=bool)
        self.xcoords = np.zeros((self.vertical_num_grid, self.horizontal_num_grid))
        self.ycoords = np.zeros((self.vertical_num_grid, self.horizontal_num_grid))

    def generate_xcoords(self):
        grid_width = self.x_len / self.horizontal_num_grid
        x = np.linspace(0, self.x_len, num = self.horizontal_num_grid + 1)[1:]
        x -= (grid_width / 2)
        for i in range(self.xcoords.shape[0]):
            self.xcoords[i, :] = x


    def generate_ycoords(self):
        grid_height = self.y_len / self.vertical_num_grid
        y = np.linspace(0, self.y_len, num=self.vertical_num_grid+1)[1:]
        y -= (grid_height / 2)
        y = y[::-1]
        for i in range(self.ycoords.shape[1]):
            self.ycoords[:, i] = y


    def can_move_left(self, row_idx, col_idx):
        if col_idx - 1 < 0:
            return False
        
        if self.visited[row_idx, col_idx - 1]:
            return False
        
        return True
    

    def can_move_right(self, row_idx, col_idx):
        if col_idx + 1 >= self.horizontal_num_grid:
            return False
        
        if self.visited[row_idx, col_idx + 1]:
            return False
        
        return True
    

    def can_move_up(self, row_idx, col_idx):
        if row_idx - 1 < 0:
            return False
        
        if self.visited[row_idx - 1, col_idx]:
            return False
        
        return True


    def search_path(self):
        path = []

        row_idx = self.vertical_num_grid - 1
        col_idx = 0

        while True:

            # mark visited
            self.visited[row_idx, col_idx] = True

            # append path
            path.append( [self.xcoords[row_idx, col_idx] , self.ycoords[row_idx, col_idx]] )

            if self.can_move_right(row_idx, col_idx):
                col_idx += 1
            elif self.can_move_left(row_idx, col_idx):
                col_idx -= 1
            elif self.can_move_up(row_idx, col_idx):
                row_idx -= 1
            else:
                break
        
        self.path = np.array( path )
    

    def generate_waypoints(self):
        
        angles = [0]

        for i in range(1, len(self.path)):
            past_point = self.path[i - 1]
            curr_point = self.path[i]

            if ( curr_point[0] - past_point[0] ) > 0:
                angles.append(0)
            elif( curr_point[0] - past_point[0] ) < 0:
                angles.append(np.pi)
            else:
                angles.append(angles[-1])

        angles = np.reshape(np.array(angles), (-1, 1))

        print(self.path.shape)
        print(angles.shape)

        temp_waypoints = np.hstack([self.path, angles])

        waypoints = []

        for i in range(len(temp_waypoints) - 1):

            curr_point = temp_waypoints[i]
            next_point = temp_waypoints[i+1]

            if next_point[-1] != curr_point[-1]:
                y = curr_point[1]

                if y < self.y_len / 2:
                    angle = -0.5 * np.pi
                else:
                    angle = 0.5 * np.pi

                waypoints.append(curr_point)
                waypoints.append([ curr_point[0], curr_point[1] , angle ] )
            
            else:
                waypoints.append(curr_point)

        waypoints.append(next_point)

        waypoints = np.array(waypoints)

        return waypoints