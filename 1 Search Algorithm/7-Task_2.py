import DR20API
import math
import numpy as np
from numpy.linalg import norm
import heapq
import matplotlib.pyplot as plt
from collections import deque

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.
def Man_distance(pos, goal=[100,100]):
    """
    Given the current position and the goal position, calculate the Manhattam Distance.
    """
    return norm(np.array(pos)-np.array(goal),ord=1)

def Euc_distance(pos, goal=[100,100]):
    """
    Given the current position and the goal position, calculate the Euclidean Distance.
    """
    return norm(np.array(pos)-np.array(goal))

def bfs(current_pos, current_map, window=15):
    """
    Using the BFS method to find the nearest obstacle within the window area.
    """
    ans, dis = 100, 1
    visited = set()
    win = [max(0,current_pos[0]-window), min(119,current_pos[0]+window), max(0,current_pos[1]-window), min(119,current_pos[1]+window)]
    que = deque([tuple(current_pos)])
    while que:
        num = len(que)
        for _ in range(num):
            pos = que.popleft()
            if pos in visited:
                continue
            visited.add(pos)
            for dx,dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                x, y = pos[0]+dx, pos[1]+dy
                if win[0] <= x <= win[1] and win[2] <= y <= win[3] and (x, y) not in visited:
                    if current_map[x][y]:
                        return dis
                    que.append((x,y))
        dis  += 1
        
    return ans

class Position:
    def __init__(self, position, g, parent, goal=[100,100]):
        self.position = position
        self.g = g
        self.parent = parent
        self.dis_cost = Man_distance(position, goal)+self.g
        self.obs_cost, self.st_cost = 0, 0
    
    def next_pos(self, current_map):
        positions = []
        for dirx, diry in ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)):
            pos_x, pos_y = self.position[0] + dirx, self.position[1] + diry
            if 0 <= pos_x <= 119 and 0 <= pos_y <= 119 and current_map[pos_x][pos_y] != 1:
                positions.append([pos_x,pos_y])
        return positions
    
    def Obs_cost(self, current_map, coefficient=8):
        """
        Calculate the extra cost induced by obstacles.
        """
        self.obs_cost = 1/bfs(self.position, current_map)*coefficient
        return self.obs_cost
    
    def Steel_cost(self, cur_pos, pre_pos=None, coefficient=-2, flag=True):
        current_ori = controller.get_robot_ori()
        vec1, vec2 = np.array(cur_pos)-np.array(pre_pos) if flag else np.array([np.sin(current_ori), np.cos(current_ori)]), np.array(self.position)-np.array(cur_pos)
        #cos = np.dot(vec1, vec2)/(norm(vec1)*norm(vec2))
        #sin = np.cross(vec1, vec2)/(norm(vec1)*norm(vec2))
        #theta = abs(np.arctan2(sin,cos))
        theta = np.dot(vec1, vec2)/(norm(vec1)*norm(vec2))
        self.st_cost = (theta)*coefficient
        return self.st_cost
    
    def __lt__(self, other):
        return self.dis_cost + self.obs_cost + self.st_cost < other.dis_cost + other.obs_cost + other.st_cost

###  END CODE HERE  ###

def Improved_A_star(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal, 
    plan a path from current position to the goal using improved A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by improved A* algorithm.
    """

    ### START CODE HERE ###
    open_lt, closed_lt, g = [], set(), 0
    # Push the starting position into the heap
    current_node =  Position(current_pos,g,None)
    heapq.heappush(open_lt, current_node)
    while not reach_goal(current_node.position, goal_pos):
        current_node = heapq.heappop(open_lt)
        if (current_node.position[0],current_node.position[1]) in closed_lt:
            continue
        closed_lt.add((current_node.position[0],current_node.position[1]))
        for next_pos in current_node.next_pos(current_map):
            if (next_pos[0],next_pos[1]) not in closed_lt:
                node = Position(next_pos,current_node.g+1,current_node)
                print(node.Obs_cost(current_map))
                if current_node.parent:
                    print(node.Steel_cost(current_node.position, current_node.parent.position))
                heapq.heappush(open_lt, node)
    path = []
    while current_node.parent:
        path.append(current_node.position)
        current_node = current_node.parent
    path = path[::-1]
    ###  END CODE HERE  ###
    return path

def reach_goal(current_pos, goal_pos):
    """
    Given current position of the robot, 
    check whether the robot has reached the goal.

    Arguments:
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    is_reached -- A bool variable indicating whether the robot has reached the goal, where True indicating reached.
    """

    ### START CODE HERE ###
    is_reached = Man_distance(current_pos, goal_pos) <= 1
    ###  END CODE HERE  ###
    return is_reached

if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.
    goal_pos = [100, 100]
    controller = DR20API.Controller()

    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    current_map = controller.update_map()

    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal.
        path = Improved_A_star(current_map, current_pos, goal_pos)
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    # Stop the simulation.
    controller.stop_simulation()