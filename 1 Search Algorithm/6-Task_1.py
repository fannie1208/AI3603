import DR20API
import heapq
import matplotlib.pyplot as plt
import numpy as np

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.
def Man_distance(pos, goal=[100,100]):
    """
    Given the current position and the goal position, calculate the Manhattam Distance.
    """
    return np.linalg.norm(np.array(pos)-np.array(goal),ord=1)

def Euc_distance(pos, goal=[100,100]):
    """
    Given the current position and the goal position, calculate the Euclidean Distance.
    """
    return np.linalg.norm(np.array(pos)-np.array(goal))

class Position:
    def __init__(self, position, g, parent, goal=[100,100]):
        self.position = position
        self.g = g
        self.parent = parent
        self.cost = Man_distance(position, goal)+self.g
    
    def __lt__(self, other):
        if self.cost == other.cost:
            return self.g < other.g
        return self.cost < other.cost
    
    def next_pos(self, current_map):
        positions = []
        for dir in [[0,1],[0,-1],[1,0],[-1,0]]:
            pos_x, pos_y = self.position[0] + dir[0], self.position[1] + dir[1]
            if 0 <= pos_x <= 119 and 0 <= pos_y <= 119 and current_map[pos_x][pos_y] != 1:
                positions.append([pos_x,pos_y])
        return positions
###  END CODE HERE  ###

def A_star(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal, 
    plan a path from current position to the goal using A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """

    ### START CODE HERE ###
    # Initialize the open list, closed list and the general cost from the starting position.
    open_lt, closed_lt, g = [], set(), 0
    # Push the starting position into the heap
    current_node =  Position(current_pos,g,None)
    heapq.heappush(open_lt, current_node)
    while tuple(current_node.position) != tuple(goal_pos):
        print(len(open_lt))
        current_node = heapq.heappop(open_lt)
        if (current_node.position[0],current_node.position[1]) in closed_lt:
            continue
        closed_lt.add((current_node.position[0],current_node.position[1]))
        for next_pos in current_node.next_pos(current_map):
            if (next_pos[0],next_pos[1]) not in closed_lt:
                heapq.heappush(open_lt, Position(next_pos,current_node.g+1,current_node))
                #closed_lt.add((next_pos[0],next_pos[1]))
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
    is_reached = Man_distance(current_pos,goal_pos) <= 5
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
        path = A_star(current_map, current_pos, goal_pos)
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    # Stop the simulation.
    controller.stop_simulation()