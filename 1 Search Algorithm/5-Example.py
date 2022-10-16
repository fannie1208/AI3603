import DR20API
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    # Define goal position
    goal_pos = [100, 100]

    # Initialize the controller of robot DR20.
    controller = DR20API.Controller()

    # Get current position of the robot.
    current_pos = controller.get_robot_pos()
    print(f'Current position of the robot is {current_pos}.')
    # Get current orientation of the robot.
    current_ori = controller.get_robot_ori()
    print(f'Current orientation of the robot is {current_ori}.')
    # Update the map of the world representing in a 120*120 array, where 0 indicating traversable and 1 indicating obstacles.
    current_map = controller.update_map()

    # Define a test path.
    path = [[16, 17], [17, 17], [18, 17], [19, 17], [20, 17], [21, 17], [22, 17], [23, 17], [24, 17], [25, 17], [26, 17], [27, 17]]
    # Visualize the map and path.
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if current_map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(current_pos[0], current_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    # Move the robot along the test path.
    print('Moving the robot ...')
    controller.move_robot(path)

    # Get current position of the robot.
    current_pos = controller.get_robot_pos()
    print(f'Current position of the robot is {current_pos}.')
    # Get current orientation of the robot.
    current_ori = controller.get_robot_ori()
    print(f'Current orientation of the robot is {current_ori}.')
    # Update the map of the world representing in a 120*120 array, where 0 indicating traversable and 1 indicating obstacles.
    current_map = controller.update_map()

    # Stop the simulation.
    controller.stop_simulation()