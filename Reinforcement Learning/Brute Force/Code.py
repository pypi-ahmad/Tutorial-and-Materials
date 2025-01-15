#First, we will define the environment and the state-action rewards:

import numpy as np

# Define the environment
grid_size = 3
goal_state = (grid_size - 1, grid_size - 1)

# Define the state-action rewards
rewards = np.zeros((grid_size, grid_size, 4))
rewards[:, :, 0] = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])  # Up
rewards[:, :, 1] = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])  # Down
rewards[:, :, 2] = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])  # Left
rewards[:, :, 3] = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])  # Right


#Next, we will define a function to calculate the state values using the Bellman equation:

def calculate_state_values(rewards):
    # Initialize the state values to 0
    state_values = np.zeros((grid_size, grid_size))

    # Perform 100 iterations of the Bellman equation
    for i in range(100):
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate the value of the current state
                value = 0
                for action in range(4):
                    next_row, next_col = get_next_state(row, col, action)
                    # Check if the next state is a valid state
                    if next_row >= 0 and next_row < grid_size and next_col >= 0 and next_col < grid_size:
                        value += 0.25 * (rewards[row, col, action] + state_values[next_row, next_col])
                # Update the state value
                state_values[row, col] = value

    return state_values

 
#Finally, we will use the state values to calculate the optimal policy:

def get_optimal_policy(state_values):
    # Initialize the policy to an array of zeros
    policy = np.zeros((grid_size, grid_size))

    # Calculate the policy for each state
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate the value of each action in the current state
            action_values = np.zeros(4)
            for action in range(4):
                next_row, next_col = get_next_state(row, col, action)
                # Check if the next state is a valid state
                if next_row >= 0 and next_row < grid_size and next_col >= 0 and next_col < grid_size:
                    action_values[action] = rewards[row, col, action] + state
