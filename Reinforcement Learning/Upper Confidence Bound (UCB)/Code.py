import numpy as np

# Define the number of arms and the exploration factor
num_arms = 10
exploration_factor = 2

# Initialize the rewards and counts for each arm
rewards = [0] * num_arms
counts = [0] * num_arms

# Define the UCB function for selecting the next arm to play
def ucb(t):
    ucb_values = [0] * num_arms
    for i in range(num_arms):
        if counts[i] > 0:
            # Calculate the mean reward and exploration term for this arm
            mean_reward = rewards[i] / counts[i]
            exploration_term = exploration_factor * np.sqrt(np.log(t) / counts[i])
            ucb_values[i] = mean_reward + exploration_term
        else:
            # If this arm has not been played yet, assign a large UCB value to it
            ucb_values[i] = float('inf')
    # Select the arm with the highest UCB value
    return np.argmax(ucb_values)

# Run the UCB algorithm for a specified number of steps
num_steps = 1000
for t in range(1, num_steps+1):
    # Select the next arm to play using UCB
    arm = ucb(t)
    # Play the arm and observe the reward
    reward = play_arm(arm)
    # Update the reward and count for the selected arm
    rewards[arm] += reward
    counts[arm] += 1


# In this code, we first define the number of arms and the exploration factor. We then initialize the rewards and counts for each arm to zero.

# The ucb() function is defined to calculate the UCB value for each arm at each step. For each arm, the mean reward and exploration term are calculated and added to obtain the UCB value. If an arm has not been played yet, a large UCB value is assigned to it to ensure it is explored.

# The main loop runs the UCB algorithm for a specified number of steps. At each step, the ucb() function is called to select the next arm to play. The arm is played and the reward is observed. The reward and count for the selected arm are then updated.

# Overall, the UCB algorithm is a simple yet effective method for balancing exploration and exploitation in reinforcement learning. It is commonly used in applications such as online advertising and recommendation systems to select the most promising actions based on limited feedback.