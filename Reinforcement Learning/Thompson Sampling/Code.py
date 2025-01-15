import numpy as np

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def pull(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward


# In the code, we define a class called ThompsonSampling with three methods - __init__(), pull(), and update(). The __init__() method initializes the alpha and beta parameters for each arm as 1. The pull() method randomly samples from the Beta distribution for each arm and selects the arm with the highest sample. The update() method updates the alpha and beta parameters for the selected arm based on the observed reward.

# Here's an example of how to use the class for a 3-armed bandit problem:

# Initialize ThompsonSampling object with 3 arms
ts = ThompsonSampling(3)

# Run 1000 trials
for i in range(1000):
    # Pull an arm
    arm = ts.pull()
    
    # Simulate reward
    reward = np.random.binomial(1, p=[0.1, 0.3, 0.5][arm])
    
    # Update the ThompsonSampling object with the observed reward
    ts.update(arm, reward)


    # In this example, we run 1000 trials of the 3-armed bandit problem. In each trial, we pull an arm using the pull() method, simulate a reward using a Bernoulli distribution with a probability of success that varies for each arm, and update the ThompsonSampling object with the observed reward using the update() method.