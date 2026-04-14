# Reinforcement Learning Foundations

A collection of Reinforcement Learning algorithms, starting from basic discrete environments and moving up to continuous control physics problems. This repository tracks my progression of implementing RL concepts from scratch.

## Project Timeline

### 1. Static Grid World
* **Environment:** Custom Discrete Grid
* **Focus:** Understanding the Bellman equation and state-action mapping.
* **Summary:** A baseline implementation of Tabular Q-learning in a finite environment to verify how reward back-propagation works.

### 2. CartPole (Tabular Q-Learning)
* **Environment:** `CartPole-v1`
* **Focus:** State discretization and dense rewards.
* **Summary:** Solving a continuous state space using a Q-table. This involved converting continuous physics observations (position, velocity, angle) into discrete bins, and tuning the exploration decay (epsilon) to prioritize balancing the pole over long periods.

### 3. MountainCar (Sparse Rewards)
* **Environment:** `MountainCar-v0`
* **Focus:** The exploration "cold start" problem and sparse rewards.
* **Summary:** The agent receives no positive feedback until the goal is reached, making standard greedy approaches fail. This implementation uses a higher-resolution state discretization (40 bins), allowing the agent to learn how to build momentum to climb the hill.

### 4. Deep Q-Networks (WIP)
* **Focus:** Function Approximation.
* **Summary:** Replacing the manual Q-table arrays with Neural Networks to handle high-dimensional state spaces without needing hardcoded bins.

---

## Tech Stack
* Python
* NumPy
* Gymnasium