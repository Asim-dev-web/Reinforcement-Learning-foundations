import gymnasium as gym
import numpy as np

env= gym.make("CartPole-v1")

num_actions= env.action_space.n

upper_bound= env.observation_space.high

np.set_printoptions(precision=2,suppress=True)

bins = [
    np.linspace(-4.8, 4.8, 10),      # Cart Position
    np.linspace(-4, 4, 10),          # Cart Velocity (clamped)
    np.linspace(-0.418, 0.418, 10),  # Pole Angle
    np.linspace(-4, 4, 10)           # Pole Angular Velocity (clamped)
]

def get_descrete_state(curr_state):
    state = []
    for i in range(len(curr_state)):
        state.append(np.digitize(curr_state[i],bins[i])-1)
    return tuple(state)
    
q = np.zeros((10,10,10,10,2),dtype=float)   

epochs = 3000
e = 100

def update_q(state,new_state,action,reward):
    q[state+(action,)] += 0.1 * (reward + 0.96 * np.max(q[new_state]) - q[state+(action,)])

for epoch in range(epochs):
    obs, info = env.reset()
    terminated, truncated = False,False
    done = False

    while(not done):
        state = get_descrete_state(obs)
        rand= np.random.randint(0,100)
        if rand<e:
            action = np.random.randint(0,num_actions)
        else: 
            action = np.argmax(q[state])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            reward= -100
            print("LOSE: Pole fell or Cart went off-track")
        if truncated:
            print("WIN: Survived 500 steps!")
        new_state = get_descrete_state(obs)
        update_q(state,new_state,action,reward)
        done = terminated or truncated
    
    e = max(1, e * 0.998)
    
print("--- TESTING PHASE ---")
for test_game in range(10):
    obs, info = env.reset()
    done = False
    score = 0
    while not done:
        state = get_descrete_state(obs)
        action = np.argmax(q[state]) # Pure intelligence, no luck
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
    print(f"Test Game {test_game} Score: {score}")
        
