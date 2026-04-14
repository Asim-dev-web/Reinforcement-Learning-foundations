import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0")
num_action = env.action_space.n
print(num_action)
print(env.observation_space)

bins= [
    np.linspace(-1.2,0.6,40),
    np.linspace(-0.07,0.07,40)
]

def get_bin(state):
    res = []
    for i in range(len(state)):
        res.append(np.digitize(state[i],bins[i]) -1 )
    return tuple(res)
    
q = np.zeros((40,40,3))

def update_q(state,new_state,action,reward):
    q[state + (action,)] += 0.1 * (reward + 0.99 * np.max(q[new_state]) - q[state + (action,)])

epoches = 7000

e= 100

for epoch in range(epoches):
    obs, info = env.reset()
    done = False
    truncated, terminated = False, False
    
    while(not done):
        state = get_bin(obs)
        rand = np.random.randint(0,100)
        if rand < e:
            action = np.random.randint(0,num_action)
        else:
            action = np.argmax(q[state])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print("Win! Car reached target!")
            
        if truncated:
            print("Lost!")
            
        new_state = get_bin(obs)
        update_q(state, new_state, action, reward)
            
        done = terminated or truncated
        
    e = max(1,e*0.999)
    
print(q)
print(f"Total non-zero cells: {np.count_nonzero(q)}")

# 2. Find the "worst" state (the bottom of the valley)
print(f"Lowest Q-value (Most pain): {np.min(q)}")

# 3. Look at the exact center of the table (where the car usually starts)
print(f"Center Q-values: {q[15, 15]}")
