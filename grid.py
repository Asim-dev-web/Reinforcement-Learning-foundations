import numpy as np
from environment import Environment

env = Environment()

np.set_printoptions(precision=2, suppress=True)

q = np.zeros((5,5,4) , dtype = float)

def update_q(i,j,k,r):
    q[i,j,k] += 0.1 * (r + (0.4 * np.max(q[env.state[0], env.state[1]])) - q[i,j,k])
    
epochs = 2000

e= 100

for epoch in range(epochs):
    env.reset()
    done = False
    while(not done):
        [i,j] = env.state
        chance = np.random.randint(0,100)
        if chance<e:
            k = np.random.randint(0,4)
        else:
            k= np.argmax(q[i,j])
            
        r, state, done= env.step(k)
        print(f"Reward:{r}, curr_state:[{env.state[0]},{env.state[1]}]")
        
        update_q(i,j,k,r)
        
    e*=0.99

print(q)