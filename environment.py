import numpy as np

class Environment:
    def __init__(self):
        self.space = np.zeros(shape=(5,5),dtype=int)
        self.action = np.array([0,1,2,3]) # 1 = up, 2 = left, 3 = down, 4 = right
        
        #target destination
        self.treasure = [4,4]
        
        #trap end
        self.trap = [2,2]
        
        self.Flag= False
        
        self.reset()
        
    def reset(self):
        self.state = [0,0]
        return self.state
    
    def step(self, action):
        if action==0:
            if self.state[0]!=0:
                self.state[0] -= 1
                
            else:
                return [-5, self.state, False]
        
        elif action==1:
            if self.state[1]!=0:
                self.state[1] -= 1
            else:
                return [-5, self.state, False]
            
        elif action==2:
            if self.state[0]!=4:
                self.state[0] += 1
            else: 
                return [-5, self.state, False]
        
        elif action==3:
            if self.state[1]!=4:
                self.state[1] += 1
            else:
                return [-5, self.state, False] 
            
        if self.state==self.treasure:
            self.Flag = True
            return [20,self.state,True]
            
        elif self.state==self.trap:
            self.Flag = True
            return [-10,self.state,True]
        
        else:
           return [-1, self.state, False]
       
if __name__=="main":
    
               
    env= Environment()
    total_rew = 0
    while(not env.Flag):
        res  = env.step(np.random.choice(env.action))
        print(f"Current State: {res[1]}, Reward: {res[0]}, Flage: {res[2]}")
        total_rew += res[0]
        
    print(f"total reward: {total_rew}")