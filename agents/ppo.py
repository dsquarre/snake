import torch
import torch.nn as nn
from env.env import Env
class NN(nn.Module):
  def __init__(self,H,W):
    super().__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Output: (32, H, W)
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),  # Output: (32, H, W)
        nn.ReLU(),
        nn.Conv2d(32,32,kernel_size=3,padding=1),
        nn.ReLU()
    )
    self.fc1 = nn.Sequential(
        nn.Flatten(),             # Output: (32 * H * W)
        nn.Linear(32 * H * W, 128),
        nn.ReLU(),
        nn.Linear(128, 4)         # Policy 
    )
    self.fc2 = nn.Sequential(
        nn.Flatten(),             # Output: (32 * H * W)
        nn.Linear(32 * H * W, 128),
        nn.ReLU(),
        nn.Linear(128, 1)         # Value 
    )

  def forward(self, x):
      x = self.cnn(x)
      policy = self.fc1(x)
      value = self.fc2(x)
      return policy,value
  
class PPO:
    def __init__(self,H,W):
        self.controller = NN(H+2,W+2)
        self.optimizer = torch.optim.Adam(self.controller.parameters(),lr=0.02)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller.to(self.device)
        self.height = H
        self.width = W
        self.buffer = []
    
    def get_state(self,env):
        state = torch.zeros([3,self.height+2,self.width+2],dtype=torch.float32,device=self.device)
        for i in range(3):
            for y in range(1,self.height+1):
                for x in range(1,self.width+1):
                    if i == 0: #head channel
                        if env.snake[0][0] == x and env.snake[0][1]==y:
                            state[i,y,x] = 1
                    elif i == 1: #body channel
                        for part in range(1,len(env.snake)):
                            if env.snake[part][0] == x and env.snake[part][1] == y:
                                state[i,y,x] = 1
                    elif i == 2: #apple channel
                        if env.apple_x == x and env.apple_y == y:
                            state[i,y,x] = 1       
        state = state.unsqueeze(0)
        return state
    
    def best_move(self,env):
        state = self.get_state(env)
        self.buffer.append(state)
        policy,value = self.controller(state)
        pi = torch.softmax(policy,dim=1).squeeze()
        moves = ['l','r','u','d']
        valid = env.valid_moves()
        for move in moves:
            if move not in valid:
                pi[moves.index(move)] = 0
                #only one move could be invalid
                break
        move = (pi==max(pi)).nonzero().squeeze()
        self.buffer.append(move) #action chosen from policy
        self.buffer.append(torch.log_softmax(policy,dim=1).squeeze())
        self.buffer.append(value)
        return moves[move]
        

    def train(self,games):
        env = Env(self.height,self.width)
        batch = []
        for g in range(1,games+1):
            if g%100 == 0:
                return batch
                #batching 50 games and solving
                #compute advantage
                #compute ratio
                #compute policy clipped loss
                #compute value loss
                self.optimizer.step()
                #clear batch
            env.reset()
            while(not env.gameover):
                move = self.best_move(env)
                reward = env.step(move,show=False)
                self.buffer.append(reward)
                if env.gameover:
                    self.buffer.append(True)
                else:
                    self.buffer.append(False)
            
            batch.append(self.buffer)
            self.buffer = []