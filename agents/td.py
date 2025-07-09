import random
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from env.env import Env
random.seed(0)
torch.manual_seed(0)

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
    self.fc = nn.Sequential(
        nn.Flatten(),             # Output: (32 * H * W)
        nn.Linear(32 * H * W, 128),
        nn.ReLU(),
        nn.Linear(128, 1)         # Scalar value V(s)
    )

  def forward(self, x):
      x = self.cnn(x)
      x = self.fc(x)
      return x

class TD:
  def __init__(self,H,W):
    self.height = H
    self.width = W
    self.value = NN(H+2,W+2)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.E_params = [torch.zeros_like(p.data,device=self.device) for p in self.value.parameters()]
    self.value.to(self.device)

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

  def best_move(self,env,g,online=False):
    moves = env.valid_moves()
    #print(moves)
    utility = {}
    for move in moves:
      new_env = env.clone()
      reward = new_env.step(move,show=False)
      new_state = self.get_state(new_env)
      utility[move] = self.value(new_state).detach().item() 
    greedy = max(utility,key=utility.get)
    epsilon =  epsilon = max(0.015, 0.2 * math.exp(-0.0005 * g))
    if random.random() < epsilon and not online: 
      return random.choice(moves)
    return greedy

  def moving_average(self,data, window_size):
    result = []
    sum_ = 0
    for i in range(len(data)):
        sum_ += data[i]
        if i >= window_size:
            sum_ -= data[i - window_size]
            result.append(sum_ / window_size)
        elif i == window_size - 1:
            result.append(sum_ / window_size)
    return result

  def train(self,games):
    env = Env(self.height,self.width)
    reward = 0
    lambd = 0.61
    gamma = 0.89
    alpha = 0.0008
    apples_eaten = []
    net_rewards = []
    avg_td_errors = []
    for g in range(1,games+1):
      if(g%1000==0):
        print(f"games done : {g}")
      net_reward = 0
      total_td_error = 0
      apples = 0
      movecount = 0
      steps = 0
      env.reset()
      for e in self.E_params:
        e.zero_()
      while not env.gameover:
        if g < 5: 
          env.render()
        state = self.get_state(env)
        move = self.best_move(env,g)
        '''if g<6:
          print(f"chose {move}")'''
        reward = env.step(move,show=False)
        if movecount>200:
          env.gameover = True
          reward = -1
        movecount += 1
        if reward > 1:
          apples += 1
        steps += 1
        net_reward += reward
        value_s = self.value(state)
        new_state = self.get_state(env)
        value_new_s = self.value(new_state).detach()
        td_error = reward + gamma*value_new_s - value_s
        if math.isnan(td_error.item()):
          print("NaN td_error!")
          return None
        total_td_error += abs(td_error.item())
        self.value.zero_grad()
        value_s.backward()
        for param, e in zip(self.value.parameters(), self.E_params):
          if param.grad is not None:
            #print(e.data.shape)
            #print(param.grad.shape)
            #print(param.data.shape)
            #return "shit"
            e.data = gamma * lambd * e.data + param.grad.data
            e.data = torch.clamp(e.data, -1e3, 1e3)
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
            param.data += alpha * td_error.item() * e.data
           
        '''if env.gameover:
          if g < 5:
            env.render()
            print(f"game over {reward}")'''
            
      #game over
      net_rewards.append(net_reward)
      avg_td_errors.append(total_td_error/steps)
      apples_eaten.append(apples)
    reward_plot = self.moving_average(net_rewards, 100)
    td_error_plot = self.moving_average(avg_td_errors, 100)
    apple_plot = self.moving_average(apples_eaten,100)
    self.Plot(reward_plot,td_error_plot,apple_plot,games)
    #print(reward_plot)
    #print(td_error_plot)

  def Plot(self,reward,td_error,apple_rate,games):
    import matplotlib.pyplot as plt
    x = [i for i in range(games-99)] 
    plt.plot(x, reward,color='green', label='Reward/game') 
    plt.plot(x,td_error,color='black', label='TDError/game')  
    plt.plot(x,apple_rate,color='red',label='apples_eaten/game')
    plt.xlabel('Games')
    plt.ylabel('Moving Averages')
    plt.legend()
    plt.savefig(f"plots/td{self.height}x{self.width}.png")
    plt.show()