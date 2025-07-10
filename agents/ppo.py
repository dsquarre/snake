import torch
import torch.serialization
import random
import copy
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
        #self.controller = NN(H+2,W+2)
        #torch.serialization.add_safe_globals([NN])
        torch.serialization.add_safe_globals([
            NN,
            nn.Sequential,
            nn.ReLU,
            nn.Conv2d,
            nn.Flatten,
            nn.Linear
        ])
        self.controller = torch.load('ppo4x4.pt')
        print('model loaded')
        self.optimizer = torch.optim.Adam(self.controller.parameters(),lr=1e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller.to(self.device)  
        self.height = H
        self.width = W
    
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
        #self.step['state']=state
        with torch.no_grad():
            policy,value = self.controller(state)
        policy = policy.detach()

        moves = ['l','r','u','d']
        valid = env.valid_moves()
        for move in moves:
            if move not in valid:
                policy[0][moves.index(move)] = -1e9
                #disallowing moving backward
                break
        pi = torch.softmax(policy,dim=1).squeeze()
        action = torch.multinomial(pi, 1).item()
        #self.step['action'] = (move) #action chosen from policy
        log_prob = torch.log_softmax(policy,dim=1).squeeze()[action]
        expected_val = value.squeeze()
        return moves[action],state,action,log_prob,expected_val
    
    def compute_returns(self,rewards, gamma=0.99):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    def train(self,games):
        env = Env(self.height,self.width)
        batch = []
        losses = []
        net_rewards = []
        apples_eaten = []
        gametime= []
        for g in range(1,games+1):
            total_loss = 0
            if g%400 == 0:
                if g%10000 == 0: print(f"{g} games done")
                total_loss = self.ppo_update(batch)
                batch.clear()
                losses.append(total_loss)
            env.reset()
            apples = 0
            net_reward = 0
            steps = 0
            trajectory = []
            
            while(not env.gameover):
                step = {}
                move,state,action,log_prob,expected_val = self.best_move(env)
                step['state'] = state
                step['action'] = action
                step['log_prob'] =  log_prob
                step['value'] = expected_val
                reward = env.step(move,show=False)
                #if g==games: env.render()
                steps+= 1
                if steps > self.height*self.width*5:
                    reward = -1
                    env.gameover = True
                step['reward'] = reward
                net_reward += reward
                if reward > 0:
                    apples += 1
                
                if env.gameover:
                    step['done'] = (True)
                    trajectory.append(copy.deepcopy(step))
                    returns = self.compute_returns([go['reward'] for go in trajectory])
                    index = 0
                    for go in trajectory:
                        ret = returns[index]
                        batch.append({
                            'state': go['state'],
                            'action': go['action'],
                            'log_prob': go['log_prob'],
                            'value': go['value'],
                            'returns': ret,
                            'done': go['done']
                        })
                        index += 1
                    
                else:
                    step['done'] = (False)
                    trajectory.append(copy.deepcopy(step))
                    step = {}
            net_rewards.append(net_reward)
            apples_eaten.append(apples)
            gametime.append(steps)
        ''' reward_plot = self.moving_average(net_rewards, 100)
        apple_plot = self.moving_average(apples_eaten,100)
        gametime_plot = self.moving_average(gametime,100)'''
        #print(len(losses))
        print("Training done")
        torch.save(self.controller,f'ppo{self.height}x{self.width}.pt')

        self.Plot(net_rewards,gametime,apples_eaten,losses,games)
        
        self.play()

    def play(self):
        steps = 0
        env = Env(self.height,self.width)
        while not env.gameover:
            state = self.get_state(env)
            #self.step['state']=state
            with torch.no_grad():
                policy,value = self.controller(state)
            policy = policy.detach()
            print(policy)
            moves = ['l','r','u','d']
            valid = env.valid_moves()
            for move in moves:
                if move not in valid:
                    policy[0][moves.index(move)] = -1e9
                    #disallowing moving backward
                    break
            pi = torch.softmax(policy,dim=1).squeeze()
            action = torch.argmax(pi).item()
            move = moves[action]
            print(f'doing {move}, expected value {value}')
            env.step(move)
            steps+= 1
            env.render()
            if steps > 5*self.height*self.width:
                env.gameover = True

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

    def Plot(self,reward,gametime,apple_rate,losses,games):
        import matplotlib.pyplot as plt
        x = [i for i in range(games)] 
        x2 = [i for i in range(1,games,400)]
        plt.plot(x, reward,color='green', label='Reward/game') 
        plt.plot(x,gametime,color='black', label='Steps/game')  
        plt.plot(x,apple_rate,color='red',label='apples_eaten/game')
        plt.plot(x2,losses,color='blue',label='loss')
        plt.xlabel('Games')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(f"plots/ppo{self.height}x{self.width}.png")
        plt.show()
        
    def ppo_update(self, batch, epochs=30, minibatch_size=128, eps_clip=0.1):
        final_loss = 0
        for _ in range(epochs):
            minibatch = random.sample(batch,minibatch_size)

            states = torch.cat([item['state'] for item in minibatch]).to(self.device)  # [B, 3, H, W]
            actions = torch.tensor([item['action'] for item in minibatch], device=self.device)  # [B]
            returns = torch.tensor([item['returns'] for item in minibatch], dtype=torch.float32, device=self.device)  # [B]
            old_log_probs = torch.tensor([item['log_prob'].item() for item in minibatch], device=self.device)
            values_old = torch.tensor([item['value'].item() for item in minibatch], device=self.device)

            # Forward pass
            policies, values_new = self.controller(states) #[B,4],[B,1]
            values_new = values_new.squeeze(-1)
            log_policies = torch.log_softmax((policies),dim=-1) #[B,4]
            new_log_probs = log_policies[range(len(actions)), actions]

            # Advantage
            advantages = returns - values_old
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Clipped surrogate
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            unclipped = ratios * advantages
            clipped = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            policy_loss = -torch.mean(torch.min(unclipped, clipped))
            F = nn.MSELoss()
            entropy = -torch.sum(torch.exp(log_policies) * log_policies, dim=1).mean()
            value_loss = F(values_new, returns)
            loss = policy_loss + 0.5 * value_loss - 0.1 * entropy
            if _ == epochs-1: final_loss = loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 0.5)
            self.optimizer.step()
        return final_loss.item()