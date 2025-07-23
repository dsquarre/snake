import torch
import torch.serialization
import random
import torch.nn as nn
from env.env import Env
import time
import absl.logging

# Set absl logging to ERROR or higher
absl.logging.set_verbosity(absl.logging.ERROR)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
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
    def __init__(self,H,W,emo,pomdp):
        self.height = H
        self.width = W
        torch.serialization.add_safe_globals([
            NN,
            nn.Sequential,
            nn.ReLU,
            nn.Conv2d,
            nn.Flatten,
            nn.Linear
        ])
        #self.controller = NN(H+2,W+2)
        try:
            self.controller = torch.load(f'trained_models/ppo{self.height}x{self.width}.pt')
            print('model loaded')
        except Exception as e:
            self.controller = NN(H+2,W+2)
            print(e)
            print("training from scratch")
        self.emo=emo
        self.pomdp=pomdp
        self.optimizer = torch.optim.Adam(self.controller.parameters(),lr=3e-4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.controller.to(self.device)  
        
    
    def get_state(self,env):
        state = torch.zeros([3,self.height+2,self.width+2],dtype=torch.float32,device=self.device)
        view = env.partial_state()
        for i in range(3):
            for y in range(self.height+2):
                for x in range(self.width+2):
                    if i == 0: #head channel
                        if env.snake[0][0] == x and env.snake[0][1]==y:
                            state[i,y,x] = 1
                    elif i == 1: #body channel
                        if self.pomdp:
                            #get partial_state
                            for pos in view:
                                if pos[0]==x and pos[1]==y:
                                    state[i,y,x] = 1
                                    #if snake body change to -1
                                    for part in range(1,len(env.snake)):
                                        if env.snake[part][0] == x and env.snake[part][1] == y:
                                            state[i,y,x] = -1
                                    #if wall change to -2
                                    if x==self.width+1 or x==0 or y==self.height+1 or y==0:
                                        state[i,y,x]=-2
                                    

                        else:
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
        pi = torch.softmax(policy,dim=1).squeeze()
        action = torch.multinomial(pi, 1).item()
        logpi = torch.log_softmax(policy,dim=1).squeeze()
        log_prob = logpi[action]
        expected_val = value.squeeze().item()
        entropy = -1*sum(pi*logpi)
        return moves[action],state,action,log_prob,expected_val,entropy
    
    def compute_adv(self,rewards,values, gamma=0.99,lamda=0.9):
        advantages = [0]*len(rewards)
        td_errors = []
        i=0
        while (i< len(rewards)):
            if i == len(rewards)-1:
                td = rewards[i] -values[i]
                td_errors.append(td)
                break
            td = rewards[i] + gamma * values[i+1] - values[i]
            td_errors.append(td)
            i+=1
            
        for i in range(len(rewards)):
            k = 0
            for j in range(i,len(rewards)):
                advantages[i]+= td_errors[j] * (lamda*gamma)**k
                k+=1

        return advantages

    def train(self,games):
        env = Env(self.height,self.width)
        batch = []
        entropies = []
        losses = []
        net_rewards = []
        apples_eaten = []
        gametime= []
        for g in range(1,games+1):
            if g%10000 == 0: print(f"{g} games done")
            if g%1000==0:  
                self.ppo_update(batch)
                batch.clear()
                #losses.append(total_loss)
            env.reset()
            if self.emo:
                env.clear_screen()
                env.render()
                time.sleep(0.5)
            apples = 0
            net_reward = 0
            steps = 0
            states = []
            actions = []
            log_prob_sa = []
            values = []
            rewards = []
            entropy = 0
            loss = 0
            while(not env.gameover):
                #step = {}
                move,state,action,log_prob,expected_val,e = self.best_move(env)
                entropy+= e
                states.append(state)
                actions.append(action)
                log_prob_sa.append(log_prob)
                values.append(expected_val)
                if move not in env.valid_moves():
                    reward = -1
                    env.gameover = True
                else:
                    reward = env.step(move,show=False)
                if self.emo:
                    print("chose move: {move}")
                    time.sleep(0.6)
                    env.clear_screen()
                    env.render()
                    from camera import Emotion
                    human = Emotion()
                    human_reward = human.reward(frames=2)
                    print(human_reward)
                    reward += human_reward*0.1
                
                steps+= 1
                if steps > self.height*self.width*2:
                    reward = -1
                    env.gameover = True
                rewards.append(reward)
                net_reward += reward
                if reward == 10:
                    apples += 1
                
                if env.gameover: 
                    advantages = self.compute_adv(rewards,values)
                    loss += (sum(advantages)/len(advantages))
                    for index in range(len(rewards)):
                        batch.append({
                            'state': states[index],
                            'action': actions[index],
                            'log_prob': log_prob_sa[index],
                            'rewards': rewards[index],
                            'advantage': advantages[index],
                        })
            net_rewards.append(net_reward)
            apples_eaten.append(apples)
            gametime.append(steps)
            entropies.append(entropy/steps)
            losses.append(loss)
        reward_plot = self.moving_average(net_rewards, 1000)
        apple_plot = self.moving_average(apples_eaten,1000)
        gametime_plot = self.moving_average(gametime,1000)
        entropy_plot = self.moving_average(entropies,1000)
        loss_plot = self.moving_average(losses,1000)
        print("Training done")
        torch.save(self.controller,f'trained_models/ppo{self.height}x{self.width}.pt')
        self.Plot(reward_plot,gametime_plot,apple_plot,entropy_plot,loss_plot)
       

    def play(self,games):
        for i in range(games):
            steps = 0
            env = Env(self.height,self.width)
            env.render()
            
            while not env.gameover:
                state = self.get_state(env)
                with torch.no_grad():
                    policy,value = self.controller(state)
                policy = policy.detach()
                moves = ['l','r','u','d']
                print(moves)
                print(policy)
                pi = torch.softmax(policy,dim=1).squeeze()
                action = torch.argmax(pi).item()
                move = moves[action]
                print(f'doing {move}, expected value {value}')
                if move not in env.valid_moves():
                    print("invalid move")
                    
               
                time.sleep(3)
                if move not in env.valid_moves():
                    print("invalid move")
                    break
                env.step(move)
                steps+= 1    
                env.clear_screen()
                env.render()
                
                if steps > 2*self.height*self.width:
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

    def Plot(self,reward,gametime,apple_rate,entropy,loss):
        import matplotlib.pyplot as plt
        x = range(len(reward))
        plt.plot(x, reward,color='green', label='Reward/game') 
        plt.plot(x,gametime,color='black', label='Steps/game')  
        plt.plot(x,apple_rate,color='red',label='apples_eaten/game')
        plt.plot(x,entropy,color='blue',label='entropy/game')
        plt.plot(x,loss,label='loss/game')
        plt.xlabel('Games')
        plt.ylabel('Moving Averages')
        plt.legend()
        plt.savefig(f"plots/ppo{self.height}x{self.width}.png")
        plt.show()
        
    def ppo_update(self, batch, epochs=30, minibatch_size=128, eps_clip=0.1):
        for _ in range(epochs):
            minibatch = random.sample(batch,minibatch_size)
            states = torch.cat([item['state'] for item in minibatch]).to(self.device)  # [B, 3, H, W]
            actions = torch.tensor([item['action'] for item in minibatch], device=self.device)  # [B]
            advantages = torch.tensor([item['advantage'] for item in minibatch], dtype=torch.float32, device=self.device)  # [B]
            old_log_probs = torch.tensor([item['log_prob'].item() for item in minibatch], device=self.device)
            rewards = torch.tensor([item['rewards'] for item in minibatch],dtype=torch.float32, device=self.device)
            # Forward pass
            policies, values_new = self.controller(states) #[B,4],[B,1]
            values_new = values_new.squeeze(-1)
            log_policies = torch.log_softmax((policies),dim=-1) #[B,4]
            new_log_probs = log_policies[range(len(actions)), actions]
                        
            # Clipped surrogate
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            unclipped = ratios * advantages
            clipped = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            policy_loss = -torch.mean(torch.min(unclipped, clipped))
            F = nn.MSELoss()
            entropy = -torch.sum(torch.exp(log_policies) * log_policies, dim=1).mean()
            value_loss = F(values_new, rewards)
            loss = policy_loss + 0.5 * value_loss - 0.9 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 0.5)
            self.optimizer.step()

