from env.env import Env
from agents.td import TD
from agents.ppo import PPO
env = Env(4,4)
env.render()

bot = PPO(4,4)
print(bot.train(200000))