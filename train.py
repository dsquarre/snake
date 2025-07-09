from env.env import Env
from agents.td import TD
from agents.ppo import PPO
bot = PPO(4,4)
print(bot.train(60))