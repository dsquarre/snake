from env.env import Env
from agents.td import TD

td = TD(4,4)
td.train(5000)