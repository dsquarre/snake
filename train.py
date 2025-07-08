from env.env import Env
from agents.td import TD

td = TD(6,6)
td.train(25000)