import argparse
from agents.ppo import PPO
from agents.td import TD

parser = argparse.ArgumentParser()
parser.add_argument("--h",type=int,required=True,help="height of grid")
parser.add_argument("--w",type=int,required=True,help="width of grid")
parser.add_argument("--agent", type=str, required=True, help="td,ppo")
parser.add_argument("--emo",type=bool,default=False,help="use human emotional rewards?")
parser.add_argument("--pomdp",type=bool,default=False,help='partial observability?')
parser.add_argument("--epochs", type=int, default=100000, help="Number of epochs for training")

args = parser.parse_args()
if args.agent == 'td':
    bot = TD(args.h,args.w)
elif args.agent == 'ppo':
    bot = PPO(args.h,args.w,args.emo,args.pomdp)

bot.train(args.epochs)