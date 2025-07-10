import random
import time
class Env:
    def clear_screen(self):
        import os
        os.system('cls' if os.name == 'nt' else 'clear')

    def __init__(self,height,width):
        #1,width would be empty; 0,width+1 would be wall
        #same for height
        self.height = height
        self.width = width
        self.snake = []
        headx = random.randint(1,width)
        heady = random.randint(1,height)
        head = [headx,heady,None] #starting from topmost left
        #[x,y,dir]
        self.snake.append(head)
        self.gameover = True
        if len(self.snake)<height*width:
            self.gameover = False
        else:
            print("game is over bro")
        self.apple_x,self.apple_y = self.make_apple()

    def reset(self):

        self.snake = []

        head = [1,1,None] #starting from topmost left
        #[x,y,dir]
        self.snake.append(head)
        self.gameover = True
        if len(self.snake)<self.height*self.width:
            self.gameover = False
        else:
            print("game is over bro")
        self.apple_x,self.apple_y = self.make_apple()

    def make_apple(self):
        valid = False
        x,y = 1,1
        while(not valid):
            x =random.randint(1,self.width)
            y = random.randint(1,self.height)
            valid = True
            for part in self.snake:
                if x==part[0] and y==part[1]:
                    valid = False
                    break
        return x,y

    def step(self,direction,show=True):
#move head in direction and other parts in their stored direction and store the i-1 direction in i
        #print(f"got {self.snake}")
        self.snake[0][2] = direction
        tmp = direction
        tail_x,tail_y,dir = 0,0,None
        ate_apple = False
        reward = 0
        for i in range(len(self.snake)):
            dir = self.snake[i][2]
            if(dir == 'u'):
                self.snake[i][1] -= 1

            elif(dir == 'd'):
                self.snake[i][1] += 1

            elif(dir == 'r'):
                self.snake[i][0] += 1

            elif(dir == 'l'):
                self.snake[i][0] -= 1
            self.snake[i][2] = tmp
            tmp = dir
            #print(f"moved part {i} in direction {dir}")
            if(self.snake[i][0] == self.apple_x and self.snake[i][1] == self.apple_y):
                ate_apple = True
                tail_x = self.snake[len(self.snake)-1][0]
                tail_y = self.snake[len(self.snake)-1][1]
                dir = self.snake[-1][2]
                reward = 10
            if(i!=0 and self.snake[0][0] == self.snake[i][0] and self.snake[0][1] == self.snake[i][1]):
                self.gameover = True
                if show :print("snake ate itself")
                #break
                reward = -1

        if(self.snake[0][0] == self.width+1 or self.snake[0][0] == 0):
            self.gameover=True
            if show : print(f"snake hit wall {self.snake[0][0]}")
            reward = -1
        if(self.snake[0][1] == 0 or self.snake[0][1] == self.height+1):
            self.gameover = True
            if show :print(f"snake hit wall {self.snake[0][1]}")
            reward = -1
        if ate_apple:
            #if actual : print("apple")
            #add tail in direction dir at same position as last tail
            #shift tail opposite to dir
            if (len(self.snake) == 1):
                if(dir == 'u'):
                    tail_y += 1

                elif(dir == 'd'):
                    tail_y -= 1

                elif(dir == 'r'):
                    tail_x -= 1

                elif(dir == 'l'):
                    tail_x += 1
            self.snake.append([tail_x,tail_y,dir])
            if len(self.snake) == self.height*self.width:
                self.gameover = True
                self.apple_x,self.apple_y = None,None
                reward = 20
            else : self.apple_x,self.apple_y = self.make_apple()

        return reward

    def render(self):
        #self.clear_screen() #clear screen
	    #print top wall
        for x in range(self.width+2):
            print("\033[47m  \033[m",end='')

        for y in range(1,self.height+1):
            print("\n\033[47m  \033[m",end='') #newline+wall
            ##grid x height  + one wall
            for x in range(1,self.width+1):
                empty = True
                if(x == self.apple_x and y == self.apple_y):
                    print("\033[41m  \033[m",end='') #make apple
                    empty = False
                else:
                    for i in range(len(self.snake)):
                        if(x == self.snake[i][0] and y == self.snake[i][1]):
                            if(i==0):
                                print("\033[102m  \033[m",end='')  #head
                                empty=False
                                break
                            else:
                                print("\033[42m  \033[m",end='')  #body
                                empty=False
                                break
                if empty:
                    print("\033[40m  \033[m",end='') #empty
            print("\033[47m  \033[m",end='')  #wall
        print()
        #print one big line
        for i in range(self.width+2):
            print("\033[47m  \033[m",end='')
        print("\033[40m\033[m\n") #empty new line
        #print(f"\nScore: {len(self.snake)}")

    def valid_moves(self):
        moves = ['l','r','u','d']
        #print(len(self.snake))
        if len(self.snake) > 1:
            direction = self.snake[0][2]
            if direction == 'u':
                invalid = 'd'
            elif direction == 'd':
                invalid = 'u'
            elif direction == 'r':
                invalid = 'l'
            else :
                invalid = 'r'
            del(moves[moves.index(invalid)])
            #print(f"deleted move {invalid}")
        return moves

    def clone(self):
        import copy
        new_env = Env(self.height,self.width)
        new_env.apple_x = self.apple_x
        new_env.apple_y = self.apple_y
        new_env.gameover = self.gameover
        new_env.snake=copy.deepcopy(self.snake)
        return new_env

    def greedy(self):
        moves = self.valid_moves()
        ok = []
        for action in moves:
            new_env = self.clone()
            #print(action,end=' ')
            reward = new_env.step(action,False)
            #print(reward)
            if reward == 1:
                return action
            if not reward == -1:
                ok.append(action)
        #print(ok)
        if not ok:
            return random.choice(moves)
        return random.choice(ok)

   
