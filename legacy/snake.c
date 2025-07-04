#include <stdio.h>
#include <stdlib.h>
#include <time.h>
void display(int [1600][3],int,int,int);
int move(int [1600][3],int,int,int,char);
char greedymove(int [1600][3],int,int,int);
//for computermove,basic idea is to store or return a movesequence rather than just a move.
//so A* search for movesequence from head to apple and then head to new tail but cut the sequence just after eating apple
char computermove(int [1600][3],int,int,int);
void main(){
	//char* head = "\033[107m \033[m";
	//printf("%s",head);
	srand(time(NULL)*8/5);
	int applei = 0;
	int applej = 20;
	int len = 3;
	int snake[1600][3];
	for(int i = 0;i<len;i++){
	snake[i][0] = 0;
	snake[i][1] = 2-i;
	snake[i][2] = 'r';
	}
	for(int i = len;i<1600;i++){
		snake[i][0] = 999;
		snake[i][1] = 999;
		snake[i][2] = 'z';
	}
	int alive = 1;
	while(alive){
		display(snake,len,applei,applej);
		//for(int i = 0;i<len;i++){ printf("%d %d %c, ",snake[i][0],snake[i][1],snake[i][2]); }
		//printf("\n");
		char dir = greedymove(snake,len,applei,applej);
		printf("%c\n",dir);
		alive = move(snake,len,applei,applej,dir);
		if(alive == 2){
			len += 1;
			int same = 1;
			//gen apple
			while(same){
				same = 0;
				applei = rand()%40;
				applej = rand()%40;
				for(int a=0;a<len;a++){
					if(applei == snake[a][0] && applej == snake[a][1]){ same = 1; }
				}
			}
		}
		//printf("%d\n",(snake[0][0]-snake[len-1][0])*(snake[0][0]-snake[len-1][0]) + (snake[0][1]-snake[len-1][1])*(snake[0][1]-snake[len-1][1]));
		for(int i= 0;i<1000000;i++);	//frame control
		}
	printf("Game Over! Snake Dead\n");
	
	}
	
	
void display(int snake[1600][3],int len,int applei,int applej){
	printf("\e[H\e[2J\e[3J"); //clear screen
	//print one big line
	for(int i = 0;i<84;i++){
		printf("\033[%dm \033[m",47);
		}
	
	//print newline+one wall+40grid values + one wall x40
	for(int i=0;i<40;i++){
		printf("\n\033[47m  \033[m"); //newline+wall
		for(int j=0;j<40;j++){
			int empty = 1;
			if(i == applei && j == applej){ printf("\033[41m  \033[m"); empty = 0;} //apple
			else{
			for(int a=0;a<len;a++){
				if(i == snake[a][0] && j == snake[a][1]){
						empty = 0;
						if(a==0){
							printf("\033[102m  \033[m");//head
							}
						else if(a == len-1){
							printf("\033[46m  \033[m");//tail
						}
						else{
							printf("\033[42m  \033[m");//body
							} 
						}
					}
			  }
			  
			if(empty){printf("\033[8m  \033[m");} //empty		
			}	
		printf("\033[47m  \033[m"); //wall
		}
	printf("\n");
	//print one big line
	for(int i = 0;i<84;i++){
		printf("\033[%dm \033[m",47);
		}
	printf("\n");
	printf("Score : %d\n",len);
	}


int move(int snake[1600][3],int len,int applei,int applej,char direction){
	//move head in direction and other parts in stored direction and store the i-1 direction in i
	snake[0][2] = direction;
	char dir = 'z';
	char tmp = direction;
	for(int i = 0;i<len;i++){
		dir = snake[i][2];
		snake[i][2] = tmp ;
		if(dir == 'u'){
			snake[i][0] -= 1;
		}
		else if(dir == 'd'){
			snake[i][0] += 1;
		}
		else if(dir == 'r'){
			snake[i][1] += 1;	
		}
		else if(dir == 'l'){
			snake[i][1] -= 1;
		}
 		tmp = dir;
 		
		if(i!=0 && snake[0][0] == snake[i][0] && snake[0][1] == snake[i][1]){return 0;}
		if(snake[0][0] == 40 || snake[0][1] == 40){return 0;}	
	}
	
	//ate apple
	if(snake[0][0] == applei && snake[0][1] == applej){
		//add tail in direction dir at same position as last tail
		snake[len][0] = snake[len-1][0];
		snake[len][1] = snake[len-1][1];
		snake[len][2] = tmp;
		//shift tail opposite to dir
		if(tmp == 'u'){
			snake[len][0] += 1;
		}
		else if(tmp == 'd'){
			snake[len][0] -= 1;
		}
		else if(tmp == 'r'){
			snake[len][1] -= 1;	
		}
		else if(tmp == 'l'){
			snake[len][1] += 1;
		}
		return 2;
	}
	for(int i = 1;i<len+1;i++){
			//printf("%d %d %c, ",snake[0][0],snake[0][1],snake[0][2]);
			if(snake[0][0] == snake[i][0] && snake[0][1] == snake[i][1]){return 0;}
		} //eats tail
		if(snake[0][0] == 40 || snake[0][0] == -1 || snake[0][1] == 40 || snake[0][1] == -1){return 0;}	//bumps wall
		//move back head
	//printf("\n");
	return 1;
}


char greedymove(int snake[1600][3],int len,int applei,int applej){
	int headi = snake[0][0]; int headj = snake[0][1];
	double initial = (headi-applei)*(headi-applei) + (headj-applej)*(headj-applej);
	char possible[3];
	if (snake[0][2] == 'r' || snake[0][2] == 'l'){
        	possible[0] = 'u';
        	possible[1] = 'd';
        	possible[2] = snake[0][2];
        }
   	 else {
        	possible[0] = 'l';
        	possible[1] = 'r';
        	possible[2] = snake[0][2];
        }
        int eval[3];
	for(int j = 0;j<3;j++){
		eval[j] = 0;
	 	char dir = possible[j];
		if(dir == 'u'){
			headi -= 1;
		}
		else if(dir == 'd'){
			headi += 1;
		}
		else if(dir == 'r'){
			headj+= 1;	
		}
		else if(dir == 'l'){
			headj -= 1;	
		}
		
		double final = (headi-applei)*(headi-applei) + (headj-applej)*(headj-applej);
		eval[j] = (initial-final);
	
		for(int i = 0;i<len+1;i++){
			if(headi == snake[i][0] && headj == snake[i][1]){eval[j] = -999;}
		} //eats tail
		if(headi == 40 || headi == -1 || headj == 40 || headj == -1){eval[j] = -999;}	//bumps wall
		
		headi = snake[0][0]; headj = snake[0][1];
	}
	//printf("%d %d %d\n ",eval[0],eval[1],eval[2]);
			//for(int i= 0;i<100000000;i++);
			
	return (eval[0]>eval[1])?((eval[0]>eval[2])?possible[0]:possible[2]):((eval[1]>eval[2])?possible[1]:possible[2]);
}

char computermove(int snake[1600][3],int len,int applei,int applej){
	char moveseq[1600];
	char nodesvisited[3];
	int nodes = 0;
	int seqlen = 0;
	int tempsnake[1600][3];
	int failed = 1;
	while(failed){
		//initialize snake
		for(int i =0;i<len;i++){
		for(int j=0;j<3;j++){
			tempsnake[i][j] = snake[i][j];}}
		//traverse moveseq
		for(int i=0;i<seqlen;i++){
			move(tempsnake,len,applei,applej,moveseq[i]);
		}
		while(1){//adding possible move
			char greedy = greedymove(tempsnake,len,applei,applej);
			char mov = greedy;
			int count = 0;
			for(int i= 0;i<nodes;i++){
				if(greedy == nodesvisited[i]){//greedy move is not good
					
					if(tempsnake[0][2] == greedy){
					greedy = mov;
					char possible[2];
					if(tempsnake[0][2] == 'u' || tempsnake[0][2] == 'd'){
						possible[0] = 'l';
						possible[0] = 'r'; 
					}
					
					else if(tempsnake[0][2] == 'r'|| tempsnake[0][2] == 'l'){
						possible[0] = 'u';
						possible[0] = 'd'; 	
					} 
					while(greedy==mov){
       					 greedy = possible[rand()%2];}
       					 count+=1;
       					 i = -1; 
       					}
       					else{
       						char possible[3];
						if (snake[0][2] == 'r' || snake[0][2] == 'l'){
        						possible[0] = 'u';
        						possible[1] = 'd';
        						possible[2] = snake[0][2];
      						  }
   						else {
        						possible[0] = 'l';
        						possible[1] = 'r';
        						possible[2] = snake[0][2];
       							 }
       						
       						greedy = possible[rand()%3];
       						count+=1;
       					 	i = -1;	
       					}
       				}
       				if(count>30){greedy = ' ';break;}
       			}
       			if(greedy==' '){
       				if(seqlen < 1){return greedymove(snake,len,applei,applej);}
       			 seqlen-=1; nodesvisited[nodes] = moveseq[seqlen]; nodes+=1; failed = 1; break; }
			int ret = move(tempsnake,len,applei,applej,greedy);
			//ate apple
			if(ret== 2){
				moveseq[seqlen]=greedy; seqlen+=1; nodes=0;
				failed = 0; break;
			}
			//ok move
			else if (ret==1){moveseq[seqlen]=greedy; seqlen+=1; nodes=0;}
			//bad move
			else{nodesvisited[nodes]=greedy; nodes+=1;failed = 1;break;}
		}	
	}
	for(int i = 0;i<seqlen-1;i++){
		//printf("%c,",moveseq[i]);
		move(snake,len,applei,applej,moveseq[i]);
		display(snake,len,applei,applej);
		for(int i= 0;i<7000000;i++);
	}//printf("\n");		
	return moveseq[seqlen-1];
}

