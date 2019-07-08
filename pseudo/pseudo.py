""" RL design - MCTS
State: 각 BS의 켜지고 꺼진 상태, BS n개면 특정 node에서 state도 n개. discrete state.
action: 각 BS에 대해 켜고 끄는 것을 action으로, 켜면 1, 끄면 0. action은 policy에 의해 결정. discrete action
policy를 최적화, 초기 policy는 랜덤. episode가 끝날때 reward에 따라 update. -->update 방식?
value: state에 의해 value값을 결정, 에) BS_3:stqte_3이 1이고, UE와 BS_3의 거리를 D로 두면, value는 D에 대한 함수가 된다.
value는 S일단 SNR로 두었음. 따라서 value는 D^-alpha. alpha는 NLos로 가정하여 -4의 값
UE는 1~m명, BS는 1~n개, 거리 matrix D는 n by m matrix로 D_ij=norm(BS_i=UE_j)
연산속도를 위해 D_ij=(BS_ix-UE_jx)**2+(BS_iy-UE_jy)**2
BS와 UE의 xy좌표는 ppp로 나중에 뿌림. 지금은 위치 정해져 있다고 생각
시작 노드: BS 모두 켜진 상태(BS_n=1 for every n)-->통신 성능이 threshold이하이면 terminate되는 조건해야하므로, 
"""
#########대략적인 MCTS##############
'''
def MCTS(node=root node, threshold, terminal)
    initialize root node
    for search tree:
        calculate the rewards
        if rewards < threshold:
            break
        elif:
            moves to next nodes
        if node arrives terminal:
            end for
        update policy(backpropagate)
    return best policy/reward
'''

'''
@@@env code에서 rl code로 보내는 거 고려할 것
action(n by 1), n은 BS개수, 10개 이하로
def step(action):
    pass
    return reward, done, next state
reset - UE 분포같은 거 다 초기화하는 녀석
input, output은 gym env 기준으로 할 예정
'''
import random
import math
import hashlib
import argparse
#필요한 변수, n(int),m(int), matrix D, BS vector(n by 1), UE vector(m by 1)
n=10 #BS의 개수
m=20 #UE의 개수
class BS():
#    def setBS(self,state=1,posisition): #initialize BS
#        self.state=state #root node(시작노드)의 BS는 켜진채로 시작. or self.state=state?
        self.position=position #나중에 ppp
        pass
class UE():
    def setUE(self,position):
        self.position=position #나중에 ppp
        pass 

class Node():   #중간에서 terminate되지 않을시 episode는 node 1~n까지, node_i는 BS_i를 policy에 따라 켜고 끈다.
    def __init__(self, state,reward):
        self.state=state
        self.reward= #SNR
        

    
    
    
def TREEPOLICY(node):
    #
    while not node.state==n: #terminal(n번째 node) 에서 terminate
        if reward is under threshold:
            terminate and backpropagation
        elif 



class State():
	NUM_TURNS = 10	
	GOAL = 0
	MOVES=[2,-2,3,-3]
	MAX_VALUE= (5.0*(NUM_TURNS-1)*NUM_TURNS)/2
	num_moves=len(MOVES)
	def __init__(self, value=0, moves=[], turn=NUM_TURNS):
		self.value=value
		self.turn=turn
		self.moves=moves
	def next_state(self):
		nextmove=random.choice([x*self.turn for x  in self.MOVES])
		next=State(self.value+nextmove, self.moves+[nextmove],self.turn-1)
		return next
	def terminal(self):
		if self.turn == 0:
			return True
		return False
	def reward(self):
		r = 1.0-(abs(self.value-self.GOAL)/self.MAX_VALUE)
		return r
	def __hash__(self):
		return int(hashlib.md5(str(self.moves).encode('utf-8')).hexdigest(),16)
	def __eq__(self,other):
		if hash(self)==hash(other):
			return True
		return False
	def __repr__(self):
		s="Value: %d; Moves: %s"%(self.value,self.moves)
		return s
	

class Node():
	def __init__(self, state, parent=None):
		self.visits=1
		self.reward=0.0	
		self.state=state
		self.children=[]
		self.parent=parent	
	def add_child(self,child_state):
		child=Node(child_state,self)
		self.children.append(child)
	def update(self,reward):
		self.reward+=reward
		self.visits+=1
	def fully_expanded(self):
		if len(self.children)==self.state.num_moves:
			return True
		return False
	def __repr__(self):
		s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
		return s
		


def UCTSEARCH(budget,root):
	for iter in range(int(budget)):
		if iter%10000==9999:
			logger.info("simulation: %d"%iter)
			logger.info(root)
		front=TREEPOLICY(root)
		reward=DEFAULTPOLICY(front.state)
		BACKUP(front,reward)
	return BESTCHILD(root,0)

def TREEPOLICY(node):
	#a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
	while node.state.terminal()==False:
		if len(node.children)==0:
			return EXPAND(node)
		elif random.uniform(0,1)<.5:
			node=BESTCHILD(node,SCALAR)
		else:
			if node.fully_expanded()==False:	
				return EXPAND(node)
			else:
				node=BESTCHILD(node,SCALAR)
	return node

def EXPAND(node):
	tried_children=[c.state for c in node.children]
	new_state=node.state.next_state()
	while new_state in tried_children:
		new_state=node.state.next_state()
	node.add_child(new_state)
	return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar):
	bestscore=0.0
	bestchildren=[]
	for c in node.children:
		exploit=c.reward/c.visits
		explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))	
		score=exploit+scalar*explore
		if score==bestscore:
			bestchildren.append(c)
		if score>bestscore:
			bestchildren=[c]
			bestscore=score
	if len(bestchildren)==0:
		logger.warn("OOPS: no best child found, probably fatal")
	return random.choice(bestchildren)

def DEFAULTPOLICY(state):
	while state.terminal()==False:
		state=state.next_state()
	return state.reward()

def BACKUP(node,reward):
	while node!=None:
		node.visits+=1
		node.reward+=reward
		node=node.parent
	return

if __name__=="__pseudo__":
	parser = argparse.ArgumentParser(description='MCTS research code')
	parser.add_argument('--num_sims', action="store", required=True, type=int)
	parser.add_argument('--levels', action="store", required=True, type=int, choices=range(State.NUM_TURNS))
	args=parser.parse_args()
	
	current_node=Node(State())
	for l in range(args.levels):
		current_node=UCTSEARCH(args.num_sims/(l+1),current_node)
		print("level %d"%l)
		print("Num Children: %d"%len(current_node.children))
		for i,c in enumerate(current_node.children):
			print(i,c)
		print("Best Child: %s"%current_node.state)
		
		print("--------------------------------")	
			
	
