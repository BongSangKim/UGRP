from __future__ import division  #python 2.x 호환. 없어도 상관없음
import time, math, random, gym
#from gym import space, logger
#from gym.utils import seeding
import numpy as np
import tensorflow as tf
#import pandas as pd

#__dict__로 변수 내부 확인가능, 디버그할때 참고
#=====================Environment code=========================================
class UDNEnv(gym.Env):
	def __init__(self):  #변수 선언
		self.d_attenuation = -4 #거리감쇠 alpha값(NLoS)
		self.BSposition = np.transpose(np.loadtxt('BSposition.csv', delimiter=','))  #BS위치는 불변, ppp.py로 csv파일 생성하여 사용, ppp.py는 pandas 사용, 경로 일치시키기. 1 by n인듯
		self.BSnum = len(self.BSposition) #BSnum=n
		self.state = tf.ones([1,self.BSnum])
		self.Area = 10000  #면적
		self.usernum = 16  #UE 개수
		self.done = False  #done값 이 True일때 terminate
		#self.BSpower = tf.ones([1,BSnum])  #BS energy consumptino을 고려하여 추가해둠
		self.d_at_tensor = tf.fill([1,self.usernum],-4)
		self.UE_Xposition = tf.random.uniform([1,self.usernum],0,self.Area)  #UE x,y를 랜덤으로 뿌린것(ppp는 아님)
		self.UE_Yposition = tf.random.uniform([1,self.usernum],0,self.Area)
		self.reward = None
		self.BS_user_distance = tf.zeros([self.BSnum,self.usernum,2]) 
		self.BSdistance = tf.zeros([self.BSnum,self.usernum]) 
		self.user_association = tf.zeros([1,self.usernum])  
		self.SNR = None
		self.timeLimit = 10000 #mcts 코드에서 작동시키기 위해 mcts.py에서 UDNEnv class로 코드 이동

	def step(self, action):
		state = action
		self.Econsumption = tf.reduce_sum(state)   #state가 1차원 벡터니까 reduce_sum은 스칼라. Econsumption은 total값임

		#BS-User 거리 계산
		for i in range(self.BSnum):
			for j in range(self.usernum):
				self.BS_user_distance[i][j][0] = self.BSposition[i][0] - self.UE_Xposition[j]
				self.BS_user_distance[i][j][1] = self.BSposition[i][1] - self.UE_Yposition[j]
		self.BSdistance = np.linalg.norm(self.BS_user_distance, axis=2, ord = 2)
		self.distance = np.array(self.BSdistance)    #np.array랑  tf행렬이랑 계산되나?? tf쪽으로 계산되는듯
		#거리 계산 종료

		self.user_association = self.association(self.state, self.BSdistance)
		self.SNR = tf.pow(self.user_association,self.d_at_tensor) #SNR 계산

		self.reward = tf.reduce_sum(self.SNR) / self.Econsumption #reward 계산
		self.UE_Xposition = tf.random.uniform([1,self.usernum],0,self.Area) #유저 위치 랜덤 배치
		self.UE_Yposition = tf.random.uniform([1,self.usernum],0,self.Area)
		return self.reward, self.state

		#BS-User association 하는 함수
	def association(self, state, distance):
		association_distance = tf.zeros([1,self.usernum])
		associationdistance = [self.BSnum, self.usernum]

		#전원 off되어 있는 BS에서 유저까지의 거리를 무한대로 설정
		for i in len(state):
			if state[i] == 0:
				for j in len(self.usernum):
					associationdistance[i][j] = math.inf  #나중에 계산 오류날 수도 있으니까 inf값 대신 area밖으로 나가는 특정값을 쓰는 것 고려.
		association_distance = tf.math.reduce_min(associationdistance, axis = 0)
		return association_distance

	def reset(self):
		self.state = tf.ones([1,self.BSnum])
		self.done = False
		return self.state
	################mcts.py에서 사용할 함수부분 ############################
	def isTerminal(self): #일단 시간조건
		if time.time() > self.timeLimit: #state is terminal 
			return True
		else: #state is nonterminal
			return False
	def getPossibleActions(self):
		pass
	def takeAction(action): #action=mcts.search(initialstate=Env.state)
		pass
	def getReward(self):
		pass
Env = UDNEnv() #Env로 인스턴스 호출, mcts.py에서 Env를 호출하여 사용
#=====================Environment code=========================================

#############################RL code-MCTS######################################
'''
##############################original code####################################
class state():
    def isTerminal():
        pass
        #return state가 terminal인지 아닌지, boolean인가??
        #terminal 조건: tree search 실행 시간, iteration 횟수 등, 우리는 노드가 n번째일 때에도 Terminal이게 짜야한다. 아마 이 조건들은 or 연산자로 묶일듯. timelimit 또는 iterationlimit중 하나만 쓰는게 편할 듯
    def getPossibleActions(): #Returns an iterable of all actions which can be taken from this state
        pass
    def takeAction(action): #Returns the state which results from taking action
        pass
    def getReward(): #Returns the reward for this state. Only needed for terminal states.
        pass
##############################original code####################################
'''
#######state class 제거 및 함수변수로 UDNEnv.state 사용#########################



#state를 class로 사용하지 않으면, state 클래스 밑에 있는 함수 네개는 따로 정의한뒤에, mcts 라이브러리에 있는 state.def() 부분을 def(UDNenv.state)형태로 바꾸면 된다.
#이렇게 하면 RL코드를 크게 수정하지 않고 돌릴 수 있을것 같음
###############################################################################    

def randomPolicy(state):
	while not Env.isTerminal():   #state.isTerminal() 등 함수 4개는 Env.isTerminal()형태로 
		try:
			action = random.choice(state.getPossibleActions())  #random.choice('아마 iterable변수')=하나 random으로 골라 return해줌
		except IndexError:
			raise Exception("Non-terminal state has no possible actions: " + str(state))
		state = state.takeAction(action) #action에 따라 state 업데이트
	return state.getReward()


class treeNode():                               #트리 노드 정의. 노드에 state 정해주면, state.isTerminal()값에 따라 노드가 터미널노드인지 결정됨
	def __init__(self, state, parent):
		self.state = state #Env.state
		self.isTerminal = Env.isTerminal()   
		self.isFullyExpanded = self.isTerminal
		self.parent = parent
		self.numVisits = 0
		self.totalReward = 0
		self.children = {}



class mcts():                   #explorationConstant는 값을 바꾸어 학습시킬 수 있다. 
	def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2), rolloutPolicy=randomPolicy):
		#timeLimit,iterationLimit는 입력안하면 None이 초기값, 둘다 입력안하면 바로 아래 ValueError나옴
		#rolloutPolicy는 따로 입력해주지 않으면 randompolicy인데, 나중에 UCB1값을 비교하는 식으로 하는 것이 좋을듯.
		#처음에는 라이브러리 기본대로 randompolicy함수 만들어서 하고, 이후 UCB1 함수 따로 만들어서 rolloutPolicy 변수값을 UCB1으로 넣어보기
		if timeLimit != None:
			if iterationLimit != None:
				raise ValueError("Cannot have both a time limit and an iteration limit")  #
			# time taken for each MCTS search in milliseconds
			self.timeLimit = timeLimit
			self.limitType = 'time'
		else: #timeLimit = None일때
			if iterationLimit == None:
				raise ValueError("Must have either a time limit or an iteration limit")
			# number of iterations of the search
			if iterationLimit < 1:
				raise ValueError("Iteration limit must be greater than one")
			self.searchLimit = iterationLimit
			self.limitType = 'iterations'
		self.explorationConstant = explorationConstant
		self.rollout = rolloutPolicy

	def search(self, initialState):  #initialstate=Env.state
		self.root = treeNode(initialState, None) #라이브러리 앞에 있는 treenode class
		print(self.limitType)
		if self.limitType == 'time':  #limitType이 time일때
			timeLimit = time.time() + self.timeLimit / 1000
			while time.time() < timeLimit: #timeLimit전까지 계속 search.....
				self.executeRound()
		else: #limitType 이 iterations일때
			for i in range(self.searchLimit):
				self.executeRound()  #SearchLmit전까지 계속 search...

		bestChild = self.getBestChild(self.root, 0)
		return self.getAction(self.root, bestChild)

	def executeRound(self):
		node = self.selectNode(self.root)
		reward = self.rollout(node.state)
		self.backpropogate(node, reward)

	def selectNode(self, node):
		while not node.isTerminal:
			if node.isFullyExpanded:
				node = self.getBestChild(node, self.explorationConstant)
			else:
				return self.expand(node)
		return node

	def expand(self, node):
		actions = node.state.getPossibleActions()
		for action in actions:
			if action not in node.children:
				newNode = treeNode(node.state.takeAction(action), node)
				node.children[action] = newNode
				if len(actions) == len(node.children):
					node.isFullyExpanded = True
				return newNode

		raise Exception("Should never reach here")

	def backpropogate(self, node, reward):
		while node is not None:
			node.numVisits += 1
			node.totalReward += reward
			node = node.parent

	def getBestChild(self, node, explorationValue):
		bestValue = float("-inf")
		bestNodes = []
		for child in node.children.values():
			nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
			if nodeValue > bestValue:
				bestValue = nodeValue
				bestNodes = [child]
			elif nodeValue == bestValue:
				bestNodes.append(child)
		return random.choice(bestNodes)

	def getAction(self, root, bestChild):
		for action, node in root.children.items():
			if node is bestChild:
				return action


test=Env.state
initialState = Env.state
action = mcts.search(Env,initialState=initialState)