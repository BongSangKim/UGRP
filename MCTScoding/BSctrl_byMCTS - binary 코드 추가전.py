import time, math, random, gym
import numpy as np
#import tensorflow as tf 
#7월 16일 텐서포기, np.array로 교체(tensor는 indexing 안됨)
#__dict__로 변수 내부 확인가능, 디버그할때 참고
#=====================Environment code=========================================
class UDNEnv(gym.Env):
	def __init__(self):
		self.d_attenuation = -4 #거리감쇠 alpha값(NLoS)
		self.BSposition = np.transpose(np.loadtxt('BSposition.csv', delimiter=','))
		#BSposition.csv는 ppp.py로 생성. 1 by n
		self.BSnum = len(self.BSposition) #BSnum=n
		self.state = np.ones((1,self.BSnum)) #BS on으로 초기화
		self.Area = 10000  #면적
		self.usernum = 16  #UE 개수
		#self.done = False  #True일때 terminate. isTerminal(self)가 있어서 사용안함
		#self.BSpower = np.ones((1,BSnum))  #BS energy consumptino을 고려하여 추가해둠
		self.d_at_tensor = np.full((1,self.usernum),-4)
		self.UE_Xposition = np.random.uniform(0,self.Area,(1,self.usernum))  #UE x,y를 랜덤으로 뿌린것(ppp는 아님)
		self.UE_Yposition = np.random.uniform(0,self.Area,(1,self.usernum))
		self.reward = None #getReward함수 변수로 쓰여야할듯
		self.BS_user_distance = np.zeros((2,self.BSnum,self.usernum))
		self.BSdistance = np.zeros((self.BSnum,self.usernum)) #BSnum by usernum 크기의 텐서 with all elements 0
		self.user_association = np.zeros([1,self.usernum])  #1 by usernum 크기의 텐서 with all elements 0
		self.SNR = None
		#self.timeLimit = 10000 #mcts.py에서 가져옴. isTerminal() 조건으로 시간 쓸때 사용. 시간조건 쓸 경우 그냥 isTerminal 변수로 넣고 여기는 날리기
		#self.possibleActions = np.ones((1,self.BSnum)) ############ 이거를 1 by BSnum인 list로 만들어서 callable하게.
		self.possibleActions = [1]*self.BSnum #numpy.ndarray는 callable하지 않기 때문에 getPossibleActions()함수 사용불가해짐. 따라서 임시로 list형으로 만드는 코드
		#print('possibleActions:',self.possibleActions)
		self.dtime = 0

	def step(self, action):
		print('$ step')
		state = action #takeAction 함수에서 newstate=action함
		self.Econsumption = np.sum(self.state)   #state가 1차원 벡터니까 reduce_sum은 스칼라. Econsumption은 total값임
		#BS-User 거리 계산
		#print(self.BSnum,'by',self.usernum)
		#print(self.BS_user_distance)
		#print(self.UE_Xposition)
		for i in range(self.BSnum):
			for j in range(self.usernum):
				self.BS_user_distance[0][i][j] = self.BSposition[i][0] - self.UE_Xposition[0][j]
				self.BS_user_distance[1][i][j] = self.BSposition[i][1] - self.UE_Yposition[0][j]
		self.BSdistance = np.linalg.norm(self.BS_user_distance, axis=1, ord = 2)
		self.distance = np.array(self.BSdistance)
		#거리 계산 종료

		self.user_association = self.association(self.state, self.BSdistance)
		self.user_association = float(self.user_association)  #SNR -4제곱하면 분수형이므로 float로 바꿔줘야 작동
		#print(self.user_association,'UE association') 
		self.SNR = np.power(self.user_association,-4) #SNR 계산

		self.reward = np.sum(self.SNR) / self.Econsumption #reward 계산
		#self.reward = np.to_float(self.reward)
		#print('set함수에서 rewardtype:',type(self.reward))
		self.UE_Xposition = np.random.uniform(0,self.Area,(1,self.usernum)) #유저 위치 랜덤 배치
		self.UE_Yposition = np.random.uniform(0,self.Area,(1,self.usernum))
		self.dtime += 1
		return self.SNR,self.reward, self.state
		
		
		#BS-User association 하는 함수
	def association(self, state, distance):
		print('$ association')
		association_distance = np.zeros((1,self.usernum))
		associationdistance = [self.BSnum, self.usernum]

		#전원 off되어 있는 BS에서 유저까지의 거리를 무한대로 설정
		#print(len(state[0]))
		#print(state,'state')
		#print(state[0])
		for i in range(len(state[0])): #state 값이 [[1 1 1 1]]형태여서 state[0]으로 사용
			if state[0][i] == 0:
				for j in len(self.usernum):
					associationdistance[i][j] = math.inf 
		association_distance = np.amin(associationdistance, axis = 0)
		return association_distance

	def reset(self):
		print('$ reset')
		self.state = np.ones((1,self.BSnum))
		self.done = False
		return self.state
	################mcts.py에서 사용할 함수부분 ############################
	def isTerminal(self): #일단 시간조건->SNR threshold로
		print('$ isTerminal() 실행:',end=' ')
		print('isTerminal is:',end=' ')
		self.timeLimit = 10
		#self.startTime = 0
		if self.dtime > self.timeLimit: #state is terminal 
			print('TRUE')
			return True
		else: #state is nonterminal
			print('FALSE')
			return False
		'''print('$ isTerminal() 실행:',end=' ')
		self.threshold = 0.000001
		#return False #terminal node나 SNR 터미널 조건 없다치고 ㄱㄱ!
		newSNR, newReward, newState = self.step(action)
		print('newSNR is:',newSNR)
		if newSNR < self.threshold:
			print("isTerminal is TRUE!!")
			return True
		else:
			return False
			#SNR 조건 넣고 싶을때 
		'''

	def getPossibleActions(self):
		print('$ getPosssibleActions')
		#state와 차원이 같고 0 or 1값을 가지는 텐서. action->state
		self.possibleActions = [0,1,2,3]
		return self.possibleActions
		'''
		#가능한 Actions 모두 출력하게 하기.
		#예시) [[0,0,0,0],....[1,1,1,1]] (BSnum=4일때)
		
		for i in range(self.BSnum):
		#	for i in range(self.BSnum):
			self.possibleActions[i]=random.randrange(0,2) #0 or 1 값
		return self.possibleActions
		'''
	def takeAction(self, action):  
		print('$ takeAction')
		newSNR, newReward, newState = self.step(action)
		return newState
	
	def getReward(action): 
		print('$ getReward')
		newSNR, newReward, newState = self.step(action)
		return newReward

Env = UDNEnv() #Env로 인스턴스 호출, mcts.py에서 Env를 호출하여 사용
#=====================Environment code=========================================

#######state class 제거 및 함수변수로 UDNEnv.state 사용#########################
#state를 class로 사용하지 않으면, state 클래스 밑에 있는 함수 네개는 따로 정의한뒤에, mcts 라이브러리에 있는 state.def() 부분을 def(UDNenv.state)형태로 바꾸면 된다.
#이렇게 하면 RL코드를 크게 수정하지 않고 돌릴 수 있을것 같음
###############################################################################    
def randomPolicy(state):
	print('$ RandomPolicy')
	while not Env.isTerminal():   #state.isTerminal() 등 함수 4개는 Env.isTerminal()형태로 
		print('state is not in terminal')
		try:
			action = random.choice(Env.getPossibleActions()) #random.choice('아마 iterable변수')=하나 random으로 골라 return해줌
		except IndexError:
			raise Exception("Non-terminal state has no possible actions: " + str(state))
		state = Env.takeAction(action) #action에 따라 state 업데이트
	print('state is in terminal, terminate randomPolicy, return reward at state\n -"state is not in terminal"이 출력이 안되면 isTerminal함수가 True만 return하는 상태입니다')
	#return Env.getReward(action)
	#=====return the reward at state=====
	return 5 #일단 pass, 터미널 state에서 reward 리턴하게끔 하기. step함수 수정이 필요할 수 있음.


class treeNode():	#트리 노드 정의. 노드에 state 정해주면, state.isTerminal()값에 따라 노드가 터미널노드인지 결정됨
	def __init__(self, state, parent):
		print('$ treeNode(class)')
		self.state = state #state = state 를 state = Env.state로 바꿈 #step Func에서 state가져옴
		self.isTerminal = Env.isTerminal()   
		self.isFullyExpanded = self.isTerminal
		self.parent = parent
		self.numVisits = 0
		self.totalReward = 0
		self.children = {}
		self.getPossibleActions = Env.getPossibleActions()
		self.takeAction = Env.takeAction
class mcts():  #explorationConstant는 값을 바꾸어 학습시킬 수 있다. 
	def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2), rolloutPolicy=randomPolicy):
		#timeLimit,iterationLimit는 입력안하면 None이 초기값, 둘다 입력안하면 바로 아래 ValueError나옴
		#rolloutPolicy는 따로 입력해주지 않으면 randompolicy인데, 나중에 UCB1값을 비교하는 식으로 하는 것이 좋을듯.
		#처음에는 라이브러리 기본대로 randompolicy함수 만들어서 하고, 이후 UCB1 함수 따로 만들어서 rolloutPolicy 변수값을 UCB1으로 넣어보기
		print('$ MCTS(class)')
		if timeLimit != None:
			if iterationLimit != None:
				raise ValueError("Cannot have both a time limit and an iteration limit")
			# time taken for each MCTS search in milliseconds
			self.timeLimit = timeLimit
			self.limitType = 'time'
		else:
			if iterationLimit == None:
				raise ValueError("Must have either a time limit or an iteration limit")
			# number of iterations of the search
			if iterationLimit < 1:
				raise ValueError("Iteration limit must be greater than one")
			print('LimitType: iteration')
			self.searchLimit = iterationLimit
			self.limitType = 'iterations'
		self.explorationConstant = explorationConstant
		self.rollout = rolloutPolicy

	def search(self, initialState):  #initialstate=Env.state
		print('$ search start')
		self.root = treeNode(initialState, None) #state=initialState, parent=None
		print('searching... until searchlimit')
		if self.limitType == 'time': 
			timeLimit = time.time() + self.timeLimit / 1000
			while time.time() < timeLimit: 
				self.executeRound()
		else: 
			for i in range(self.searchLimit):
				self.executeRound()  
		bestChild = self.getBestChild(self.root, 0)
		print('search over...')
		return self.getAction(self.root, bestChild)

	def executeRound(self):
		print('$ executeRound')
		node = self.selectNode(self.root)
		reward = self.rollout(node.state)
		self.backpropagate(node, reward) #reward가 지금 list값. node가 int값

	def selectNode(self, node):
		print('$ selectNode')
		while not node.isTerminal:
			print('node.isTerminal is: False')
			if node.isFullyExpanded:
				node = self.getBestChild(node, self.explorationConstant)
			else:
				return self.expand(node)
		print('node.isTerminal is: True\n -"node.isTerminal is: False"가 출력이 안되면 isTerminal함수가 True만 return하는 상태입니다')
		return node

	def expand(self, node):
		print('$ expand')
		################################### 임시로 pass 코드
		#actions = [0,1,1,1,1,1,1] 
		################################### 임시로 pass코드
		actions =Env.getPossibleActions()
		#actions = node.getPossibleActions() #원래 코드, node에 따라야 하니까 이게 맞음
		for action in actions:
			if action not in node.children:
				newNode = treeNode(node.takeAction(action), node)
				node.children[action] = newNode 
				if len(actions) == len(node.children):
					node.isFullyExpanded = True
				return newNode
		raise Exception("Should never reach here")

	def backpropagate(self, node, reward):
		while node is not None:
			node.numVisits += 1
			node.totalReward += reward  
			node = node.parent
			print('$ backpropagate')
	def getBestChild(self, node, explorationValue):
		print('$ getBestChild')
		bestValue = float("-inf")
		#bestNodes = [1,2,3] #오류 pass, bestNodes가 update가 안되고 있음
		bestNodes = []
		for child in node.children.values():
			nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
			if nodeValue > bestValue:
				bestValue = nodeValue
				bestNodes = [child]
			elif nodeValue == bestValue:
				bestNodes.append(child)
		print('====getBestChilde is:',random.choice(bestNodes))
		return random.choice(bestNodes)

	def getAction(self, root, bestChild):
		print('$ getAction')
		for action, node in root.children.items():
			if node is bestChild:
				return action

initialState = Env.state
MCTS=mcts(None,1)
action = MCTS.search(initialState)
print('=============변수 체크용=============')
print('BSnum개수',Env.BSnum)
print('Env.state is:',Env.state)
#print('Env.isTerminal is:',Env.isTerminal())
print('possibleActions are:',Env.possibleActions)
print('learnt action is:',action)