from __future__ import division  #python 2.x 호환

import time        #time.time() 사용, 현재시간 호출
import math        #제곱근, log 사용
import random      #random.choice()사용

###################라이브러리 쓸 때 입력해야하는 것##############################
class state():
    def isTerminal():
        pass
        #return state가 terminal인지 아닌지, boolean인가??
        #terminal 조건: tree search 실행 시간, 노드 만든 횟수 등,
    def getPossibleActions(): #Returns an iterable of all actions which can be taken from this state
        pass
    def takeAction(action): #Returns the state which results from taking action
        pass
    def getReward(): #Returns the reward for this state. Only needed for terminal states.
        pass
    
###############################################################################    

def randomPolicy(state):
    while not state.isTerminal():   #state class안에 isTerminal을 가져오는 것
        try:
            action = random.choice(state.getPossibleActions())  #random.choice('아마 iterable변수')=하나 random으로 골라 return해줌
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action)
    return state.getReward()


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


class mcts():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
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
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

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
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
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


import math
import gym
#from gym import space, logger
#from gym.utils import seeding
import numpy as np
import tensorflow as tf
#import pandas as pd


class UDNEnv(gym.Env):

	def __init__(self):
		#변수 선언
		self.d_attenuation = -4
		self.state = None
		self.BSposition = np.transpose(np.loadtxt('BSposition.csv', delimiter=','))
		self.BSnum = len(BSposition)
		self.Area = 10000
		self.usernum = 16
		self.done = False
		self.BSpower = tf.ones([1,BSnum])
		self.d_at_tensor = tf.fill([1,usernum],-4)
		self.US_Xposition = tf.random_uniform([1,usernum],0,Area)
		self.US_Yposition = tf.random_uniform([1,usernum],0,Area)
		self.reward = None
		self.BS_user_distance = tf.zeros(BSnum,usernum,2)
		self.BSdistance = tf.zeros(BSnum,usernum)
		self.user_assosiation = tf.zeros([1,usernum])
		self.SNR = None

	def step(self, action):
		state = self.state
		state = action
		Econsumption = tf.reduce_sum(state)
		distance = np.array(BSdistance)
		
		#BS-User 거리 계산
		for i in range(BSnum):
			for j in range(usernum):
				BS_user_distance[i][j][0] = BSposition[i][0] - US_Xposition[j]
				BS_user_distance[i][j][1] = BSposition[i][1] - US_Yposition[j]
		BSdistance = np.linalg.norm(BS_user_distance, axis=2, ord = 2)
		#거리 계산 종료
		
		user_assosiation = assosiation(state, BSdistance)
		SNR = tf.pow(user_assosiation,d_at_tensor) #SNR 계산

		reward = tf.reduce_sum(SNR) / tf.reduce_sum(state) #reward 계산

		self.US_Xposition = tf.random_uniform([1,usernum],0,Area) #유저 위치 랜덤 배치
		self.US_Yposition = tf.random_uniform([1,usernum],0,Area)
		return reward, state

		#BS-User assosiation 하는 함수
	def assosiation(self, state, distance):
		assosiation_distance = tf.zeros([1,usernum])
		assosiationdistance = [BSnum, usernum]

		#전원 off되어 있는 BS에서 유저까지의 거리를 무한대로 설정
		for i in len(state):
			if state[i] == 0:
				for j in len(usernum):
					assosiationdistance[i][j] = math.inf
		
		assosiation_distance = tf.math.reduce_min(assosiationdistance, axis = 0)
		return assosiation_distance


	def reset(self):
		self.state = self.tf.ones([1,BSnum])
		self.done = False
		return state

