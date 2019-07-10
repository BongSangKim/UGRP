from __future__ import division  #python 2.x 호환

import time        #time.time() 사용, 현재시간 호출
import math        #제곱근, log 사용
import random      #random.choice()사용

###################라이브러리 쓸 때 입력해야하는 것##############################
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
    
#state를 class로 사용하지 않으면, state 클래스 밑에 있는 함수 네개는 따로 정의한뒤에, mcts 라이브러리에 있는 state.def() 부분을 def(UDNenv.state)형태로 바꾸면 된다.
#이렇게 하면 RL코드를 크게 수정하지 않고 돌릴 수 있을것 같음
###############################################################################    

def randomPolicy(state):
    while not state.isTerminal():   #state class안에 isTerminal을 가져오는 것
        try:
            action = random.choice(state.getPossibleActions())  #random.choice('아마 iterable변수')=하나 random으로 골라 return해줌
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        state = state.takeAction(action) #action에 따라 state 업데이트
    return state.getReward()


class treeNode():                               #트리 노드 정의. 노드에 state 정해주면, state.isTerminal()값에 따라 노드가 터미널노드인지 결정됨
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
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


    def search(self, initialState):             #라이브러리 실행 이전에 initialState=UDNEnv.state 코드 한줄 넣기
        self.root = treeNode(initialState, None) #라이브러리 앞에 있는 treenode class

        if self.limitType == 'time':  #limitType이 time일때
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit #timeLimit전까지 계속 search.....
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
