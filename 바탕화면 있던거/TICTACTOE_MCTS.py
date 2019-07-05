from math import *
import random
import copy


class TicTacToe:
    def __init__(self, state = [0,0,0,0,0,0,0,0,0]):
        self.playerJustMoved = 2
        self.state = state
        
    def Clone(self):
        state = TicTacToe()
        state.state = self.state[:]
        state.playerJustMoved = self.playerJustMoved
        return state
    
    def DoMove(self, move):
        assert move >= 0 and move <= 8 and self.state[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.state[move] = self.playerJustMoved
        
        
    def GetMoves(self):
        if self.checkState() != 0:
            return []
        
        else:
            moves = []
            for i in range(9):
                if self.state[i] == 0:
                    moves.append(i)
                    
            return moves
        
    def GetResult(self, playerjm):
        result = self.checkState()
        assert result != 0
        if result == -1:
            return 0.5
        
        elif result == playerjm:
            return 1.0
        else:
            return 0.0
        
        
    def checkState(self):
        for (x,y,z) in [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]:
            if self.state[x] == self.state[y] == self.state[z]:
                if self.state[x] == 1:
                    return 1
                elif self.state[x] == 2:
                    return 2
                
        if [i for i in range(9) if self.state[i] == 0] == []:
            return -1
        return 0
    
    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".0X"[self.state[i]]
            if i % 3 == 2:
                s += "\n"
        return s
        
class Node:
    def __init__(self, move = None, parent = None, state = None):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()
        self.playerJustMoved = state.playerJustMoved
        
    def UCTSelectChild(self):
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2 * log(self.visits) / c.visits))
        return s[-1]
    
    def AddChild(self, m ,s):
        n = Node(move = m, parent = self, state = copy.deepcopy(s))
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        self.visits += 1
        self.wins += result
        
    def __repr__(self):
        return "[M" + str(self.move) + " W/V " + str(self.wins) + "/" + str(self.visits) + " U" + str(self.untriedMoves) + "]"
    
    def ChildrenToString(self):
        s =""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s
    
    
def UCT(rootstate, itermax):
    rootnode = Node(state = rootstate)
    
    for i in range(itermax):
        node = rootnode
        state = copy.deepcopy(rootstate)
        
        #selection
        while node.untriedMoves == [] and node.childNodes != []:
            node = node.UCTSelectChild()
            state.DoMove(node.move)
        
        #Expansion
        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)
        
        #simulation
        while state.GetMoves() != []:
            state.DoMove(random.choice(state.GetMoves()))
        
        #BackPropagation
        while node != None:
            node.Update(state.GetResult(node.playerJustMoved))
            node = node.parentNode
            
    print rootnode.ChildrenToString()
    
    s = sorted(rootnode.childNodes, key = lambda c: c.wins/c.visits)
    return sorted(s, key = lambda c: c.visits)[-1].move
    
        
def UCTPlayGame():
    state = TicTacToe()
    while state.GetMoves() != []:
        print str(state)
        if state.playerJustMoved == 2:
            rootstate = copy.deepcopy(state)
            m = UCT(rootstate, itermax = 10000)
        else:
            m = input("which Do you want? : ")
            m -= 1
        print "Best Move : " + str(m) + "\n"
        state.DoMove(m)
        
    if state.GetResult(state.playerJustMoved) == 1.0:
        print "Player " + str(state.playerJustMoved) + " Wins!!"
 
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print "Payer " + str(3 - state.playerJustMoved) + " Wins!!"

    else: print "Draw!!"
    
    
if __name__ == "__main__":
    UCTPlayGame() 
            
