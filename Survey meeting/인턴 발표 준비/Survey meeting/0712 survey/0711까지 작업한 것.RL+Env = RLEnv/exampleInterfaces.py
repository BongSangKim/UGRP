class StateInterface():
    def getPossibleActions(self):
        raise NotImplementedError()

    def takeAction(self, action):
        raise NotImplementedError()

    def isTerminal(self):
        raise NotImplementedError()

    def getReward(self):
        # only needed for terminal states
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()


class ActionInterface():
    def __eq__(self, other):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError()
