# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax(agentIndex, depth, gameState):
            # Terminal state or depth limit reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman (Maximizing agent)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Ghosts (Minimizing agents)
            else:
                return minValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            maxScore = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                score = minimax(1, depth, successor)
                maxScore = max(maxScore, score)
            return maxScore

        def minValue(agentIndex, depth, gameState):
            minScore = float('inf')
            nextAgent = agentIndex + 1
            numAgents = gameState.getNumAgents()
            # If this is the last ghost, the next agent will be Pacman and depth will increase
            if agentIndex == numAgents - 1:
                nextAgent = 0
                depth += 1
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                score = minimax(nextAgent, depth, successor)
                minScore = min(minScore, score)
            return minScore

        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning.
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction.
        """

        def maxValue(state, depth, alpha, beta): #function to calculate the maximum value for the Pacman agent
            #if the game is over or the maximum depth is reached, return the evaluation function value
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            maxEval = float('-inf')
            for action in state.getLegalActions(0): #get all legal actions for Pacman (agent index 0)
                successor = state.generateSuccessor(0, action)
                maxEval = max(maxEval, minValue(successor, depth, 1, alpha, beta)) #recursively call minValue for the ghost
                if maxEval > beta:
                    return maxEval
                alpha = max(alpha, maxEval)
            return maxEval

        def minValue(state, depth, agentIndex, alpha, beta): #function to calculate the minimum value for ghost agents
            #if the game is over, return the evaluation function value
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            minEval = float('inf')
            numAgents = state.getNumAgents() #number of agents in the game
            for action in state.getLegalActions(agentIndex): #get all legal actions for the ghost agent
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == numAgents - 1: #last ghost, go to next depth and call maxValue
                    minEval = min(minEval, maxValue(successor, depth + 1, alpha, beta))
                else:
                    minEval = min(minEval, minValue(successor, depth, agentIndex + 1, alpha, beta))
                if minEval < alpha:
                    return minEval
                beta = min(beta, minEval)
            return minEval

        bestAction = None #variable to store the best action
        alpha = float('-inf')
        beta = float('inf')
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0): #evaluate each legal action for Pacman (agent index 0)
            successor = gameState.generateSuccessor(0, action)
            score = minValue(successor, 0, 1, alpha, beta)
            if score > bestScore: #update bestScore and bestAction if the current score is higher than the bestScore
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def maxValue(state, depth): #function to calculate the maximum value for the Pacman agent
            newDepth = depth + 1 #increase the depth since Pacman has taken a turn
            #if the game is over or the maximum depth is reached, return the evaluation function value
            if state.isWin() or state.isLose() or newDepth == self.depth:
                return self.evaluationFunction(state)
            maxEval = float('-inf')
            actions = state.getLegalActions(0) #get all legal actions for Pacman (agent index 0)
            for action in actions: #evaluate each action and choose the one with the highest value
                successor = state.generateSuccessor(0, action)  #generate the successor state
                maxEval = max(maxEval, expectValue(successor, newDepth, 1)) #gecursively call expectValue for the ghost
            return maxEval

        def expectValue(state, depth, agentIndex): #function to calculate the expected value for ghost agents
            if state.isWin() or state.isLose():  #if the game is over, return the evaluation function value
                return self.evaluationFunction(state)
            actions = state.getLegalActions(agentIndex) #get all legal actions for the ghost agent
            expectedValue = 0 #initialize expectedValue to 0
            numActions = len(actions) #number of legal actions
            for action in actions: #evaluate each action and accumulate the expected value
                successor = state.generateSuccessor(agentIndex, action) #generate the successor state
                #if the agent is the last ghost, call maxValue for Pacman's turn
                if agentIndex == state.getNumAgents() - 1:
                    expectedValue += maxValue(successor, depth)
                else:
                    expectedValue += expectValue(successor, depth, agentIndex + 1) #recursively call expectValue for the next ghost
            if numActions == 0:
                return 0
            return expectedValue / numActions

        bestAction = None #to store the best action
        bestScore = float('-inf')

        for action in gameState.getLegalActions(0):  #evaluate each legal action for Pacman (agent index 0)
            successor = gameState.generateSuccessor(0, action) #generate the successor state
            score = expectValue(successor, 0, 1) #get the expected value for the successor state
            if score > bestScore: #update bestScore and bestAction if the current score is higher than the bestScore
                bestScore = score
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    DESCRIPTION:
        This evaluation function divides the final score of the state into two parts:
        1. When the ghosts are scared (identified by scaredTimes > 0).
        2. Normal ghosts.

        Common evaluation score between both parts includes the current score, the reciprocal of the sum of food distances, and the number of remaining food pellets.

        In the first case, from the total score we subtract the distance to the ghosts and the number of power pellets, as the ghosts are currently scared. So, the closer Pacman is to the ghosts, the better the score.

        In the second case, since the ghosts are not scared, we add the distance to the ghosts and the number of power pellets to the total score.
    """
    pacmanPos = currentGameState.getPacmanPosition() #get Pacman's position
    foodGrid = currentGameState.getFood() #get the food grid
    ghostStates = currentGameState.getGhostStates() #get the ghost states
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates] #get the scared timers for each ghost

    #Manhattan distance to all food pellets
    foodPositions = foodGrid.asList()
    from util import manhattanDistance
    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodPositions]

    #Manhattan distance to each ghost
    ghostPositions = [ghostState.getPosition() for ghostState in ghostStates]
    ghostDistances = [manhattanDistance(pacmanPos, ghostPos) for ghostPos in ghostPositions]

    #number of power pellets
    numPowerPellets = len(currentGameState.getCapsules())

    score = currentGameState.getScore() #get the current game score
    numRemainingFoods = len(foodPositions) #number of remaining food pellets
    totalScaredTime = sum(scaredTimers) #total scared time of all ghosts
    totalGhostDistance = sum(ghostDistances) #total Manhattan distance to all ghost

    #reciprocal of the sum of food distances
    reciprocalFoodDistance = 0
    if sum(foodDistances) > 0:
        reciprocalFoodDistance = 1.0 / sum(foodDistances)

    #common score calculations: add reciprocal of food distances and subtract number of remaining foods
    score += reciprocalFoodDistance - numRemainingFoods

    if totalScaredTime > 0:
        #case when ghosts are scared
        score += totalScaredTime - numPowerPellets - totalGhostDistance
    else:
        #case when ghosts are not scared
        score += totalGhostDistance + numPowerPellets

    return score

# Abbreviation
better = betterEvaluationFunction
