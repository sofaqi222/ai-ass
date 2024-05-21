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
    """
    Your minimax agent (question 2)
    """

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
        "*** YOUR CODE HERE ***"

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
            actions = gameState.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(gameState)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                score = minimax(1, depth, successor)
                maxScore = max(maxScore, score)
            return maxScore

        def minValue(agentIndex, depth, gameState):
            minScore = float('inf')
            nextAgent = agentIndex + 1
            numAgents = gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(gameState)
            # If this is the last ghost, the next agent will be Pacman and depth will increase
            if agentIndex == numAgents - 1:
                nextAgent = 0
                depth += 1
            for action in actions:
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
        # def minimax(agentIndex, depth, gameState):
        #     # Terminal state or depth limit reached
        #     if gameState.isWin() or gameState.isLose() or depth == self.depth:
        #         return self.evaluationFunction(gameState)
        #
        #     # Pacman (Maximizing agent)
        #     if agentIndex == 0:
        #         return maxValue(agentIndex, depth, gameState)
        #     # Ghosts (Minimizing agents)
        #     else:
        #         return minValue(agentIndex, depth, gameState)
        #
        # def maxValue(agentIndex, depth, gameState):
        #     maxScore = float('-inf')
        #     for action in gameState.getLegalActions(agentIndex):
        #         successor = gameState.generateSuccessor(agentIndex, action)
        #         score = minimax(1, depth, successor)
        #         maxScore = max(maxScore, score)
        #     return maxScore
        #
        # def minValue(agentIndex, depth, gameState):
        #     minScore = float('inf')
        #     nextAgent = agentIndex + 1
        #     numAgents = gameState.getNumAgents()
        #     # If this is the last ghost, the next agent will be Pacman and depth will increase
        #     if agentIndex == numAgents - 1:
        #         nextAgent = 0
        #         depth += 1
        #     for action in gameState.getLegalActions(agentIndex):
        #         successor = gameState.generateSuccessor(agentIndex, action)
        #         score = minimax(nextAgent, depth, successor)
        #         minScore = min(minScore, score)
        #     return minScore
        #
        # bestAction = None
        # bestScore = float('-inf')
        # for action in gameState.getLegalActions(0):
        #     successor = gameState.generateSuccessor(0, action)
        #     score = minimax(1, 0, successor)
        #     if score > bestScore:
        #         bestScore = score
        #         bestAction = action
        #
        # return bestAction

        # numberOfGhosts = gameState.getNumAgents() - 1
        #
        # # Used only for pacman agent hence agentindex is always 0.
        # def maxLevel(gameState, depth):
        #     currDepth = depth + 1
        #     if gameState.isWin() or gameState.isLose() or currDepth == self.depth:  # Terminal Test
        #         return self.evaluationFunction(gameState)
        #     maxvalue = -999999
        #     actions = gameState.getLegalActions(0)
        #     for action in actions:
        #         successor = gameState.generateSuccessor(0, action)
        #         maxvalue = max(maxvalue, minLevel(successor, currDepth, 1))
        #     return maxvalue
        #
        # # For all ghosts.
        # def minLevel(gameState, depth, agentIndex):
        #     minvalue = 999999
        #     if gameState.isWin() or gameState.isLose():  # Terminal Test
        #         return self.evaluationFunction(gameState)
        #     actions = gameState.getLegalActions(agentIndex)
        #     for action in actions:
        #         successor = gameState.generateSuccessor(agentIndex, action)
        #         if agentIndex == (gameState.getNumAgents() - 1):
        #             minvalue = min(minvalue, maxLevel(successor, depth))
        #         else:
        #             minvalue = min(minvalue, minLevel(successor, depth, agentIndex + 1))
        #     return minvalue
        #
        # # Root level action.
        # actions = gameState.getLegalActions(0)
        # currentScore = -999999
        # returnAction = ''
        # for action in actions:
        #     nextState = gameState.generateSuccessor(0, action)
        #     # Next level is a min level. Hence calling min for successors of the root.
        #     score = minLevel(nextState, 0, 1)
        #     # Choosing the action which is Maximum of the successors.
        #     if score > currentScore:
        #         returnAction = action
        #         currentScore = score
        # return returnAction
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
