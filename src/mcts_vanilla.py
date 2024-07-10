
from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 1000
explore_faction = 2.
MAX_DEPTH = 100

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """Traverses the tree until the end criterion are met.
    e.g., find the best expandable node (node with untried action) if it exists,
    or else a terminal node.

    Args:
        node: A tree node from which the search is traversing.
        board: The game setup.
        state: The state of the game.
        bot_identity: The bot's identity, either 1 or 2.

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node.
    """
    while node.child_nodes and not node.untried_actions:
        if node.parent is None:
            is_opponent = False
        else:
            # Determine if the current node represents an opponent's move
            is_opponent = board.current_player(state) != bot_identity

        ucb_values = {action: ucb(child, is_opponent) for action, child in node.child_nodes.items()}
        best_action = max(ucb_values, key=ucb_values.get)
        node = node.child_nodes[best_action]
        state = board.next_state(state, best_action)
        
    return node, state


def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    if node.untried_actions:
        # Select an untried action
        action = node.untried_actions.pop()

        # Get the new state by applying the action
        new_state = board.next_state(state, action)

        # Create a new child node
        new_node = MCTSNode(parent=node, parent_action=action, action_list=board.legal_actions(new_state))

        # Add the new node to the child nodes of the current node
        node.child_nodes[action] = new_node

    return node, state


def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    depth = 0
    while not board.is_ended(state) and depth < MAX_DEPTH:
        legal_actions = board.legal_actions(state)
        if not legal_actions:
            # If no legal actions are available, break the loop
            break
        # Choose a random legal action
        move = choice(legal_actions)
        # Update the state by applying the action
        state = board.next_state(state, move)
        depth += 1

    return state


def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    while node is not None:
        node.visits += 1

        if won:
            node.wins += 1

        node = node.parent
    
def ucb(node: MCTSNode, is_opponent: bool):
    """ Calcualtes the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    if node.visits == 0:
        return float('inf')  # Ensure unvisited nodes are prioritized
    
    # Calculate the win rate
    win_rate = node.wins / node.visits
    if is_opponent:
        win_rate = 1 - win_rate  # Invert the win rate if it's the opponent's move

    # Calculate the exploration term
    exploration_term = explore_faction * sqrt(log(node.parent.visits) / node.visits)
    
    # Calculate and return the UCB value
    return win_rate + exploration_term

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    best_action = None
    max_visits = -1

    # Iterate over all child nodes (possible actions)
    for action, child in root_node.child_nodes.items():
        # Check if the current child node has more visits than the current max_visits
        if child.visits > max_visits:
            max_visits = child.visits  # Update max_visits
            best_action = action  # Update best_action

    return best_action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        state = current_state
        node = root_node

        # Do MCTS - This is all you!
        # ...
        # Selection
        node, state = traverse_nodes(node, board, state, bot_identity)

        # Expansion
        if node.untried_actions:
            node, state = expand_leaf(node, board, state)

        # Simulation
        final_state = rollout(board, state)

        # Backpropagation
        won = is_win(board, final_state, bot_identity)
        backpropagate(node, won)

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)
    
    #print(f"Action chosen: {best_action}")
    return best_action
