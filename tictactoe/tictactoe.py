"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Start of game
    if board == initial_state():
        return X
    
    # Determine player by counting number of moves
    move_count = 0
    for row in board:
        for field in row:
            if field is not EMPTY:
                move_count += 1
    # Even move count -> X's turn
    if move_count % 2 == 0:
        return X
    # Odd move count -> O's turn
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    action_set = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] is None:
                action_set.add((i,j))
    return action_set


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] is not EMPTY:
        raise Exception("Not a valid action, the cell must be empty!")
    result_board = initial_state()
    for i in range(3):
        for j in range(3):
            result_board[i][j] = board[i][j]
    result_board[action[0]][action[1]] = player(board)
    return result_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows
    for row in board:
        if row.count(X) == 3: return X
        if row.count(O) == 3: return O
    
    # Check cols
    for col in range(3):
        col_vector = []
        for row in board:
            col_vector.append(row[col])
        if col_vector.count(X) == 3: return X
        if col_vector.count(O) == 3: return O
    
    # Check diags
    if board[0][0] == board[1][1] == board[2][2] == X: return X
    if board[0][0] == board[1][1] == board[2][2] == O: return O

    if board[0][2] == board[1][1] == board[2][0] == X: return X
    if board[0][2] == board[1][1] == board[2][0] == O: return O

    # No winner
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    
    # Check if the game is still ongoing
    # If no field is EMPTY it's a draw and terminated.
    terminated = True
    for row in board:
        for field in row:
            if field == EMPTY:
                terminated = False  
    return terminated


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    if winner(board) == O:
        return -1
    return 0


def max_value(board, alpha, beta):
    """
    Determines the highest valued action recursively.
    """
    value = -math.inf

    if terminal(board):
        return utility(board)
    
    for possible_action in actions(board):
        value = max(value, min_value(result(board, possible_action), alpha, beta))
        alpha = max(value, alpha)
        if beta <= alpha:
            break
    return value


def min_value(board, alpha, beta):
    """
    Determines the lowest valued action recursively.
    """
    value = math.inf

    if terminal(board):
        return utility(board)

    for possible_action in actions(board):
        value = min(value, max_value(result(board, possible_action), alpha, beta))
        beta = min(value, beta)
        if beta <= alpha:
            break
    return value


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    action = None
    # alpha: max. value that X can guarantee
    # beta: min. value that O can guarantee
    start_alpha = -math.inf
    start_beta = math.inf

    if terminal(board):
        return action

    if player(board) == X:
        value = -math.inf
        for possible_action in actions(board):
            action_value = min_value(result(board, possible_action), start_alpha, start_beta)
            if action_value > value:
                value = action_value
                action = possible_action
        return action
    
    if player(board) == O:
        value = math.inf
        for possible_action in actions(board):
            action_value = max_value(result(board, possible_action), start_alpha, start_beta)
            if action_value < value:
                value = action_value
                action = possible_action
        return action