# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()





        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]
    

    def generate_moves(self, my_pos, adv_pos, max_steps, chess_board):
        
        # BFS
        my_pos = np.array(my_pos)
        state_queue = [(my_pos, 0)]
        visited = {tuple(my_pos)}

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_steps:
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return visited
    

    def bfs(self, my_pos, adv_pos, max_steps, chess_board):
        
        # BFS
        # start_time = time.time()
        my_pos = np.array(my_pos)
        state_queue = [(my_pos, 0)]
        visited = {tuple()}
        

        r, c = my_pos
        for dir in range(4):
            if chess_board[r, c, dir]:
                continue
            visited.add(tuple([r, c, dir]))

        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) 

        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_steps:
                break
            
            # save all four directions for barriers
            for dir in range(4):
                if tuple([r, c, dir]) in visited or chess_board[r, c, dir]:
                    continue
                visited.add(tuple([r, c, dir]))

            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                r , c = next_pos
                if np.array_equal(next_pos, adv_pos) or tuple([r, c, dir]) in visited or chess_board[r, c, dir]:
                    continue
                
                # visited.add(tuple([r, c, dir]))
                state_queue.append((next_pos, cur_step + 1))
        end_time = time.time()
        # time_taken = end_time - start_time
        return visited

    # def evaluate_board(self, chess_board, max_pos, min_pos, maximizing_player):
    #     return 0

    


    def minimax(self, chess_board, max_pos, min_pos, max_player_turn, current_depth, max_depth, max_steps, alpha, beta):
        # Base case : if at max depth or game is over
        has_ended, utility = self.check_endgame(chess_board, max_pos, min_pos)

        if current_depth == max_depth or has_ended:
            return utility
        
        # Case 1 : Max Player Turn
        if max_player_turn:
            best_val = -np.inf
            for pos in self.bfs(max_pos, min_pos, max_steps, chess_board):
                # Make move
                r, c, dir = pos
                chess_board[r, c, dir] = True
                value = self.minimax(chess_board, pos, min_pos, False, current_depth + 1, max_depth, max_steps, alpha, beta)
                best_val = max(best_val, value)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val
        
        # Case 2 : Min Player Turn
        else:
            best_val = np.inf
            for pos in self.bfs(min_pos, max_pos, max_steps, chess_board):
                value = self.minimax(chess_board, pos, min_pos, True, current_depth + 1, max_depth, max_steps, alpha, beta)
                best_val = min(best_val, value)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            return best_val



    def check_endgame(self, chess_board, max_pos, min_pos):
        
        # Player 0 is max
        # Player 1 is min
        # Union-Find
        board_size = len(chess_board)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
                
        p0_r = find(tuple(max_pos))
        p1_r = find(tuple(min_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)

        if p0_r == p1_r:
            return False, 0
        
        if p0_score > p1_score:
            return True, 1 # Max player won
        elif p0_score < p1_score:
            return True, -1 # Min player won
        else:
            return True, 0 # Draw
        