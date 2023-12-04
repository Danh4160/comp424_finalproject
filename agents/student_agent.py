# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import random


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
        
        # best_r, best_c, best_dir = self.find_best_move(chess_board, my_pos, adv_pos, max_step, 3)
        best_r, best_c, best_dir = self.find_best_move_IDS(chess_board, my_pos, adv_pos, max_step, 10)

        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return (best_r, best_c), best_dir
    

    def reoder_moves(self, possible_moves, state, my_pos, adv_pos, max_steps):
        scores = []

        # evaluate each move
        for move in possible_moves:
            r, c, dir = move
            state[move] = True
            score = self.evaluate(state, (r, c), adv_pos, max_steps)
            state[move] = False
            scores.append(score)

        # Reorder the moves
        possible_moves = np.array(possible_moves)
        sorted_moves_idx = np.argsort(scores)[::-1]
        sorted_moves_idx = sorted_moves_idx[:5]
        return possible_moves[sorted_moves_idx]
        

    def find_best_move_IDS(self, state, my_pos, adv_pos, max_steps, max_depth):
        start_time = time.time()
        # Initial score
        best_score = -np.inf
        best_moves = []

        moves = self.generate_moves(my_pos, adv_pos, max_steps, state)
        m = self.reoder_moves(moves, state, my_pos, adv_pos, max_steps) 
        for move in m:
            move = tuple(move)
           
            time_taken = time.time() - start_time
            if (time_taken >= 2):
                break
            r, c, dir = move
            # Perform move on the state
            state[move] = True
            minimax_score = self.iterative_deepening_search(state, (r,c), adv_pos, max_steps, max_depth, start_time)
            # Undo the move on the state
            state[move] = False

            # Update best score and add to list of best move
            if minimax_score > best_score:
                best_score = minimax_score
                best_moves = [move]

            elif minimax_score == best_score:
                best_moves.append(move)

        
        # Pick random best moves out of the list of best moves
        random_best_move = random.randint(0, len(best_moves) - 1)
        best_r, best_c, best_dir = best_moves[random_best_move]

        return best_r, best_c, best_dir



    def iterative_deepening_search(self, state, my_pos, adv_pos, max_steps, max_depth, start_time):
        score = 0
        for depth in range(max_depth):
            time_taken = time.time() - start_time
            if time_taken >= 2:
                break
            score = self.basic_minimax(state, depth, False, my_pos, adv_pos, max_steps, -np.inf, np.inf, start_time)
            if (np.isinf(score)):
                break
        return score



    def find_best_move(self, state, my_pos, adv_pos, max_steps, depth):

        start_time = time.time()
        # Initial score
        best_score = -np.inf
        best_moves = []

        moves = self.generate_moves(my_pos, adv_pos, max_steps, state)
        m = self.reoder_moves(moves, state, my_pos, adv_pos, max_steps) 
        for move in m:
            move = tuple(move)
            time_taken = time.time() - start_time
            if (time_taken >= 2):
                break
            r, c, dir = move
            # Perform move on the state
            state[move] = True
            minimax_score = self.basic_minimax(state, depth, False, (r, c), adv_pos, max_steps, -np.inf, np.inf, start_time)
            # Undo the move on the state
            state[move] = False
            # Update best score and add to list of best move
            if minimax_score > best_score:
                best_score = minimax_score
                best_moves = [move]

            elif minimax_score == best_score:
                best_moves.append(move)

        
        # Pick random best moves out of the list of best moves
        random_best_move = random.randint(0, len(best_moves) - 1)
        best_r, best_c, best_dir = best_moves[random_best_move]

        return best_r, best_c, best_dir


    def generate_moves(self, my_pos, adv_pos, max_steps, chess_board):
        visited = {my_pos}
        my_pos = np.array(my_pos)
        state_queue = [(my_pos, 0)]
        possible_moves = []

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1)) 

        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step > max_steps:
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]: 
                    continue

                # Store each direction of the current r, c
                possible_moves.append(tuple([r, c, dir]))
                next_pos = cur_pos + move
                
                next_r, next_c = next_pos
                if np.array_equal(next_pos, adv_pos) or tuple([next_r, next_c]) in visited:
                    continue
                
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        
        return possible_moves
    

    def evaluate(self, state, my_pos, adv_pos, max_steps):
         
        # The number of surrounding barriers
        # The more close to the opponent the better the move
        adv_r, adv_c = adv_pos
        my_r, my_c = my_pos

        # barriers directly around adverse
        barriers_around_adv = sum(state[adv_r, adv_c, :])  
        if barriers_around_adv == 4:
            barriers_around_adv = np.inf

        # Number of non barriers around my agent
        barriers_around_agent = sum(state[my_r, my_c, :])
        if barriers_around_agent == 4:
            barriers_around_agent = -np.inf
        
        non_barriers_around_agent = 4 - sum(state[my_r, my_c, :]) 
        if non_barriers_around_agent == 0 and barriers_around_adv != 4:
            non_barriers_around_agent = -np.inf

        
        # Distance from the middle point
        # We want to stay in the middle as often as possible
        middle_board = len(state) / 2
        distance_middle_agent = (np.abs(middle_board - my_r) + np.abs(middle_board - my_c))

        # Distance between agent and opponent
        distance_agent_adv = (np.abs(adv_r - my_r) + np.abs(adv_c - my_c))

        # Distance between middle and adv
        distance_middle_adv = np.abs(middle_board - adv_r) + np.abs(middle_board - adv_c)

        # Limit number of moves
        # num_adv_moves = len(self.generate_moves(adv_pos, my_pos, max_steps - 3, state))
        # num_my_moves = len(self.generate_moves(my_pos, adv_pos, max_steps - 3, state))

        # diff_num_moves = num_my_moves - num_adv_moves
        
        return barriers_around_agent + 3 * barriers_around_adv + 3 * non_barriers_around_agent - (0.2*distance_middle_agent) - (0.5*distance_agent_adv) + distance_middle_adv + 100#(3*diff_num_moves) + 100 # offset


    def basic_minimax(self, state, depth, is_max_turn, max_pos, min_pos, max_steps, alpha, beta, start_time):
        # Check if game has ended

        if time.time() - start_time >= 2:
            return self.evaluate(state, max_pos, min_pos, max_steps)
        
        has_ended = self.check_endgame(state, max_pos, min_pos)
        
        if depth == 0 or has_ended:
            return self.evaluate(state, max_pos, min_pos, max_steps)

        if is_max_turn:
            best_score = -np.inf            
            # Generate all possible moves 
            possible_moves = self.generate_moves(max_pos, min_pos, max_steps, state)
            possible_moves = self.reoder_moves(possible_moves, state, max_pos, min_pos, max_steps)
            for move in possible_moves:
                move = tuple(move)
                r, c, _ = move
                state[move] = True
                value = self.basic_minimax(state, depth - 1, False, (r, c), min_pos, max_steps, alpha, beta, start_time)
                state[move] = False
                best_score = max(value, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:   
            best_score = np.inf
            # Generate all possible moves 
            possible_moves = self.generate_moves(max_pos, min_pos, max_steps, state)
            possible_moves = self.reoder_moves(possible_moves, state, min_pos, max_pos, max_steps)
            for move in possible_moves:
                move = tuple(move)
                r, c, _ = move
                state[move] = True
                value = self.basic_minimax(state, depth - 1, True, max_pos, (r,c), max_steps, alpha, beta, start_time)
                state[move] = False
                best_score = min(value, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score


    def check_endgame(self, chess_board, my_pos, adv_pos):
        
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
                
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        
        if p0_r == p1_r:
            return False
        
        return True
    
    def check_valid_input(self, x, y, dir, x_max, y_max):
        return 0 <= x < x_max and 0 <= y < y_max and dir in self.dir_map
    