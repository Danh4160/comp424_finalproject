# Human Input agent
from agents.agent import Agent
from store import register_agent
import numpy as np
import sys
import time

@register_agent("human_agent")
class HumanAgent(Agent):
    def __init__(self):
        super(HumanAgent, self).__init__()
        self.name = "HumanAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        print("Available Moves:")

        test_board = np.copy(chess_board)
        generated_moves, time = self.bfs(my_pos=my_pos, adv_pos=adv_pos, max_steps=max_step, chess_board=test_board)
        print("generated moves", generated_moves, time)
        t_r, t_c, t_dir = generated_moves.pop()
        print("first move", t_r, t_c, t_dir)
        print("Board before move", test_board[t_r, t_c, t_dir])
        test_board[t_r, t_c, t_dir] = True
        print("Board after move", test_board[t_r, t_c, t_dir])

        text = input("Your move (x,y,dir) or input q to quit: ")
        while len(text.split(",")) != 3 and "q" not in text.lower():
            print("Wrong Input Format!")
            text = input("Your move (x,y,dir) or input q to quit: ")
        if "q" in text.lower():
            print("Game ended by user!")
            sys.exit(0)
        x, y, dir = text.split(",")
        x, y, dir = x.strip(), y.strip(), dir.strip()
        x, y = int(x), int(y)



        while not self.check_valid_input(
            x, y, dir, chess_board.shape[0], chess_board.shape[1]
        ):
            print(
                "Invalid Move! (x, y) should be within the board and dir should be one of u,r,d,l."
            )
            text = input("Your move (x,y,dir) or input q to quit: ")
            while len(text.split(",")) != 3 and "q" not in text.lower():
                print("Wrong Input Format!")
                text = input("Your move (x,y,dir) or input q to quit: ")
            if "q" in text.lower():
                print("Game ended by user!")
                sys.exit(0)
            x, y, dir = text.split(",")
            x, y, dir = x.strip(), y.strip(), dir.strip()
            x, y = int(x), int(y)
        my_pos = (x, y)
        return my_pos, self.dir_map[dir]

    def check_valid_input(self, x, y, dir, x_max, y_max):
        return 0 <= x < x_max and 0 <= y < y_max and dir in self.dir_map
    

    def bfs(self, my_pos, adv_pos, max_steps, chess_board):
        
        # BFS
        start_time = time.time()
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
        time_taken = end_time - start_time
        return visited, time_taken

    

