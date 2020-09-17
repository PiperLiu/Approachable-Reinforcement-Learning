import numpy as np

class Board:
    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = (1, 2)
        self.states = {}
    
    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and can not be \
                    less than {}'.format(self.n_in_row))
        # 当前 player 编号
        self.current_player = self.players[start_player]
        self.available = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
    
    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1, 2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]
    
    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move
    
    def do_move(self, move):
        self.states[move] = self.current_player
        self.available.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move
    
    def get_current_player(self):
        return self.current_player
    
    def current_state(self):
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            # 第 3 个平面，整个棋盘只有一个 1 ，表达上一手下的哪个
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            # 第 4 个平面，全 1 或者全 0 ，描述现在是哪一方
            square_state[3][:, :] = 1.0
        # 为什么这里要反转？
        return square_state[:, ::-1, :]
    
    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        # 是否满足双方至少下了 5 子（粗略地）
        moved = list(set(range(width * height)) - set(self.available))
        if len(moved) < self.n_in_row + 2:
            return False, -1
        
        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if w in range(width - n + 1) \
                    and len(set(states.get(i, -1) for i in range(m, m+n))) == 1:
                return True, player
            
            if h in range(height - n + 1) \
                    and len(set(states.get(i, -1) for i in range(m, m+n*width, width))) == 1:
                return True, player
            
            if w in range(width - n + 1) and h in range(height - n + 1) \
                    and len(set(states.get(i, -1) for i in range(m, m+n*(width+1), width+1))) == 1:
                return True, player
            
            if w in range(n - 1, width) and h in range(height - n + 1) \
                    and len(set(states.get(i, -1) for i in range(m, m+n*(width-1), width-1))) == 1:
                return True, player
            
        return False, -1
    
    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.available):
            return True, -1
        return False, -1

class Game:
    def __init__(self, board: Board):
        self.board = board
    
    def start_self_play(self, player, is_shown=False, temp=1e-3):
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # 保存 self-play 数据
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 执行一步落子
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # 从每一个 state 对应的 player 的视角保存胜负信息
                winner_z = np.zeros(len(current_players))
                if winner != -1:
                    winner_z[np.array(current_players) == winner] = 1.0
                    winner_z[np.array(current_players) != winner] = - 1.0
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player: ", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winner_z)
    
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 or 1')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is ", players[winner])
                    else:
                        print("Game end. Tie")
                return winner
    
    def graphic(self, board, player1, player2):
        width = board.width
        height = board.height
        
        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('x'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')
