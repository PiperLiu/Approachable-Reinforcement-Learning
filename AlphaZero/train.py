import random
import numpy as np
from collections import deque
import os.path as osp
from .game import Board, Game
from .mcts_alphaZero import MCTSPlayer
from .policy_value_net_pytorch import PolicyValueNet

DIRNAME = osp.dirname(__file__)

class TrainPipeline:
    def __init__(self, init_model=None):
        # 棋盘相关参数
        self.board_width = 8
        self.board_height = 8
        self.n_in_row = 5
        self.board = Board(
            width=self.board_width,
            height=self.board_height,
            n_in_row=self.n_in_row
        )
        self.game = Game(self.board)
        # 自我对弈相关参数
        self.temp = 1.0
        self.c_puct = 5
        self.n_playout = 400
        # 训练更新相关参数
        self.learn_rate = 2e-3
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.check_freq = 2  # 保存模型的概率
        self.game_batch_num = 3000  # 训练更新的次数
        if init_model:
            # 如果提供了初始模型，则加载其用于初始化策略价值网络
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height,
                model_file=init_model
            )
        else:
            # 随机初始化策略价值网络
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height
            )
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1
        )
    
    def run(self):
        """ 执行完整的训练流程 """
        for i in range(self.game_batch_num):
            episode_len = self.collect_selfplay_data()
            if len(self.data_buffer) > self.batch_size:
                loss, entropy = self.policy_update()
                print((
                    "batch i:{}, "
                    "episode_len:{}, "
                    "loss:{:.4f}, "
                    "entropy:{:.4f}"
                ).format(i+1, episode_len, loss, entropy))
                # save performance per update
                loss_array = np.load(DIRNAME + '/loss.npy')
                entropy_array = np.load(DIRNAME + '/entropy.npy')
                loss_array = np.append(loss_array, loss)
                entropy_array = np.append(entropy_array, entropy)
                np.save(DIRNAME + '/loss.npy', loss_array)
                np.save(DIRNAME + '/entropy.npy', entropy_array)
                del loss_array
                del entropy_array
            else:
                print("batch i:{}, episode_len:{}".format(i+1, episode_len))
            # 定期保存模型
            if (i+1) % self.check_freq == 0:
                self.policy_value_net.save_model(
                    DIRNAME + '/current_policy.model'
                )
    
    def collect_selfplay_data(self):
        """ collect self-play data for training """
        winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
        play_data = list(play_data)[:]
        episode_len = len(play_data)
        # augment the data
        play_data = self.get_equi_data(play_data)
        self.data_buffer.extend(play_data)
        return episode_len
    
    def get_equi_data(self, play_data):
        """ play_data: [(state, mcts_prob, winner_z), ...] """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)
                ), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data
    
    def policy_update(self):
        """ update the policy-value net """
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        loss, entropy = self.policy_value_net.train_step(
            state_batch, mcts_probs_batch, winner_batch, self.learn_rate
        )
        return loss, entropy
