import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import numpy as np
import cv2
import sys
import os.path as osp
dirname = osp.dirname(__file__)
sys.path.append(dirname)
from game.wrapped_flappy_bird import GameState
import random

GAME = 'flappy bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 250.  # 训练前观察的步长
EXPLORE = 3.0e6  # 随机探索的时间
FINAL_EPSILON = 1.0e-4  # 最终探索率
INITIAL_EPSILON = 0.1  # 初始探索率
REPLAY_MEMORY = 50000  # 经验池的大小
BATCH = 32  # mini-batch 的大小
FRAME_PER_ACTION = 1  # 跳帧

class Experience_Buffer:
    def __init__(self, buffer_size=REPLAY_MEMORY):
        super().__init__()
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add_experience(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:len(self.buffer) + len(experience) - self.buffer_size] = []
        self.buffer.extend(experience)
    
    def sample(self, samples_num):
        samples_data = random.sample(self.buffer, samples_num)
        train_s = [d[0] for d in samples_data]
        train_s = np.asarray(train_s)
        train_a = [d[1] for d in samples_data]
        train_a = np.asarray(train_a)
        train_r = [d[2] for d in samples_data]
        train_r = np.asarray(train_r)
        train_s_ = [d[3] for d in samples_data]
        train_s_ = np.asarray(train_s_)
        train_terminal = [d[4] for d in samples_data]
        train_terminal = np.asarray(train_terminal)
        return train_s, train_a, train_r, train_s_, train_terminal

class Deep_Q_N:
    def __init__(self, lr=1.0e-6, model_file=None):
        self.gamma = GAMMA
        self.tau = 0.01
        # tf
        self.learning_rate = lr
        self.q_model = self.build_q_net()
        self.q_target_model = self.build_q_net()
        if model_file is not None:
            self.restore_model(model_file)
    
    def save_model(self, model_path):
        self.q_model.save(model_path + '0')
        self.q_target_model.save(model_path + '1')
    
    def restore_model(self, model_path):
        self.q_model.load_weights(model_path + '0')
        self.q_target_model.load_weights(model_path + '1')
    
    def build_q_net(self):
        model = keras.Sequential()
        h_conv1 = kl.Conv2D(
            input_shape=(80, 80, 4),
            filters=32, kernel_size=8,
            data_format='channels_last',
            strides=4, padding='same',
            activation='relu'
        )
        h_pool1 = kl.MaxPool2D(
            pool_size=2, strides=2, padding='same'
        )
        h_conv2 = kl.Conv2D(
            filters=64, kernel_size=4,
            strides=2, padding='same',
            activation='relu'
        )
        h_conv3 = kl.Conv2D(
            filters=64, kernel_size=3,
            strides=1, padding='same',
            activation='relu'
        )
        h_conv3_flat = kl.Flatten()
        h_fc1 = kl.Dense(512, activation='relu')
        qout = kl.Dense(ACTIONS)
        model.add(h_conv1)
        model.add(h_pool1)
        model.add(h_conv2)
        model.add(h_conv3)
        model.add(h_conv3_flat)
        model.add(h_fc1)
        model.add(qout)

        model.compile(
            optimizer=ko.Adam(lr=self.learning_rate),
            loss=[self._get_mse_for_action]
        )

        return model
    
    def _get_mse_for_action(self, target_and_action, current_prediction):
        targets, one_hot_action = tf.split(target_and_action, [1, 2], axis=1)
        active_q_value = tf.expand_dims(tf.reduce_sum(current_prediction * one_hot_action, axis=1), axis=-1)
        return kls.mean_squared_error(targets, active_q_value)
    
    def _update_target(self):
        q_weights = self.q_model.get_weights()
        q_target_weights = self.q_target_model.get_weights()

        q_weights = [self.tau * w for w in q_weights]
        q_target_weights = [(1. - self.tau) * w for w in q_target_weights]
        new_weights = [
            q_weights[i] + q_target_weights[i]
            for i in range(len(q_weights))
        ]
        self.q_target_model.set_weights(new_weights)
    
    def _one_hot_action(self, actions):
        action_index = np.array(actions)
        batch_size = len(actions)
        result = np.zeros((batch_size, 2))
        result[np.arange(batch_size), action_index] = 1.
        return result
    
    def epsilon_greedy(self, s_t, epsilon):
        s_t = s_t.reshape(-1, 80, 80, 4)
        amax = np.argmax(self.q_model.predict(s_t)[0])
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            a_t = amax
        else:
            a_t = random.randint(0, 1)
        return a_t
    
    def train_Network(self, experience_buffer):
        # 打开游戏状态与模拟器进行通信
        game_state = GameState(fps=100)
        # 获得第1个状态并将图形进行预处理
        do_nothing = 0
        # 与游戏交互1次
        x_t, r_0, terminal = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        # 开始训练
        epsilon = INITIAL_EPSILON
        t = 0
        while "flappy bird" != "angry bird":
            a_t = self.epsilon_greedy(s_t, epsilon)
            # epsilon 递减
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            # 运用动作与环境交互1次
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
            # 将数据存储到经验池中
            experience = np.reshape(np.array([s_t, a_t, r_t, s_t1, terminal]), [1, 5])
            experience_buffer.add_experience(experience)
            # 在观察结束后进行训练
            if t > OBSERVE:
                # 采集样本
                train_s, train_a, train_r, train_s_, train_terminal = experience_buffer.sample(BATCH)
                target_q = []
                read_target_Q = self.q_target_model.predict(train_s_)
                for i in range(len(train_r)):
                    if train_terminal[i]:
                        target_q.append(train_r[i])
                    else:
                        target_q.append(train_r[i] * GAMMA * np.max(read_target_Q[i]))
                # 训练 1 步
                one_hot_actions = self._one_hot_action(train_a)
                target_q = np.asarray(target_q)
                target_and_actions = np.concatenate((target_q[:, None], one_hot_actions), axis=1)
                loss = self.q_model.train_on_batch(train_s, target_and_actions)
                # 更新旧的网络
                self._update_target()
            # 往前推进1步
            s_t = s_t1
            t += 1
            # 每 10000 次迭代保存1次
            if t % 10000 == 0:
                dirname = osp.dirname(__file__)
                self.save_model(dirname + '\\saved_networks')
            if t <= OBSERVE:
                print("OBSERVER", t)
            else:
                if t % 1 == 0:
                    print("train, steps", t, "/epsion ", epsilon, "action_index", a_t, "/reward", r_t)


if __name__=="__main__":
    buffer = Experience_Buffer()
    brain = Deep_Q_N()
    brain.train_Network(buffer)

