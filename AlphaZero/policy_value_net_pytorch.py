import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_learning_rate(optimizer, lr):
    """ Set the learning rate to the given value """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    """ 定义策略价值网络结构 """
    def __init__(self, board_width, board_height):
        super().__init__()
        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)
    
    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val

class PolicyValueNet:
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4
        self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=self.l2_const)
        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)
    
    def policy_value_fn(self, board):
        legal_positions = board.available
        current_state = np.ascontiguousarray(
            board.current_state().reshape(
                -1, 4, self.board_width, self.board_height
            )
        )
        log_act_probs, value = self.policy_value_net(
            Variable(torch.from_numpy(current_state)).float()
        )
        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """ perform a training step """
        state_batch = Variable(torch.FloatTensor(state_batch))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs))
        winner_batch = Variable(torch.FloatTensor(winner_batch))
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)
        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = - torch.mean(
            torch.sum(mcts_probs * log_act_probs, 1)
        )
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # policy entropy, for monitoring only
        entropy = - torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
        )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params
    
    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)

