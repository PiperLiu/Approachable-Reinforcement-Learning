import copy
import numpy as np

def Softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode:
    """ A node in the MCTS tree. """
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
    
    def select(self, c_puct):
        """ Return: A tuple of (action, next_node) """
        return max(
            self._children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct)
        )
    
    def get_value(self, c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
    
    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
    
    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits
    
    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    
    def is_leaf(self):
        return self._children == {}
    
    def is_root(self):
        return self._parent is None


class MCTS(object):
    """ An implementation of Monte Carlo Tree Search. """
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
    
    def _playout(self, state):
        """ 完整的执行选择、扩展评估和回传更新等步骤 """
        node = self._root
        # 选择
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)
        # 扩展及评估
        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:  # 平局
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )
        # 回传更新
        node.update_recursive(-leaf_value)
    
    def get_move_probs(self, state, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        
        act_visits = [
            (act, node._n_visits) for act, node in self._root._children.items()
        ]
        acts, visits = zip(*act_visits)
        # 注意，这里是根据 visits 算出的动作选择概率
        act_probs = Softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs
    
    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

class MCTSPlayer:
    """ AI player based on MCTS """
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
    
    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.available
        move_probs = np.zeros(board.width * board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(
                    acts, p = 0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # 更新根节点，复用搜索子树
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                # 重置根节点
                self.mcts.update_with_move(-1)
                location = board.move_to_location(move)
                print("AI move: %d, %d\n".format(location[0], location[1]))
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")
    
    def set_player_ind(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_with_move(-1)
    
    def __str__(self):
        return "MCTS {}".format(self.player)
