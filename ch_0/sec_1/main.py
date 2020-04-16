from kb_game import KB_Game

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
k_gamble = KB_Game()
total = 2000

k_gamble.train(play_total=total, policy='e_greedy', epsilon=0.05)
k_gamble.plot(colors='r', policy='e_greedy', style='-.')
k_gamble.reset()

k_gamble.train(play_total=total, policy='boltzmann', temperature=1)
k_gamble.plot(colors='b', policy='boltzmann', style='--')
k_gamble.reset()

k_gamble.train(play_total=total, policy='ucb', c_ratio=0.5)
k_gamble.plot(colors='g', policy='ucb', style='-')
k_gamble.reset()

plt.show()