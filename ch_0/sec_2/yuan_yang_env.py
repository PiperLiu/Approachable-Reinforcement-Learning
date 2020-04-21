import pygame
import random
from load_images import *
import numpy as np

class YuanYangEnv:
    def __init__(self):
        self.states = []
        for i in range(0, 100):
            self.states.append(i)
        self.actions = ['e', 's', 'w', 'n']
        self.gamma = 0.8
        self.value = np.zeros((10, 10))

        self.viewer = None
        self.FPSCLOCK = pygame.time.Clock()
        
        self.screen_size = (1200, 900)
        self.bird_position = (0, 0)
        self.limit_distance_x = 120
        self.limit_distance_y = 90
        self.obstacle_size = [120, 90]
        self.obstacle1_x = []
        self.obstacle1_y = []
        self.obstacle2_x = []
        self.obstacle2_y = []
        
        for i in range(8):
            # obstacle 1
            self.obstacle1_x.append(360)
            if i <= 3:
                self.obstacle1_y.append(90 * i)
            else:
                self.obstacle1_y.append(90 * (i + 2))
            # obstacle 2
            self.obstacle2_x.append(720)
            if i <= 4:
                self.obstacle2_y.append(90 * i)
            else:
                self.obstacle2_y.append(90 * (i + 2))
        
        self.bird_male_init_position = [0.0, 0.0]
        self.bird_male_position = [0, 0]
        self.bird_female_init_position = [1080, 0]

    def collide(self, state_position):
        flag = 1
        flag1 = 1
        flag2 = 1

        # obstacle 1
        dx = []
        dy = []
        for i in range(8):
            dx1 = abs(self.obstacle1_x[i] - state_position[0])
            dx.append(dx1)
            dy1 = abs(self.obstacle1_y[i] - state_position[1])
            dy.append(dy1)
        mindx = min(dx)
        mindy = min(dy)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag1 = 0

        # obstacle 2
        dx_second = []
        dy_second = []
        for i in range(8):
            dx1 = abs(self.obstacle2_x[i] - state_position[0])
            dx_second.append(dx1)
            dy1 = abs(self.obstacle2_y[i] - state_position[1])
            dy_second.append(dy1)
        mindx = min(dx_second)
        mindy = min(dy_second)
        if mindx >= self.limit_distance_x or mindy >= self.limit_distance_y:
            flag2 = 0
        
        if flag1 == 0 and flag2 == 0:
            flag = 0
        
        # collide edge
        if state_position[0] > 1080 or \
                state_position[0] < 0 or \
                state_position[1] > 810 or \
                state_position[1] < 0:
            flag = 1
        
        return flag
    
    def find(self, state_position):
        flag = 0
        if abs(state_position[0] - self.bird_female_init_position[0]) < \
                self.limit_distance_x and \
                abs(state_position[1] - self.bird_female_init_position[1]) < \
                self.limit_distance_y:
            flag = 1
        return flag
    
    def state_to_position(self, state):
        i = int(state / 10)
        j = state % 10
        position = [0, 0]
        postion[0] = 120 * j
        postion[1] = 90 * i
        return position
    
    def position_to_state(self, position):
        i = position[0] / 120
        j = position[1] / 90
        return int(i + 10 * j)
    
    def reset(self):
        # 随机产生一个初始位置
        flag1 = 1
        flag2 = 1
        while flag1 or flag2 == 1:
            state = self.states[int(random.random() * len(self.states))]
            state_position = self.state_to_position(state)
            flag1 = self.collide(state_position)
            flag2 = self.find(state_position)
        return state

    def transform(self, state, action):
        current_position = self.state_to_position(state)
        next_position = [0, 0]
        flag_collide = 0
        flag_find = 0

        flag_collide = self.collide(current_position)
        flag_find = self.find(current_position)
        if flag_collide == 1:
            return state, -1, True
        if flag_find == 1:
            return state, 1, True

        if action == 'e':
            next_position[0] = current_position[0] + 120
            next_position[1] = current_position[1]
        if action == 's':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] + 90
        if action == 'w':
            next_position[0] = current_position[0] - 120
            next_position[1] = current_position[1]
        if action == 'n':
            next_position[0] = current_position[0]
            next_position[1] = current_position[1] - 90

        flag_collide = self.collide(next_position)
        if flag_collide == 1:
            return self.position_to_state(current_position), -1, True
        
        flag_find = self.find(next_position)
        if flag_find == 1:
            return self.position_to_state(next_position), 1, True
        
        return self.position_to_state(next_position), 0, False
    
    def gameover(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
    
    def render(self):
        if self.viewer is None:
            pygame.init()
        
        self.viewer = pygame.display.set_mode(self.screen_size, 0, 32)
        pygame.display.set_caption("yuanyang")
        # load pic
        self.bird_male = load_bird_male()
        self.bird_female = load_bird_female()
        self.background = load_background()
        self.obstacle = load_obstacle()

        # self.viewer.blit(self.bird_female, self.bird_female_init_position)
        # self.viewer.blit(self.bird_male, self.bird_male_init_position)

        self.viewer.blit(self.background, (0, 0))
        self.font = pygame.font.SysFont('times', 15)

        self.viewer.blit(self.background, (0, 0))
        for i in range(11):
            pygame.draw.lines(self.viewer,
                    (255, 255, 255),
                    True,
                    ((120 * i, 0), (120 * i, 900)),
                    1
                )
            pygame.draw.lines(self.viewer,
                    (255, 255, 255),
                    True,
                    ((0, 90 * i), (1200, 90 * i)),
                    1
                )

        for i in range(8):
            self.viewer.blit(self.obstacle, (self.obstacle1_x[i], self.obstacle1_y[i]))
            self.viewer.blit(self.obstacle, (self.obstacle2_x[i], self.obstacle2_y[i]))
        
        self.viewer.blit(self.bird_female, self.bird_female_init_position)
        self.viewer.blit(self.bird_male, self.bird_male_init_position)

        for i in range(10):
            for j in range(10):
                surface = self.font.render(str(
                    round(float(self.value[i, j]), 3)), True, (0, 0, 0)
                )
                self.viewer.blit(surface, (120 * i + 5, 90 * j + 70))
        
        pygame.display.update()
        self.gameover()
        self.FPSCLOCK.tick(30)


if __name__ == "__main__":
    yy = YuanYangEnv()
    yy.render()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

