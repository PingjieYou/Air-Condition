# -*- coding: utf-8 -*-


import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 15  # grid height
MAZE_W = 15  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r'] # 上下左右
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 画网格
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect) # 获取当前位置
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # 红色方块移动

        s_ = self.canvas.coords(self.rect)  # 现在的位置

        # 奖励机制
        if s_ == self.canvas.coords(self.oval):
            # 当前位置处于终点
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            # 当前位置处于阻塞区域
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            # 可行区域
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update() # 走下一步


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break
# 设置随机数生成器的种子，以确保实验的可重复性
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

# 初始化迷宫环境，获取动作空间大小和状态空间大小
BATCH_SIZE = 128  # 批处理大小，即从经验回放中选择的转换数量
GAMMA = 0.9  # 折扣因子，用于计算未来奖励的折扣总和
EPS_START = 0.9  # ε的起始值
EPS_END = 0.05  # ε的终止值
EPS_DECAY = 1000  # ε的指数衰减速率，数值越大，衰减越慢
TAU = 0.005  # 目标网络的更新速率
LR = 1e-4  # AdamW优化器的学习速率

env = Maze()

# 从迷宫环境中获取动作数量和状态观测数量
n_actions = env.n_actions
state = env.reset()
n_observations = len(state)
steps_done = 0
episode_durations = []

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义Transition（转换）命名元组以存储经验回放所需的信息
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# 创建一个ReplayMemory类，用于存储和采样转换
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 定义DQN类，这是一个神经网络模型，用于估计在给定状态下采取不同动作的价值
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# 创建策略网络和目标网络，实例化优化器和经验回放缓冲区。
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000) # 设置经验回放缓冲区的最大容量


class DeepQNetwork:
    def __init__(self, n_actions: int, n_observations: int, action_space: list):
        self.n_actions = n_actions  # 行为数量
        self.n_observations = n_observations  # 状态空间
        self.action_space = action_space  # 行为空间

    def select_action(self, state):
        # epsilon-greedy策略，以一定概率选择随机动作
        global steps_done
        sample = random.random() # 生成0～1之间的随机数
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY) # 计算当前ε的值
        steps_done += 1
        action_ = self.choose_action() # 选择行为
        # 以ε-greedy策略选择行为,即以ε的概率随机选择行为，以1-ε的概率选择当前最优行为,ε的值会随着训练的进行不断衰减,从而使智能体的行为越来越接近最优策略
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1) # 选择当前最优行为
        else:
            return torch.tensor([[action_]], device=device, dtype=torch.long) # 选择随机行为

    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    # Training loop
    def optimize_model(self):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # 将批处理转换为每个转换的批处理，以便能够计算每个元素的损失
        batch = Transition(*zip(*transitions))
        # 计算非最终状态的掩码并连接批处理元素
        # (最终状态将是模拟结束后的状态)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 计算Q(s_t, a) - 模型计算Q(s_t)，然后我们选择执行的动作列。这些是根据策略网络对每个批处理状态的预测，按照action_batch中的动作索引选择状态动作值
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # 计算V(s_{t+1}) - 从非最终下一个状态的预测中获取最大的预测值。
        # 预期状态动作值是基于“旧”目标网络计算的;选择其最佳奖励与max(1)[0]。
        # 这是基于掩码合并的，因此我们将具有预期状态值或0的状态合并在一起，以防状态是最终状态。
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
        # 计算预期的Q值 - 通过将预期状态动作值与预期状态值相乘来获得预期的状态动作值。
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # 计算Huber损失
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 优化模型
        optimizer.zero_grad()
        loss.backward()
        # 通过将梯度裁剪到范围[-1,1]来稳定训练
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    def choose_action(self):
        return np.random.randint(0, self.n_actions)


if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 50

RL = DeepQNetwork(env.n_actions, len(env.reset()), env.action_space)


for i_episode in range(num_episodes):
    # 初始化环境和状态
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # [1, 4]
    for t in count():
        env.render() # 显示画面
        action = RL.select_action(state) # 选择行为,范围为0～3
        observation, reward, terminated = env.step(action) # 执行行为，获得下一个状态，奖励和是否终止
        reward = torch.tensor([reward], device=device)
        done = terminated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 存储状态，行为，下一状态和奖励
        memory.push(state, action, next_state, reward)

        # 状态更新
        state = next_state

        # 优化模型
        RL.optimize_model()

        # 软更新目标网络参数，即：θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            break