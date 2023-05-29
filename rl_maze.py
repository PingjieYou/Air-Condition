# -*- encoding: utf-8 -*-
'''
@File    :   rl_maze.py
@Time    :   2023/05/22 14:36:24
@Author  :   YouPingJie
@Function:   
'''

import math
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from itertools import count
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pyamaze import maze, agent
from collections import namedtuple, deque

Infeasible_Steps = 0
Turns = 0
Path_Length = 0
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Maze():
    """迷宫环境"""
    def __init__(self,maze_map) -> None:
        super(Maze, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.solution = []
        self.maze_map = maze_map
    
    def reset(self):
        """
        重置迷宫，返回起点

        :return: 起点坐标
        """
        self.solution = [(1, 1)]
        return self.solution[-1]
    
    def step(self,action,rows,cols):
        """
        根据动作返回下一个状态

        :param action: _description_
        """
        current_state = self.solution[-1]        
        
        if action == 0: # up
            if current_state[0] > 1:
                if self.maze_map[current_state]['N'] == 0:
                    return 'terminal',-1,True
                self.solution.append((current_state[0]-1,current_state[1]))
        elif action == 1: # down
            if current_state[0] < rows:
                if self.maze_map[current_state]['S'] == 0:
                    return 'terminal',-1,True
                self.solution.append((current_state[0]+1,current_state[1]))
        elif action == 2: # left
            if current_state[1] > 1:
                if self.maze_map[current_state]['W'] == 0:
                    return 'terminal',-1,True
                self.solution.append((current_state[0],current_state[1]-1))
        elif action == 3: # right
            if current_state[1] < cols:
                if self.maze_map[current_state]['E'] == 0:
                    return 'terminal',-1,True
                self.solution.append((current_state[0],current_state[1]+1))

        new_state = self.solution[-1]

        return ('terminal',1,True) if new_state == (rows,cols) else (new_state,0,False)

class ReplayMemory(object):
    """经验回放，存储agent的经验"""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """存放经验"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """神经网络"""
    def __init__(self, n_observations,n_actions,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DeepQNetwork():
    def __init__(self,n_actions,n_observation,action_space,eval_net,target_net) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.n_observation = n_observation
        self.action_space = action_space
        self.eval_net = eval_net
        self.target_net = target_net

    def choose_action(self):
        """随机生成概率，用于选择动作"""
        return np.random.randint(0,self.n_actions)

    def select_action(self,state,eps_start,eps_end,eps_decay,steps_done):
        """
        根据状态选择动作

        :param state: 状态，即agent的位置
        """
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        action = self.choose_action()

        return self.eval_net(state).max(1)[1].view(1,1).detach() if sample > eps_threshold else torch.tensor([[action]],dtype=torch.long).cuda()
    
    def optimize(self,memory,batch_size,gamma,optimizer,device):
        """优化模型"""
        if len(memory) < batch_size:
            return

        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)),device=device,dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.eval_net(state_batch).gather(1,action_batch)
        next_state_values = torch.zeros(batch_size,device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.eval_net.parameters(),100)
        optimizer.step()

        return loss

def plot_steps(step_history):
    plt.plot(step_history)
    plt.title('Steps Cost')
    plt.xlabel('Epoch')
    plt.ylabel('Steps')
    plt.show()

def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Maze")

    parser.add_argument('--rows', type=int, default=4, help="maze rows")
    parser.add_argument('--cols', type=int, default=4, help="maze cols")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-6, help="learning rate")
    parser.add_argument('--gamma', type=float, default=0.9, help="reward discount")
    parser.add_argument('--eps_start', type=float, default=0.9, help="eps start")
    parser.add_argument('--eps_end', type=float, default=0.05, help="eps end")
    parser.add_argument('--eps_decay', type=float, default=200, help="eps decay")
    parser.add_argument('--tau',type=float,default=0.005,help="soft update")
    parser.add_argument('--epochs',type=int,default=1000,help="epochs")

    opts = parser.parse_args()

    random.seed(1024)
    np.random.seed(1024)
    torch.manual_seed(1024)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maze_ = maze(opts.rows,opts.cols)
    maze_.CreateMaze(loopPercent=100)
    maze_map = maze_.maze_map
    maze_agent = agent(maze_,shape="square",filled=False,footprints=True)

    env = Maze(maze_map)

    state = env.reset()
    n_actions = env.n_actions
    n_observation = len(state)
    episode_durations = []
    steps_done = 0

    eval_net = DQN(n_observation,n_actions).to(device)
    target_net = DQN(n_observation,n_actions).to(device)
    target_net.load_state_dict(eval_net.state_dict())

    optimizer = optim.AdamW(eval_net.parameters(),lr=opts.lr,amsgrad=True)
    memory = ReplayMemory(10000)

    q_net = DeepQNetwork(n_actions,n_observation,env.action_space,eval_net,target_net)
    solution_list = []

    loss_history = []
    step_history = []

    for epoch in range(opts.epochs):
        state = env.reset()
        state = torch.tensor(state,dtype=torch.float32).unsqueeze(0).to(device) # [1,4]

        for t in count():
            action = q_net.select_action(state,opts.eps_start,opts.eps_end,opts.eps_decay,steps_done)
            observation, reward, terminated = env.step(action,opts.rows,opts.cols)
            reward = torch.tensor([reward],device=device)
            done = terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation,dtype=torch.float32).unsqueeze(0).to(device)

            memory.push(state,action,next_state,reward)
            state = next_state

            loss = q_net.optimize(memory,opts.batch_size,opts.gamma,optimizer,device)
            if loss is not None:
                loss_history.append(loss.cpu().detach().numpy())

            target_net_state_dict = target_net.state_dict()
            eval_net_state_dict = eval_net.state_dict()
            for key in eval_net_state_dict:
                target_net_state_dict[key] = opts.tau * eval_net_state_dict[key] + (1 - opts.tau) * target_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

            if done:
                print(epoch)
                if reward == 1:
                    step_history.append(len(env.solution)-1)
                break

    plot_loss(loss_history)
    plot_steps(step_history)

    for epoch in range(opts.epochs):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # [1,4]

        for t in count():
            action = q_net.select_action(state, opts.eps_start, opts.eps_end, opts.eps_decay, steps_done)
            observation, reward, terminated = env.step(action, opts.rows, opts.cols)
            reward = torch.tensor([reward], device=device)
            done = terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

            memory.push(state, action, next_state, reward)
            state = next_state

            q_net.optimize(memory, opts.batch_size, opts.gamma, optimizer, device)

            target_net_state_dict = target_net.state_dict()
            eval_net_state_dict = eval_net.state_dict()
            for key in eval_net_state_dict:
                target_net_state_dict[key] = opts.tau * eval_net_state_dict[key] + (1 - opts.tau) * target_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

            if done:
                print(epoch)
                episode_durations.append(t + 1)
                if reward == 1:
                    solution = env.solution
                    solPath = env.solution
                    solPath.reverse()
                    a = agent(maze_, shape="square", filled=False, footprints=True)
                    maze_.tracePath({a: solPath}, delay=100)
                    maze_.run()
                break


if __name__ == '__main__':
    main()
