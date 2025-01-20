import random

import numpy as np
import torch
import os
from maddpg import MADDPG
# 定义了智能体的类
# *********************新增了智能体的观测量o*********************
class Agent:
    def __init__(self, agent_id, args):
        # 创建包含各种参数的对象
        self.args = args
        # 智能体的ID
        self.agent_id = agent_id
        # 每个智能体创建一个多智能体深度确定性策略梯度对象，以便后续使用该策略来选择动作和学习
        self.policy = MADDPG(args, agent_id)
        # 新观测量
        self.t_control = 0.0 # 控制时间t
        self.gas_flow = 0.0 # 干燥气体的体积流量
        self.gas_pressure = 0.0 # 气体压力
        self.gas_temperture = 0.0 # 气体的绝对温度
        self.paper_fiber = 0.0 #干燥纸业的绝干纤维物量
        # 三元组表示
        self.R_s = 0.0  # 起止干度 R_(0:t/∆t)^s
        self.R_b = 0.0  # 起止干度 R_(0:t/∆t)^b
        self.T_s = 0.0  # 起止温度 T_(0:t/∆t)^s
        self.T_b = 0.0  # 起止温度 T_(0:t/∆t)^b
        self.heat_absorbed = 0.0  # 吸取的热量
        self.task_done = False  # 任务完成状态

    # *********************增加了多个奖励函数用作计算最终奖励函数*********************
    # 用气成本物料量负值 gas_cost : r_bas
    def get_gas_cost(self):
        # 替换为具体的公式求解
        r_bas = -random.randint(0,100)
        return r_bas
    # 蒸汽物料量 steam_cost : r_vap
    def get_steam_cost(self):
        # 替换为具体的公式求解
        r_vap = -random.randint(0,100)
        return r_vap
    # 新鲜蒸汽用气成本 fresh_steam_cost: r_air
    def get_fresh_steam_cost(self):
        # 替换为具体的公式求解
        r_air = -random.randint(0,100)
        return r_air
    # 纸张无法烘干 paper_is_dry : r_thr
    def paper_is_dry(self):
        # 增设条件
        r_thr = 0
        if self.R_b >= 0.80:# 增设烘干条件
            r_thr += 1
        return r_thr

    # 根据当前观察（环境状态）o、噪声率noise_rate和随机因素epsilon从策略中选择一个动作
    # 动作的范围在-self.args.high_action到self.args.high_action之间
    # 维度是self.args.action_shape[self.agent_id]
    def select_action(self, o, noise_rate, epsilon):# 选择动作
        # ε-贪婪策略，以防智能体总是选择当前认为最优的动作，陷入局部最优解，而没有机会发现全局最优解
        # epsilon 是[0, 1]范围内。通常在训练初期赋予较大的值，使得智能体更多进行探索；随着训练的进行，ε逐渐减小，使得智能体更多地利用已经学到的策略
        if np.random.uniform() < epsilon:
            # 智能体将随机选择动作
            u = np.random.uniform(self.args.low_action, self.args.high_action, self.args.action_shape[self.agent_id])
            # print("动作u：", u)
        else:
            # 智能体将根据当前策略选择动作
            # 将观察值o转化成tensor并输入到策略的actor网络中，得到输出动作pi
            # 给定观察下输出的动作
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{}'.format(pi))
            # 动作pi转化为numpy数组，添加高斯噪声，然后将动作裁剪到指定范围内
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            # 限制智能体选择了超出范围的动作
            u = np.clip(u, self.args.low_action, self.args.high_action)
            # print("动作1u:", u)
            #u = np.clip(u, self.args.low_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        # 基于更新的数据和其他智能体的信息，更新智能体的策略
        self.policy.train(transitions, other_agents)

