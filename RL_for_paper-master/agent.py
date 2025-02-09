import random
import math
import numpy as np
import torch
import os
from maddpg import MADDPG
# 定义了智能体的类
# *********************新增了智能体的观测量o*********************
class Agent:
    def __init__(self, agent_id, args):
        # 常数
        self.RA = 8.314 # 气体常数
        self.M_air = 22.4 # L/mol
        self.K = 0.1 # 损失系数
        self.n = 0.8 # 烘缸的干燥效率
        # 安托万常数，适用于水，温度范围 1 - 100°C
        self.A = 8.07131
        self.B = 1730.63
        self.C = 233.426
        # 理想干度
        self.Ideal_R = 0.92
        # 创建包含各种参数的对象
        self.args = args
        # 智能体的ID
        self.agent_id = agent_id
        # 每个智能体创建一个多智能体深度确定性策略梯度对象，以便后续使用该策略来选择动作和学习
        self.policy = MADDPG(args, agent_id)
        # 新观测量
        self.t_control = 300 # 控制时间t 300s 5min
        self.gas_flow = 0.0 # 干燥气体的体积流量
        self.gas_pressure = 0.0 # 气体压力
        self.gas_temperture = 0.0 # 气体的绝对温度
        self.paper_fiber = 0.0 #干燥纸业的绝干纤维物量
        # 三元组表示
        self.R_s = 0.0  # 起始干度 R_(0:t/∆t)^s
        self.R_b = 0.0  # 终止干度 R_(0:t/∆t)^b
        self.T_s = 0.0  # 起始温度 T_(0:t/∆t)^s
        self.T_b = 0.0  # 终止温度 T_(0:t/∆t)^b
        self.heat_absorbed = 0.0  # 吸取的热量

        # 新增可视化指标（LSTM预测指标）
        self.C_Rh = 0.0 # 蒸汽散热过程中蒸汽的热量比
        self.E_mix = 0.0 # 蒸汽混合模块配比混合效率（即5分钟内的平均每秒生成的米数）
        self.W_steam = 0.0 # t时刻通入汽水分离器的物料量
        self.C_Rs = 0.0 # 经过汽水分离器的蒸汽（乏汽）量比
        self.H_steam = 2260 # t时刻（即5分钟内平均）通入的新鲜蒸汽的焓值 KJ/mol
        self.H_steam_water = 0.0 # 烘缸组流到汽水分离器的冷凝水的焓值
        self.H_steam_vapor = 0.0 # 烘缸组流到汽水分离器的蒸汽的焓值
        self.P_second_steam = 0.0 # 二次蒸汽的压力
        self.T_second_steam = 0.0 # 二次蒸汽的绝对温度
        self.V_second_steam = 0.0 # 二次蒸汽的体积流量
        self.H_second_steam = 0.0 # 二次蒸汽的晗值

        self.task_done = False  # 任务完成状态

    # 计算水的汽化热
    def antoine_equation(self, T):
        """
        使用安托万方程计算水在给定温度下的饱和蒸气压 (mmHg)
        :param T: 温度 (°C)
        :return: 饱和蒸气压 (mmHg)
        """
        return 10 ** (self.A - self.B / (T + self.C))

    # 计算水的汽化热
    def clausius_clapeyron(self, T1, P1, T2, P2):
        """
        使用克劳修斯 - 克拉佩龙方程计算汽化热 (J/mol)
        :param T1: 温度 1 (K)
        :param P1: 温度 1 对应的饱和蒸气压 (mmHg)
        :param T2: 温度 2 (K)
        :param P2: 温度 2 对应的饱和蒸气压 (mmHg)
        :return: 汽化热 (J/mol)
        """
        return -self.RA * (math.log(P2 / P1) / ((1 / T2) - (1 / T1)))

    # 计算水的汽化热
    def calculate_enthalpy_of_vaporization(self, T):
        """
        计算给定温度和压力下水的汽化热 (J/mol)
        :param T: 温度 (°C)
        :param P: 压力 (mmHg)
        :return: 汽化热 (J/mol)
        """
        # 选择一个参考温度 (例如 100°C)
        T_ref = 100
        P_ref = self.antoine_equation(T_ref)

        # 将温度转换为开尔文
        T_K = T + 273.15
        T_ref_K = T_ref + 273.15

        # 计算给定温度下的饱和蒸气压
        P_T = self.antoine_equation(T)

        # 使用克劳修斯 - 克拉佩龙方程计算汽化热
        return self.clausius_clapeyron(T_ref_K, P_ref, T_K, P_T)

    # 汽水分离过程中热量损失
    def I_lamda(self):
        I = self.K * self.M_air * abs(self.H_steam_water - self.H_steam_vapor)
        return I
    # 流到汽水分离器的热量
    def O_steam(self):
        O_steam = self.W_steam * ((1 - self.C_Rs) * self.H_steam_water + self.C_Rs * self.H_steam_vapor) + self.I_lamda()
        return O_steam
    # t时刻新鲜蒸汽热量
    def I_steam(self):
        pass
    #  t时刻二次蒸汽热量
    def I_second_steam(self):
        I_second_stream = (self.P_second_steam * self.M_air / self.RA * self.T_second_steam) * self.V_second_steam * self.H_second_steam
        return I_second_stream
    # 混合模块混合气体配比热量
    def Q_steam(self):
        Q_steam = self.I_second_steam() + self.I_steam()
        return Q_steam


    # *********************增加了多个奖励函数用作计算最终奖励函数*********************
    # 用气成本物料量负值 gas_cost : r_bas
    def get_gas_cost(self):
        # 纸张质量
        m_paper = self.paper_fiber / self.R_s
        # 蒸发水分的质量
        m_steam_water = m_paper * ((1 - self.R_s) / (1 - self.R_b) - 1)
        enthalpy = self.calculate_enthalpy_of_vaporization(self.R_b)
        # 蒸发相应水分所需热量
        Q_steam = m_steam_water * enthalpy
        # 用气物料量
        W_steam = Q_steam / (self.n * self.H_steam)
        r_bas = -W_steam
        return r_bas
    # 蒸汽物料量 steam_cost : r_vap
    def get_steam_cost(self):
        # 纸张质量
        m_paper = self.paper_fiber / self.R_s
        # 理想蒸发水分量
        m_ideal_steam_water = m_paper * ((1 - self.R_s) / (1 - self.Ideal_R) - 1)
        m_steam_water = m_paper * ((1 - self.R_s) / (1 - self.R_b) - 1)
        Delta_m_water = m_steam_water - m_ideal_steam_water
        r_vap = -Delta_m_water
        return r_vap
    # 新鲜蒸汽用气成本 fresh_steam_cost: r_air
    def get_fresh_steam_cost(self):
        # 替换为具体的公式求解
        r_air = -self.W_steam
        return r_air
    # 纸张无法烘干 paper_is_dry : r_thr
    def paper_is_dry(self):
        # 增设条件
        r_thr = 0
        if self.R_b >= 0.92:# 增设烘干条件
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

