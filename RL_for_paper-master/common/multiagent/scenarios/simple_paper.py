import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.agents = [Agent() for i in range(num_agents)]
        world.landmarks = [Landmark() for i in range(num_landmarks)]# 基准数据
        weights = [1.0, 1.5, 2.0]  # 示例权重
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = False # 不碰撞
            agent.silent = True # 通信
            agent.size = 0.0
            agent.task_done = False  # 增加任务状态
            agent.weight = weights[i]  # 初始化权重
            agent.color = np.array([0.35, 0.35, 0.85])  # 设置颜色
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark_{i}'
            landmark.collide = False
            landmark.movable = False
            landmark.color = np.array([0.25, 0.25, 0.25])  # 设置颜色
        self.reset_world(world)
        return world
# *********************增加观测量的范围*********************
    def reset_world(self, world):
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p) # 可视化坐标
            agent.state.p_vel = np.zeros(world.dim_p) #
            agent.state.c = np.zeros(world.dim_c) # 通信维度
            agent.task_done = False
            # 需要根据数据情况修改
            agent.t_control = np.random.uniform(0,100,1)# 控制时段范围
            agent.gas_flow = np.random.uniform(0,100,1)# 气体流量范围
            agent.gas_pressure = np.random.uniform(0,100,1) # 气体压力范围
            agent.gas_temperture = np.random.uniform(0,100,1)# 气温范围
            agent.paper_fiber = np.random.uniform(0,100,1)# 干燥纸页的绝干纤维物料量范围
            agent.R_s = np.random.uniform(0.48, 0.92,1)# 纸页开始干度范围——要调整
            agent.R_b = np.random.uniform(0.48, 0.92,1)# 纸页终止干度范围
            agent.T_s = np.random.uniform(20, 100,1)# 起始温度
            agent.T_b = np.random.uniform(20, 100,1)# 终止温度
            # 根据上面四个数计算吸取的热量：(公式要改)——对应公式9
            agent.heat_absorbed = (agent.R_b - agent.R_s)*(agent.T_b-agent.T_s)
        for i, landmark in enumerate(world.landmarks):# 基准
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):# 获取基准数据
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):# 是否收集基准数据——可视化
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    # def good_agents(self, world):
    #     return [agent for agent in world.agents if not agent.adversary]
    # *********************重设奖励函数*********************
    # 奖励函数的设置
    def reward(self, agent, world):
        rew = 0
        if agent.task_done:
            r_bas = agent.get_gas_cost
            r_vap = agent.get_steam_cost
            r_air = agent.get_fresh_steam_cost
            r_thr = agent.paper_is_dry
            rew = r_bas + r_vap + r_air + r_thr
        return rew

    # *********************根据新增的观测量进行相应修改*********************
    def observation(self, agent, world):
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]
        # task_status = [other.task_done for other in world.agents]
        # 需要添加的观测量
        t_control = [agent.t_control]
        gas_flow = [agent.gas_flow]
        gas_pressure = [agent.gas_pressure]
        gas_temperture = [agent.gas_temperture]
        paper_fiber = [agent.paper_fiber]
        R_s = [agent.R_s]
        R_b = [agent.R_b]
        T_s = [agent.T_s]
        T_b = [agent.T_b]
        heat_absorbed = [agent.heat_absorbed]

        return np.concatenate(
            [agent.state.p_pos] + entity_pos + other_pos + t_control + gas_flow + gas_pressure + gas_temperture + paper_fiber + R_s + R_b + T_s + T_b + heat_absorbed)


    def is_task_done(self, agent): # 增加随机时间和干度，随机范围
        # 根据 agent 的 ID 或其他属性判断任务是否完成（需要修改判断条件）
        if agent.agent_id == 0:
            if agent.R_s > 0.8 and agent.T_s > 80:
                agent.task_done = True
            else:
                agent.task_done = False
        elif agent.agent_id == 1:
            if agent.R_b > 0.8 and agent.T_b > 80:
                agent.task_done = True
            else:
                agent.task_done = False
        elif agent.agent_id == 2:
            if agent.heat_absorbed > 100:
                agent.task_done = True
            else:
                agent.task_done = False
        else:
            agent.task_done = False