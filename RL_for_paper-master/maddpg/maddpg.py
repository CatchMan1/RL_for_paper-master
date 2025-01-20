import torch
import os
# from actor_critic import Actor, Critic
from actor_critic import Actor,Critic
class MADDPG:
    # 因为不同的agent的观察量、动作维度可能不一样，所以神经网络不同,需要agent_id来区分
    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        # 创建Actor网络和Critic网络
        # Actor网络用于选择动作，Critic网络用于评估动作
        # 实际用于智能体的行动选择，通过训练数据实时更新，根据最新的参数来选择动作
        self.actor_network = Actor(args, agent_id)
        self.critic_network = Critic(args)

        # 创建目标网络，包括目标Actor网络和目标Critic网络
        # 用于确保训练过程中的稳定性，不会直接参与训练，不会通过梯度下降更新，通过每隔一定时间用一小部分最新的actor_network，critic_network参数来更新
        # 使用_soft_update_target_network避免网络频繁更新导致的不稳定，保存训练网络参数
        self.actor_target_network = Actor(args, agent_id)
        self.critic_target_network = Critic(args)

        # 加载网络权重到目标网络，确保目标网络和Actor网络和Critic网络一开始是同步的
        # 直接加载，刚开始运行默认初始化权重，Actor网络和Critic网络训练后要复制更新到目标网络
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        # 创建Adam优化器更新权重
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)

        # 保存模型路径
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        # 加载模型
        if os.path.exists(self.model_path + '/actor_params.pkl'):
            self.actor_network.load_state_dict(torch.load(self.model_path + '/actor_params.pkl'))
            self.critic_network.load_state_dict(torch.load(self.model_path + '/critic_params.pkl'))
            print('Agent {} successfully loaded actor_network: {}'.format(self.agent_id,
                                                                          self.model_path + '/actor_params.pkl'))
            print('Agent {} successfully loaded critic_network: {}'.format(self.agent_id,
                                                                           self.model_path + '/critic_params.pkl'))

    # 软更新需要将Actor网络和Critic网络参数计算结果复制到目标网络参数的数据部分
    def _soft_update_target_network(self):
        # self.args.tau控制更新的幅度，在[0,1]间
        # 计算公式为目标参数 = （1 - tau）*原目标参数 + tau * 当前参数。
        # 目标网络每次只更新部分，从而保持稳定性
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # 训练
    # transitions：包含从环境中采样的多智能体的状态转移数据
    # other_agents：其他智能体的引用，这些智能体包含它们各自的策略
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32)
        # 训练时获取即时奖励，r就是reward
        r = transitions['r_%d' % self.agent_id]
        # 用来装每个agent经验中的各项，存储在transitions字典中
        # o、u 和 o_next，分别用于存储所有智能体的状态、动作和下一状态
        o, u, o_next = [], [], []
        # 遍历每一个智能体，提取其状态、动作和下一状态，并添加到相应的列表中
        for agent_id in range(self.args.n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])

        # 计算目标Q值
        # u_next用于存储所有智能体在下一状态下选择的动作
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    # 因为某个agent是当前agent，用其目标 Actor 网络生成下一状态的动作并添加到 u_next
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 其他智能体，其策略中的目标 Actor 网络生成下一状态的动作并添加到 u_next
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            # 使用目标 Critic 网络计算下一状态和下一动作的Q值 q_next
            q_next = self.critic_target_network(o_next, u_next).detach()
            # 目标Q值 q_next是reward 加上折扣因子 gamma 和 q_next 的乘积（对应累计折扣奖励的公式）
            # 结合了当前奖励和未来预期Q值，逼近真实Q值
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # 使用当前 Critic 网络计算当前状态和动作的Q值 q_value
        q_value = self.critic_network(o, u)
        # 求目标Q值 q_next和当前Q值 q_value 的均方误差以反向传播更新
        # critic_loss为损失函数
        # Critic 网络要通过最小化Q值预测和目标Q值之间的均方误差来训练，确保了价值网络对状态-动作对的评估更加准确，同时损失函数最小
        critic_loss = (target_q - q_value).pow(2).mean()

        # 计算 Actor 损失
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        # actor_loss 为当前 Critic 网络在所有状态和动作下预测的Q值的负均值
        # Actor 网络要通过最小化负的Critic网络评估的Q值来训练，这等效于最大化Q值，从而生成最优的策略
        actor_loss = - self.critic_network(o, u).mean()
        # if self.agent_id == 0:
        #     print('critic_loss is {}, actor_loss is {}'.format(critic_loss, actor_loss))
        # 更新网络参数
        self.actor_optim.zero_grad()
        # 反向传播计算 Actor 损失的梯度，更新 Actor 网络的参数
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        # 反向传播计算 Critic 损失的梯度，更新 Critic 网络的参数
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    # 保存模型
    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')
        torch.save(self.critic_network.state_dict(), model_path + '/' + num + '_critic_params.pkl')
