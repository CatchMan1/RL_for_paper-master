import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
# 策略网络，用于在给定的观察下决定采取何种动作
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        # gent_id: 用于区分不同的智能体
        # 使用三个全连接层输出动作
        self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # 通过tanh函数约束在动作空间范围内[0,1]，然后乘以max_action得到动作值
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


# 价值网络，用于评估给定状态和动作对的价值，即Q值。
# 从环境中获得的即时反馈作为奖励，用于更新价值
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.q_out = nn.Linear(64, 1)

    def forward(self, state, action):
        # 将状态，动作拼接在一起
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            # 归一化，数据被缩放到统一的尺度上变化，使得神经网络的训练稳定和高效，避免部分相对过大数据
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


# 通过反向传播和优化器更新价值网络示例
# 假设有一个状态，动作和下一个状态，以及即时反馈作为奖励
# state = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
# action = torch.tensor([[0.5]], dtype=torch.float32)
# next_state = torch.tensor([[1.5, 2.5]], dtype=torch.float32)
# reward = torch.tensor([[1.0]], dtype=torch.float32)
#
# critic = Critic(state_dim=2, action_dim=1)
# # 当前Q值
# current_q_value = critic(state, action)
# # 目标Q值 (假设next_action已由Actor确定)
# next_action = torch.tensor([[0.4]], dtype=torch.float32)
# next_q_value = critic(next_state, next_action)
# target_q_value = reward + 0.99 * next_q_value
#
# # 基于目标Q值和当前Q值计算损失
# loss = nn.MSELoss()(current_q_value, target_q_value.detach())
#
# # 反向传播和优化
# optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()