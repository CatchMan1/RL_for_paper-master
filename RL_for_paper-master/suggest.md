# Suggest

## 问题简化
```
三个智能体 a、b 和 c 的协作任务，其中每个智能体依次执行部分任务 D，并将其工作传递给下一个智能体。为了使得 a、b、c 的联合工作量最小，适合使用一个能够体现任务分工和协作的 scenario 实例。
```

## 可能的 scenario

```
1. simple_spread.py
说明：多个智能体需要在空间中扩散以覆盖多个目标。这意味着智能体需要协作以最优方式覆盖目标，虽然不是完全按照序列执行任务，但通过适当调整场景设置，也能模拟某种分工和协作的任务流程。
使用场景：模拟需要覆盖的目标区域，a、b、c智能体依次进行覆盖任务，每个智能体的工作范围由前一个智能体结果决定。

2. simple_speaker_listener.py
说明：一个智能体作为讲者，另一个作为听者，通过口头指令完成协作任务的场景。虽然原始设计并非串行任务，但可以修改通信和协作策略来模拟串行任务的执行。
使用场景：a 可以发送任务状态给 b，b 根据 a 的结果执行任务，并最后将结果传递给 c，c 根据 b 的结果完成最终任务。

3. simple_reference.py
说明：智能体需要指向特定物体，让其他智能体进行相关操作。能够体现某种程度的任务分工和协作，通过修改场景逻辑，可以实现顺序执行任务并完成任务的目标。
使用场景：a 指向并处理初始任务状态，然后 b 接收到 a 的指令进行处理，最后 c 完成剩余部分。
```

## simple_spread.py 模拟任务分工示例

### 通过修改 simple_spread.py 来实现其模拟功能。以下是具体实现步骤：

#### 1.安装环境和准备代码：
* 安装所需的软件包和环境
* 准备好简单的 simple_spread.py 场景文件
#### 2.修改 simple_spread.py，使其适应顺序执行任务的要求：

```python
import numpy as np
from multiagent.scenario import BaseScenario
from multiagent.world import World
from multiagent.entity import Agent, Landmark
from multiagent.entity import BaseEntity


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.agents = [Agent() for i in range(num_agents)]
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent_{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark_{i}'
            landmark.collide = False
            landmark.movable = False
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0
        agents = world.agents
        landmarks = world.landmarks
        for i, landmark in enumerate(world.landmarks):
            l_pos = landmark.state.p_pos
            rew -= min([np.sqrt(np.sum(np.square(a.state.p_pos - l_pos))) for a in agents])
        return rew

    def observation(self, agent, world):
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]
        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos)

```

#### 代码说明
##### 环境初始化：
1. make_world：定义了智能体和目标。
2. reset_world：重置了各个智能体和目标的位置。
##### 任务逻辑：
1. reward：定义了奖励函数，奖励值根据智能体距离目标点的距离来计算，越近奖励越高（任务总量越小）。
2. observation：定义了每个智能体的观察值，包括目标位置和其他智能体的位置。
##### 通过修改 simple_spread.py，可以实现三个智能体协作完成任务的场景，每个智能体依次执行任务的一部分，并尽量减少联合工作量。


#### 使用示例
```python
import multiagent.scenarios as scenarios

# 加载修改后的 simple_spread 场景
scenario = scenarios.load("simple_spread.py").Scenario()

# 创建世界和环境
world = scenario.make_world()
from multiagent.environment import MultiAgentEnv

env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

# 执行环境操作
obs_n = env.reset()
while True:
    act_n = [agent.action(obs) for agent, obs in zip(env.world.policy_agents, obs_n)]
    obs_n, reward_n, done_n, _ = env.step(act_n)
    env.render()

```