## scenario 类定义了特定的环境场景及其规则和设置

下面是一些常见的 scenario 实例及其特定功能说明。
```
1. simple.py
功能：定义一个包含少量智能体和目标的简单场景，测试基本的追逐和导航任务。
特征：
    - 少量智能体
    - 简单的目标点
    - 基本的导航和追逐逻辑
2. simple_adversary.py
功能：设置包含数个好智能体和一个坏智能体的场景，用于测试对抗学习。
特征：
    - 多个合作智能体
    - 一个对抗智能体
    - 测试对抗学习策略
3. simple_crypto.py
功能：创建一个基于加密通信的场景，测试智能体在安全通信方面的能力。
特征：
    - 加密和解密任务
    - 通信安全
    - 智能体间信息交换
4. simple_push.py
功能：定义一个智能体需要将对手推出特定区域的场景，用于测试推挤策略。
特征：
    - 推挤任务
    - 特定区域目标
    - 力学交互
5. simple_reference.py
功能：创建一个智能体需要指向特定物体的场景，用于测试参考和指向能力。
特征：
    - 目标引用任务
    - 物体指向
    - 智能体间协调
6. simple_speaker_listener.py
功能：一个智能体作为讲者，另一个作为听者，通过口头指令完成协作任务的场景。
特征：
    - 语音通信
    - 协作任务
    - 讲者和听者角色
7. simple_spread.py
功能：定义一个多个智能体需要在空间中扩散以覆盖多个目标的场景。
特征：
    - 扩散任务
    - 空间覆盖
    - 多智能体协调
8. simple_tag.py
功能：设置包含多个追逐者和逃避者的场景，用于测试标签和反标签策略。
特征：
    - 追逐与逃避
    - 标签任务
    - 交互策略
9. simple_world_comm.py
功能：创建一个包含多个智能体的复杂通信场景，用于测试世界级别的通信策略。
特征：
    - 复杂通信任务
    - 世界级别的交互
    - 多智能体通信
```


示例代码片段
以下是如何加载和使用某个 scenario 的示例代码段：
```python
import multiagent.scenarios as scenarios

# 加载指定场景
scenario = scenarios.load("simple_spread.py").Scenario()

# 创建世界
world = scenario.make_world()

# 创建多智能体环境
from multiagent.environment import MultiAgentEnv

env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

# 使用环境进行操作
obs_n = env.reset()
for agent in env.world.policy_agents:
    print(agent.name)
    # 打印每个智能体的初始观察值
    print(env._get_obs(agent))

```
