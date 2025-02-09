import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
# from multiagent.multi_discrete import MultiDiscrete
from multi_discrete import MultiDiscrete
from replay_buffer import Buffer

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)  # agent的数量
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = False # 动作空间为连续的[0,1]范围
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        # configure spaces
        self.action_space = []
        self.observation_space = []
        # 对每个智能体进行动作的量化描述u:动作;o:观察状态（空间维度设置）
        # *********************取消了智能体之间的交互动作*********************
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:  # 如果是离散的动作空间
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)  # 离散的动作数量{0，1，2，3，4}
            else:
                u_action_space = spaces.Box(low=agent.d_range, high=agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)  # 连续的范围
            if agent.movable:
                total_action_space.append(u_action_space)


            # communication action space 交互动作
            if self.discrete_action_space:# 如果动作空间离散
                c_action_space = spaces.Discrete(world.dim_c)
            else: # 连续动作
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)

            if not agent.silent:
                total_action_space.append(c_action_space)

            # print("total_action_space:",total_action_space)

            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space 观测量维度——见simple_paper的observation
            # 确定观测的维度
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            # agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

# *********************更新了智能体的具体迭代动作*********************
# 使用step_1进行观测量、动作等的迭代
    def step_1(self, action_n, total_u=None):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        reward = 0
        self.agents = self.world.policy_agents
        # set action for each agent
        if total_u == None:
            for i, agent in enumerate(self.agents):
                self._set_action(action_n[i], agent, self.action_space[i])
        else:
            for i, agent in enumerate(self.agents):
                self._set_action(action_n[i], agent, self.action_space[i], total_u[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        # 三个智能体根据自身权重获得总奖励
        if all(a.task_done for a in self.agents):
            for a in self.agents:
                reward += 10 * a.weight  # 根据每个 agent 的权重给予奖励

        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

# *********************对智能体的动作进行了动作定义*********************
# 动作定义，采用连续动作服从高斯分布
    def get_total_action(self, history_u):
        flattened = np.array([item for sublist in history_u for item in sublist])
        # 根据历史动作计算均值 mu 和标准差 sigma
        mu = np.mean(flattened)
        sigma = np.std(flattened)
        # 从高斯分布中采样动作
        action = np.random.normal(mu, sigma)
        # 将动作限制在 [0, 1] 范围内
        action = np.clip(action, 0, 1)
        action = np.array([action])
        return action

    # 重置环境到初始状态，并重新记录所有智能体的观察值，确保每个回合的开始都是从相同或预定义的初始状态开始，便于训练和评估
    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # 用于判断特定智能体是否完成
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

# *********************修改了智能体具体的动作输出*********************
# 进行动作的输出
    def _set_action(self, action, agent, action_space, history_u = None, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        # agent.action.c = np.zeros(self.world.dim_c)
        # 判断是否为离散空间
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        # 在本项目中不是离散空间
        else:
            action = [action]

        if agent.movable:
            # physical action
            # 判断是否为离散输入
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)# 初始化动作
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            # 本项目中不是离散，走else
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                # 走这一步，得到连续动作, 这里要定义动作
                else:
                    # 定义为如果没有历史值时
                    if not isinstance(history_u, np.ndarray):
                        agent.action.u = action[0]

                    # 缓冲区有历史值时走这里
                    else:
                        agent.action.u = self.get_total_action(history_u)

            # 设置sensitivity为1 需根据实际情况修改
            sensitivity = 1.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
                print("action.c:",action[0])
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0
    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    # def step(self, action_n, time):
    #     obs_n = []
    #     reward_n = []
    #     done_n = []
    #     info_n = {'n': []}
    #     i = 0
    #     for env in self.env_batch:
    #         obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
    #         i += env.n
    #         obs_n += obs
    #         # reward = [r / len(self.env_batch) for r in reward]
    #         reward_n += reward
    #         done_n += done
    #     return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
