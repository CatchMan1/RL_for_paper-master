import numpy as np
import inspect
import functools


# 保存参数以创建智能体或者环境等
def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


# 从指定的场景创建一个多智能体环境，并返回该环境以及包含环境信息的 args 对象
# *********************根据具体情况进行修改，包括动作范围，取消好人坏人等*********************
def make_env(args):
    # 创建全局环境
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)# 初始全部为默认值
    # env = MultiAgentEnv(world)
    # args.n_players = env.n  # 包含敌人的所有玩家个数
    # print("n_players的数量:",args.n_players)
    args.n_agents = env.n   # 智能体的数量
    # print("n_agents的数量:",args.n_agents)
    args.obs_shape = [env.observation_space[i].shape[0] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度
    # print("obs_shape:",args.obs_shape)
    action_shape = []
    for content in env.action_space:
        action_shape.append(len(content.shape))
    args.action_shape = action_shape[:args.n_agents]  # 每一维代表该agent的act维度
    # print("动作的多少:",args.action_shape)
    # 动作的范围——[0, 1]
    args.high_action = 1
    args.low_action = 0
    # 设置存储历史动作
    return env, args
