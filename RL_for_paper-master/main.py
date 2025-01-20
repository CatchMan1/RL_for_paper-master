from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch
import os
os.environ['SUPPRESS_MA_PROMPT'] = '1'


# python main.py --scenario-name=simple_tag --evaluate-episodes=10

if __name__ == '__main__':
    # get the params
    args = get_args()# 初始化基础参数
    env, args = make_env(args)# 在多智能体下的参数增设
    runner = Runner(args, env)# 初始化智能体
    if args.evaluate:# 初始第一次默认为False
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
