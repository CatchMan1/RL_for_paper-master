import threading
import numpy as np


# 经验回放缓冲区，在训练过程中进行经验采样
class Buffer:
    def __init__(self, args):
        # 可以存储的经验条数
        self.size = args.buffer_size
        self.args = args
        # 已存储的经验数量
        self.current_size = 0
        # 创建字典用于存储经验
        self.buffer = dict()
        # o，u，r，u_next 智能体i的状态，动作，奖励，下一状态
        for i in range(self.args.n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
            self.buffer['u_%d' % i] = np.empty([self.size, self.args.action_shape[i]])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, self.args.obs_shape[i]])
        # 线程锁，保护共享资源
        self.lock = threading.Lock()

    # 存储经验方法
    def store_episode(self, o, u, r, o_next):
        # 以transition的形式存，每次只存一条经验
        idxs = self._get_storage_idx(inc=1)
        for i in range(self.args.n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]

    # 从回放缓冲区中采样经验方法
    def sample(self, batch_size):
        # 创建临时字典存储采样数据
        temp_buffer = {}
        # 随机获取采样索引
        idx = np.random.randint(0, self.current_size, batch_size)
        # 根据采样索引提取保存的经验数据
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        return temp_buffer

    # 获取存储索引方法
    def _get_storage_idx(self, inc=None):
        # inc默认为1
        inc = inc or 1
        # 当前大小 + inc <= 缓冲区，生成存储索引
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        # 根据溢出量，重新合并形成经验存储字典
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        # 返回最旧的经验
        if inc == 1:
            idx = idx[0]
        return idx
