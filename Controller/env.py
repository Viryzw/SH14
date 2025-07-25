import gym
from gym import spaces
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.Manager import Manager
from Utils.Refresher import TargetRefresher
from Utils.Scorer import score1, score2

class UAVUSVEnv(gym.Env):
    def __init__(self):
        super(UAVUSVEnv, self).__init__()

        self.manager = Manager()
        self.refresher = TargetRefresher(min_interval=200, max_interval=600)
        self.max_step = 144000
        self.step_count = 0

        # UAV 和 USV 配置
        self.uav_ids = ['1', '2']
        self.usv_ids = ['1', '2', '3', '4']

        # 控制维度：每个 UAV/USV 控制 (v, ω)，共 6 个机器人 × 2 = 12
        self.action_space = spaces.Box(low=np.array([-1]*12), high=np.array([1]*12), dtype=np.float32)

        # 观测空间设计：每个 UAV/USV 的 [x, y, heading] + 探测情况
        low = np.array([0, 0, -np.pi, 0] * 6, dtype=np.float32)
        high = np.array([9260, 9260, np.pi, 1] * 6, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._init_simulation()

    def _init_simulation(self):
        self.uavs = [
            ["1", [0, 5630], 0],
            ["2", [0, 3630], 0]
        ]
        self.usvs = [
            ["1", [0, 6130], 0],
            ["2", [0, 5130], 0],
            ["3", [0, 4130], 0],
            ["4", [0, 3130], 0]
        ]
        self.targets = ['1', '2']
        self.manager.init_objects(self.uavs, self.usvs, self.targets, t=0)
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._init_simulation()
        obs = self._get_obs()
        info = {}  # 可选附加信息
        return obs, info


    def step(self, action):
        action = np.clip(action, -1, 1)  # 限制范围

        # 控制最大速度和角速度
        MAX_V = 100
        MAX_W = np.pi / 4

        # 构造控制指令
        controls = []

        # UAV 控制
        for i, uid in enumerate(self.uav_ids):
            v = float(action[i*2]) * MAX_V
            w = float(action[i*2+1]) * MAX_W
            controls.append(["uav", uid, v, w])

        # USV 控制
        offset = len(self.uav_ids) * 2
        for i, uid in enumerate(self.usv_ids):
            v = float(action[offset + i*2]) * MAX_V
            w = float(action[offset + i*2+1]) * MAX_W
            controls.append(["usv", uid, v, w])

        # 更新环境
        self.manager.update(controls, t=self.step_count)
        new_targets = self.refresher.refresh(self.step_count)
        self.manager.update_targets(new_targets, t=self.step_count)
        self.step_count += 1

        # 奖励设计
        reward = self._get_reward()
        terminated = self.step_count >= self.max_step
        truncated = False  # 或根据实际中断逻辑设置
        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        观测设计：
        - 每个 UAV/USV 的位置 [x, y] + heading
        - 是否探测到目标 (0 或 1)
        """
        obs = []
        for typ in ['uav', 'usv']:
            ids = self.uav_ids if typ == 'uav' else self.usv_ids
            for uid in ids:
                state = self.manager.get_state(typ, uid)
                if state:
                    pos, heading = state
                    detected = self.manager.get_detected(typ, uid)
                    detect_flag = 1 if detected else 0
                    obs.extend([pos[0], pos[1], heading, detect_flag])
                else:
                    obs.extend([0, 0, 0, 0])  # 缺省值
        return np.array(obs, dtype=np.float32)

    def _get_reward(self):
        """
        基于 P, S1, S2 的复合得分
        """
        if self.refresher.current_id <= 1:
            return 0.0
        P = len(self.manager.time1) / (self.refresher.current_id - 1)
        S1 = score1(self.manager.time1)
        S2 = score2(self.manager.time2)
        return P * (S1 + S2)

    def render(self, mode='human'):
        pass
