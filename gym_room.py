"""
# With PPO2, with the default parameters, the learning was fluctuating at the steady state by more than the offset value degC.
# In RL, when there is fluctuation at the steady state, I needed to decrease the learning rate (in my case from 0.001 to 0.0001) and increase the batch size by a factor of 10
# the batch size is represented by the parameter n_steps in PPO2 from Stable Baseline, and its defalut value is 128.

# Also for the reward function, I specified to give a reward of 1 for any error less than 0.5 deg C(offset value) and
# and reduce the reward with a decreasing exponential function
"""
import argparse
import pathlib
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import DQN, PPO, SAC
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from torch import nn

from ac_room import PIDController, Room

# 项目根目录
PROJECT_ROOT = pathlib.Path(__file__).parent
# Tensorboard 日志储存路径
LOG_PATH = PROJECT_ROOT / "logs"
# 模型文件储存路径
CKPT_PATH = PROJECT_ROOT / "checkpoints"
# 命令行参数解析器
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--mode", choices=["train", "predict"], default="train",
                    help="train: train the model; predict: predict the model")


class GymACRoom(gym.Env):
    """
    Gym environment for the air conditioning room.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        T_set: float = 25,
        mC: float = 300,
        K: float = 20,
        Q_AC_max: float = 1000,
        simulation_time: int = 12 * 60 * 60,
        control_step: int = 300
    ):
        """
        初始化模拟环境。

        Args:
            - T_set: 目标温度，单位为 degC。
            - mC: 房间的热容量，单位为 kg.kj/(kg.degC)。
            - K: 房间的传热系数，单位为 W/(m.degC)。
            - Q_AC_Max: 空调的最大功率，单位为 W。
            - simulation_time: 模拟的总时间，单位为秒。
            - control_step: 控制步长，单位为秒。
        """
        super().__init__()
        # 目标温度
        self.T_set = T_set
        # 创建房间传热模型
        self.ac_room = Room(
            mC=mC,
            K=K,
            Q_AC_max=Q_AC_max,
            simulation_time=simulation_time,
            control_step=control_step
        )

        # 模拟时间步长（秒）, 空调最大功率
        self.timestep = control_step
        self.Q_AC_max = Q_AC_max

        # RL 的动作空间：空调功率的控制比例，范围为 [-1, 1]，负数表示制冷，正数表示制热
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        # RL 的观测空间（即状态）：房间温度的偏差，范围为 [-100, 100]
        # n_obs = 1 表示状态的维度
        # 如果只使用当前房间内温度和设定温度的偏差作为状态，那么 n_obs = 1
        # 如果还要使用其他状态，比如外部温度，空调功率等，那么 n_obs > 1
        # 通常来说，状态越多，模型的表达能力越强，但训练难度也越大
        n_obs = 1
        self.observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(n_obs,), dtype=np.float32
        )
        # 初始状态
        self.observation = np.empty(n_obs, dtype=self.observation_space.dtype)

        # 模拟过程中的变量
        self.iter = None

    def reset(self, **kwargs):
        """
        重置模拟环境。

        P.S. reset 方法不支持其它自定义参数

        Returns:
            - observation: 初始状态。
        """
        self.ac_room.reset(T_in=generate_init_temperature(), T_set=self.T_set)
        self.iter = 0
        self.observation[0] = self.ac_room.T_in - self.ac_room.T_set
        info = {}
        return self.observation, info

    def sync_init_temperture(self, init_T_in: float):
        """
        同步两个环境的初始温度。

        Args:
            - T_in: 初始温度。

        Returns:
            - observation: 初始状态。
        """
        self.ac_room.T_in = init_T_in
        self.observation[0] = self.ac_room.T_in - self.ac_room.T_set
        return self.observation

    def step(self, action: float):
        # 更新房间内温度
        self.ac_room.update_T_in(action=action)
        # 更新状态
        self.observation[0] = self.ac_room.T_in - self.ac_room.T_set
        # 根据迭代次数判断是否结束
        done = self.iter >= self.ac_room.max_iteration - 1

        # 根据状态计算奖励
        # reward NoOffSet
        reward = np.exp(-(abs(10 * self.observation))).item()

        # # Reward with Offset
        # if abs(self.observation) < 0.5:
        #     reward = 1
        # else:
        #     reward = np.exp(-(abs(self.observation)-0.5))

        # 其它需要返回的信息
        info = {}
        # 是否截断
        truncated = False

        # 更新迭代次数
        self.iter += 1
        return self.observation, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass


def generate_init_temperature():
    """
    生成一个随机的初始温度。
    """
    return np.random.randint(20, 30)


if __name__ == "__main__":
    # 获取命令行参数
    args = PARSER.parse_args()
    mode = args.mode

    # 设定随机种子
    seed = 0
    np.random.seed(seed)

    # 设定模拟环境的参数
    env_params = {
        "mC": 300,
        "K": 20,
        "Q_AC_max": 1000,
        "simulation_time": 12 * 60 * 60,
        "control_step": 300
    }

    # 设定目标温度、目标温度的容忍范围
    T_set = 25
    margin = 0.5

    # 设定 PI / PID 控制器
    kp = 0.025
    ki = 0.02
    kd = 0.015
    pid_controller = PIDController(kp, ki, kd, T_set, mode="PI")

    # 设定 Policy 网络
    policy_kwargs = {
        # "feature_extraction": "mlp",
        "activation_fn": nn.ReLU,
        "net_arch": {"pi": [8, 8], "vf": [8, 8]}
    }

    if mode == "train":
        # train 模式

        # 用于 RL 的环境
        env_rl = GymACRoom(**env_params)
        check_env(env_rl)
        # 学习率
        # 可以依次尝试多个学习率，用于寻找最优参数
        learning_rate = [0.0001]

        for lr in learning_rate:
            model = PPO(
                policy="MlpPolicy",
                env=env_rl,
                policy_kwargs=policy_kwargs,
                verbose=1,
                learning_rate=lr,
                n_steps=1280,
                tensorboard_log=LOG_PATH,
            )
            model.learn(total_timesteps=1000000)
            model.save(str(CKPT_PATH / "AC_PPO_exp_no_offset.zip"))

    else:
        # predict 模式
        # 使用训练好的模型进行预测
        # 同时使用 PI 控制器进行对比

        # 迭代次数
        n_iter = 1000

        # RL 环境 & PI 控制器环境
        env_rl = GymACRoom(**env_params)
        env_pi = GymACRoom(**env_params)
        # 初始化环境，保证两个环境的初始状态一致
        obs_rl, _ = env_rl.reset()
        obs_pi, _ = env_pi.reset()
        # 同步初始温度
        obs_pi = env_pi.sync_init_temperture(env_rl.ac_room.T_in)

        # 加载训练好的模型
        model = PPO.load(str(CKPT_PATH / "AC_PPO_exp_no_offset.zip"))

        # 记录内部温度、目标温度、目标温度上下限、时间的变化序列
        T_in_rl = np.empty(n_iter)
        T_in_pi = np.empty(n_iter)
        T_set = env_rl.ac_room.T_set * np.ones_like(T_in_rl)
        T_set_high = T_set + 0.5
        T_set_low = T_set - 0.5
        T_seq = np.empty(n_iter)

        for i in range(n_iter):
            # 记录当前时间步的时间（分钟）
            # 以及做出动作前的房间温度
            T_seq[i] = env_rl.timestep / 60 * i
            T_in_rl[i] = env_rl.ac_room.T_in
            T_in_pi[i] = env_pi.ac_room.T_in

            # RL
            # 策略 + 状态 -> 动作
            # 动作 -> 环境 -> 奖励 & 状态
            action_rl, _states = model.predict(obs_rl)
            obs_rl, rewards_rl, dones_rl, _, info_rl = env_rl.step(action_rl)
            # print("RL: Action {} -> State {}".format(action_rl, obs_rl))

            # PI
            # 温差 -> 控制信号
            # 控制信号 -> 环境 -> 奖励 & 状态
            error = obs_pi[0]
            control_signal = pid_controller.update_by_error(error)
            if abs(error) <= margin:
                action_pi = 0
            else:
                action_pi = control_signal
            obs_pi, rewards_pi, dones_pi, _,  info_pi = env_pi.step(
                action=action_pi)
            # print("PI: Action {} -> State {}".format(action_pi, obs_pi))

            # 两个环境模拟的步数都是一样的，所以只需要检查一个环境是否结束
            # 如果结束，那么重置环境
            if dones_rl is True:
                obs_rl, _ = env_rl.reset()
                obs_pi, _ = env_pi.reset()
                obs_pi = env_pi.sync_init_temperture(env_rl.ac_room.T_in)

        # 绘制温度变化曲线
        fig, ax = plt.subplots()
        plt.plot(T_seq, T_in_rl, "r--", label="RL_PPO")
        plt.plot(T_seq, T_in_pi, "b--", label="PI")
        plt.plot(T_seq, T_set, "g--", label="T_set")
        plt.plot(T_seq, T_set_high, "k--", label="T_set_high")
        plt.plot(T_seq, T_set_low, "k--", label="T_set_low")
        plt.xlabel("Iteration time (min)")
        plt.ylabel("Temperature (deg. C)")
        plt.legend()
        plt.show()

        # plotly 绘制可交互图
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T_seq, y=T_in_rl, name="T_in_RL",
                      mode="markers", marker_color="rgba(0, 200, 0, .8)"))
        fig.add_trace(go.Scatter(x=T_seq, y=T_in_pi, name="T_in_PI",
                      mode="markers", marker_color="rgba(200, 0, 0, .8)"))
        fig.add_trace(go.Scatter(x=T_seq, y=T_set, name="Tset",
                      mode="markers", marker_color="rgba(0, 0, 200, 1)"))
        fig.show()

        # to see the reward episod progress, get the http address by pasting the following command in the conda directory where this code is running.
        # tensorboard --logdir ./checkpoints/
