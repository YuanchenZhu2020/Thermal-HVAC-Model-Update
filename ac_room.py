"""
房间温度的 PI / PID 控制器模拟，房间温度的变化由以下方程描述：

T_in(t+1) = T_in(t) - 0.001 * (timestep / mC) * (K * (T_in(t) - T_out(t)) + Q_AC(t))

其中，T_in(t) 为时刻 t 的房间温度，T_out(t) 为时刻 t 的外部温度，Q_AC(t) 为时刻 t 的空调功率，mC 为房间的热容量，K 为房间的传热系数。

PI / PID 控制器的控制信号为：

control_signal(t) = p_controller(error(t)) + I_controller(error_cum(t)) + D_controller(error_diff(t))

其中，error(t) 为时刻 t 的温度误差，error_cum(t) 为时刻 t 之前的温度误差累积，error_diff(t) 为 t 时刻的差分误差。

在本模拟中，房间的外部温度 T_out(t) 为一个随时间变化的序列，前半段为 28 度，后半段为 32 度。

房间的初始温度 T_in(0) 为一个随机数，范围为 20-40 度。

PI / PID 控制器的参数为：

    kp = 0.025, ki = 0.02, kd = 0.01

房间的参数为：

    mC = 300 kg.kj/(kg.degC)
    K = 20 W/(m.degC)
    Q_AC_Max = 1500 W
    timestep = 300 sec
    simulation_time = 12*60*60 sec
    control_step = 300 sec

房间的目标温度 T_set = 25 度，容忍范围为 0.5 度。
"""
import matplotlib.pyplot as plt
import numpy as np


class Room:
    """
    房间的模拟环境。
    """

    def __init__(
        self,
        mC: float = 300,
        K: float = 20,
        Q_AC_max: float = 1500,
        simulation_time: int = 12 * 60 * 60,
        control_step: int = 300
    ):
        """
        初始化模拟环境。

        Args:
            - mC: 房间的热容量，单位为 kg.kj/(kg.degC)。
            - K: 房间的传热系数，单位为 W/(m.degC)。
            - Q_AC_Max: 空调的最大功率，单位为 W。
            - simulation_time: 模拟的总时间，单位为秒。
            - control_step: 控制步长，单位为秒。
        """
        self.timestep = control_step
        self.max_iteration = int(simulation_time / self.timestep)

        self.mC = mC
        self.K = K
        self.Q_AC_max = Q_AC_max

        # 模拟过程中的变量
        self.iter = None
        self.T_in = None
        self.Q_AC = None
        self.T_set = None
        self.T_out = np.empty(self.max_iteration)

    def reset(self, T_in: int = 20, T_set: int = 25):
        """
        将房间的状态重置为初始状态，即设定初始温度 T_in，重置迭代次数 iteration 为 0

        Args:
            - T_in: 房间的初始温度。
        """
        self.iter = 0
        self.schedule(T_set)
        self.T_in = T_in

    def schedule(self, T_set: int):
        """
        设定房屋目标温度 T_set 和外部温度 T_out。

        T_out 为一个随时间变化的序列，在迭代过程中，前半段为 28 度，后半段为 32 度。

        Args:
            - T_set: 房屋目标温度
        """
        # 目标温度
        self.T_set = T_set
        # 重置外部温度变化序列，前半段为 28 度，后半段为 32 度
        self.T_out[:int(self.max_iteration/2)] = 28
        self.T_out[int(self.max_iteration/2):int(self.max_iteration)] = 32

    def update_T_in(self, action: float):
        """
        更新房间的温度状态。

        下面解释了房间温度的变化方程：

        - T_in - T_out: 房间内外温度差，单位为摄氏度。
        - K * (T_in - T_out): 假设房间墙壁厚度为 1 米，计算房间的每秒传热量，单位为 W。
        - Q_AC: 表示空调系统每秒提供或吸收的热量，单位为 W。
        - K * (T_in - T_out) + Q_AC: 用于计算房间每秒的温度变化，单位为 W，即 J/s。
        - 0.001 * (timestep / mC) * (K * (T_in - T_out) + Q_AC): 
            用于计算房间每步的温度变化，单位为摄氏度。

            - 其中 0.001 为单位转换系数，将 W 转换为 kj/s。
            - 每秒的温度变化量 (kj/s) 除以房间的热容量 mC (kj/degC)，得到每秒的温度变化量 (degC/s)。
            - 每秒的温度变化量 (degC/s) 乘以控制步长 timestep (s)，得到每步的温度变化量 (degC)。

        Args:
            - action: 控制比例，按照这个比例决定空调系统的功率，正数表示制冷，负数表示加热。
        """
        self.Q_AC = action * self.Q_AC_max
        self.T_in = self.T_in - 0.001 * \
            (self.timestep / self.mC) * \
            (self.K * (self.T_in - self.T_out[self.iter]) + self.Q_AC)
        self.iter += 1


class PIDController:
    """
    PI / PID 控制器
    """

    def __init__(
        self,
        kp: float, ki: float, kd: float = None,
        set_point: float = 0,
        mode: str = "PI"
    ) -> None:
        """
        初始化 PI / PID 控制器。

        Args:
            - kp: 比例增益。
            - ki: 积分增益。
            - kd: 微分增益。
            - set_point: 设定值。
            - mode: 控制器类型，"PI" 或 "PID"。
        """
        self.mode = mode
        # PID 控制器参数
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # 设定值
        self.set_point = set_point
        # 累积误差 & 上一次误差
        # 分别用来计算积分和微分
        self.error_cum = 0
        self.prev_error = 0

    def get_pi_signal(self, error: float) -> float:
        """
        获取 PI 控制器的控制信号。

        Args:
            - error: 和设定值之间的误差。

        Returns:
            - control_signal: 控制信号。
        """
        control_signal = self.kp * error + self.ki * self.error_cum
        return control_signal

    def get_pid_signal(self, error: float) -> float:
        """
        获取 PID 控制器的控制信号。

        Args:
            - error: 和设定值之间的误差。

        Returns:
            - control_signal: 控制信号。
        """
        pi_signal = self.get_pi_signal(error)
        control_signal = pi_signal + self.kd * (error - self.prev_error)
        return control_signal

    def step(self, current_value: float):
        """
        更新控制器，获取当前时间步的控制信号。

        Args:
            - current_value: 当前测量值。

        Returns:
            - control_signal: 控制信号。
        """
        # 计算当前时间步的误差
        error = current_value - self.set_point
        # 利用误差更新控制器参数，并返回当前时间步的控制信号
        return self.update_by_error(error)

    def update_by_error(self, error: float):
        """
        根据当前误差更新控制器参数，并返回当前时间步的控制信号。

        Args:
            - error: 当前误差。
        """
        # 更新积分误差
        self.error_cum += error
        # 根据 mode 计算控制信号
        if self.mode == "PI":
            control_signal = self.get_pi_signal(error)
        elif self.mode == "PID":
            control_signal = self.get_pid_signal(error)
        else:
            raise ValueError("PIDCotroller mode must be 'PI' or 'PID'.")
        # 更新前一次的误差，用于下一个时间步计算误差微分
        self.prev_error = error
        return control_signal


if __name__ == "__main__":
    # 设定随机种子，使得每次运行的结果一致
    seed = 0
    np.random.seed(seed)

    # 设定模拟环境的参数
    mC = 300
    K = 20
    Q_AC_max = 1500
    simulation_time = 12 * 60 * 60
    control_step = 300

    # 设定模拟的迭代次数
    n_iter = 500
    # 设定初始温度、目标温度、目标温度的容忍范围
    T_in = np.random.randint(20, 40)
    T_set = 25
    margin = 0.5

    # 设定 PI / PID 控制器
    kp = 0.025
    ki = 0.02
    kd = 0.015
    pid_controller = PIDController(kp, ki, kd, T_set, mode="PID")

    # 创建房间对象，初始化房间温度和目标温度
    room = Room(
        mC=mC,
        K=K,
        Q_AC_max=Q_AC_max,
        simulation_time=simulation_time,
        control_step=control_step
    )
    room.reset(T_in=T_in)

    # 记录内部温度、目标温度、目标温度上下限、时间的变化序列
    T_in = np.empty(n_iter)
    T_set = room.T_set * np.ones_like(T_in)
    T_set_high = T_set + margin
    T_set_low = T_set - margin
    T_seq = np.empty(n_iter)

    for i in range(n_iter):
        # 记录当前时间步的时间（分钟）和房间温度
        T_seq[i] = room.timestep / 60 * i
        T_in[i] = room.T_in

        # PI 控制器，计算当前时间步的控制信号
        control_signal = pid_controller.step(room.T_in)

        # 如果温度误差小容忍范围，则不需要控制
        # 否则，选择一个与误差成正比的空调功率
        if abs(pid_controller.prev_error) <= margin:
            current_action = 0
        else:
            current_action = control_signal

        # 更新房间温度
        room.update_T_in(action=current_action)

        # 如果到达模拟的最大时间，则重新开始模拟
        if room.iter == room.max_iteration:
            room.reset(T_in=np.random.randint(20, 40))

    # 绘制温度变化曲线
    fig, ax = plt.subplots(1, 1)
    plt.plot(T_seq, T_in, "b--", label="T_in")
    plt.plot(T_seq, T_set, "g--", label="T_set")
    plt.plot(T_seq, T_set_high, "k--", label="T_set_high")
    plt.plot(T_seq, T_set_low, "k--", label="T_set_low")
    plt.xlabel("Iteration time (min)")
    plt.ylabel("Temperature (deg. C)")
    plt.legend()
    plt.show()
