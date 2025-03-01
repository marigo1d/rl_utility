import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库
import gym

import rl_utils


# class CliffWalkingEnv:
#     def __init__(self, ncol, nrow):
#         self.nrow = nrow
#         self.ncol = ncol
#         self.x = 0  # 记录当前智能体位置的横坐标
#         self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标
#
#     def step(self, action):  # 外部调用这个函数来改变当前位置
#         # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
#         # 定义在左上角
#         change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
#         self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
#         self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
#         next_state = self.y * self.ncol + self.x
#         reward = -1
#         done = False
#         if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
#             done = True
#             if self.x != self.ncol - 1:
#                 reward = -100
#         return next_state, reward, done
#
#     def reset(self):  # 回归初始状态,坐标轴原点在左上角
#         self.x = 0
#         self.y = self.nrow - 1
#         return self.y * self.ncol + self.x

def discretize_state(observation, n_states):
    """将连续状态转换为离散状态"""
    env_low = [-4.8, -3.4 * 10 ** 38, -0.42, -3.4 * 10 ** 38]
    env_high = [4.8, 3.4 * 10 ** 38, 0.42, 3.4 * 10 ** 38]
    state = [0] * len(observation)
    for i, s in enumerate(observation):
        # 将每个维度都线性映射到[0, 1]
        state[i] = (s - env_low[i]) / (env_high[i] - env_low[i])
        # 将每个维度都离散化为n_states个状态
        state[i] = int(state[i] * (n_states - 1))
    # 将多维状态转换为一维状态
    return state[0] * n_states ** 3 + state[1] * n_states ** 2 + state[2] * n_states + state[3]


class Sarsa:
    """ Sarsa算法 """

    def __init__(self, n_states, n_actions, epsilon, epsilon_decay, epsilon_min, alpha, gamma):
        self.Q_table = np.zeros([n_states, n_actions])  # 初始化Q(s,a)表格
        self.n_actions = n_actions  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def update(self, s0, a0, r, s1, a1, done):
        if done:
            target = r
        else:
            target = r + self.gamma * self.Q_table[s1, a1]
        td_error = target - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

        # 更新epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class nstep_Sarsa:
    """ n步Sarsa算法 """

    def __init__(self, n, n_states, epsilon, epsilon_decay, epsilon_min, alpha, gamma, n_action):
        self.Q_table = np.zeros([n_states, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n = n  # 采用n步Sarsa算法
        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:  # 若保存的数据可以进行n步更新
            G = self.Q_table[s1, a1]  # 得到Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]  # 不断向前计算每一步的回报
                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)  # 将需要更新的状态动作从列表中删除,下次不必更新
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            # n步Sarsa的主要更新步骤
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []
            # 衰减epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    np.random.seed(0)
    alpha = 0.1
    epsilon = 1.0  # epsilon初始值
    epsilon_decay = 0.995  # epsilon衰减率
    epsilon_min = 0.01  # epsilon最小值
    gamma = 0.9
    n_states = 20  # 每个维度离散化为20个状态
    n_actions = env.action_space.n
    agent = Sarsa(n_states ** 4, n_actions, epsilon, epsilon_decay, epsilon_min, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()[0]
                # 将连续状态转换为离散状态
                state = discretize_state(state, n_states)
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, done, info, _ = env.step(action)
                    # 将连续状态转换为离散状态
                    next_state = discretize_state(next_state, n_states)
                    next_action = agent.take_action(next_state)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state, next_action,
                                 done)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('CartPole-v1'))

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.figure()
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()
