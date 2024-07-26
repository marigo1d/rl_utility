from tqdm import tqdm
import numpy as np
import torch
import collections
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


class Improved_ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max(self.priorities, default=1))  # 新样本默认最高优先级

    def sample(self, batch_size):
        # 根据优先级计算采样概率
        priorities_array = np.array(self.priorities)
        sampling_probabilities = priorities_array ** self.alpha / np.sum(priorities_array ** self.alpha)
        sample_indices = random.choices(range(len(self.buffer)), weights=sampling_probabilities, k=batch_size)

        # 采样数据
        transitions = [self.buffer[i] for i in sample_indices]
        state, action, reward, next_state, done = zip(*transitions)

        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def downsize_average(data, window_size):
    """
  对列表数据进行分组平均。

  Args:
    data: 要进行平均的列表数据。
    window_size: 每多少个元素取一次平均。

  Returns:
    包含平均值的列表。返回值列表的长度将是原始列表长度的 1/window_size
  """
    averaged_data = []
    for i in range(0, len(data), window_size):
        average = sum(data[i:i + window_size]) / window_size
        averaged_data.append(average)
    return averaged_data


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, info, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def discrete_to_continuous(value, from_min, from_max, to_min, to_max):
    # 首先检查输入值是否在原始范围内
    if value < from_min or value > from_max:
        raise ValueError(f"Input value must be between {from_min} and {from_max}")

    # 计算原始范围和目标范围的跨度
    from_span = from_max - from_min
    to_span = to_max - to_min

    # 将值归一化到 0-1 范围
    scaled_value = (value - from_min) / from_span

    # 将归一化的值映射到新的范围
    return to_min + (scaled_value * to_span)


def continuous_to_discrete(value, from_min, from_max, to_min, to_max):
    if isinstance(value, np.ndarray):
        value = value.item()

    # 检查输入值是否在原始范围内
    if value < from_min or value > from_max:
        raise ValueError(f"Input value must be between {from_min} and {from_max}")

    # 计算原始范围和目标范围的跨度
    from_span = from_max - from_min
    to_span = to_max - to_min

    # 将值归一化到 0-1 范围
    scaled_value = (value - from_min) / from_span

    # 将归一化的值映射到新的范围，并四舍五入到最近的整数
    return round(to_min + (scaled_value * to_span))


def smooth(a, window_size):
    if window_size % 2 == 0:
        window_size += 1  # 确保窗口大小为奇数

    half_window = window_size // 2

    # 对原始数组进行填充，以处理边界情况
    padded = np.pad(a, (half_window, half_window), mode='edge')

    # 使用卷积来计算移动平均
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, window, mode='valid')

    return smoothed


def calculate_window_stats(y, window_size):
    """
    计算滑动窗口的均值和标准差，对边缘情况进行填充处理。

    Args:
    y (list or np.array): 输入的奖励列表
    window_size (int): 窗口大小

    Returns:
    tuple: 包含两个numpy数组，分别是均值(means)和标准差(stds)
    """
    y = np.array(y)
    n = len(y)

    # 填充数组，确保边缘元素也有足够的数据计算
    pad_width = window_size // 2
    padded_y = np.pad(y, (pad_width, pad_width), mode='edge')

    means = np.zeros(n)
    stds = np.zeros(n)

    for i in range(n):
        window = padded_y[i:i + window_size]
        means[i] = np.mean(window)
        stds[i] = np.std(window)

    return means, stds