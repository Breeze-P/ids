# 背景：给系统呈现一系列数据，目标是系统检测出攻击数据，但是不能每次都提示是攻击数据，这样认为是检测失败
# 观察点：数据集给定的特征向量，最高是1最低是0
# 决策：0 什么也不做，通过，1，报警，不通过
# 奖励：检测到攻击并正确+1，通过攻击-1，检测到正常数据并通过不给分，错将正常数据当作攻击-1
# 一次训练结束的条件：训练1000次，或者通过了一次攻击
import gym
import pandas as pd
import numpy as np

class IDS_Env(gym.Env):
    def __init__(self, data):
        super().__init__()
        self.data = data

        # 设置决策和观察空间
        self.action_space = gym.spaces.Discrete(2)  # either 0 (NORMAL) or 1 (ATTACK)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(data.shape[1] - 1,), dtype=np.float64
        )

        # 设置奖励
        # (true_label, action) -> reward
        self.rewards = {(0, 0): 0, (0, 1): -1, (1, 1): 1, (1, 0): -1}

        # 设置训练次数
        self.max_steps = 1000
        self.steps_beyond_done = None

    # 用于重启下一次训练
    def reset(self):
        self.steps_beyond_done = None
        self.current_step = 0

        # 开始时的决策是随机的
        self.i = np.random.randint(0, self.data.shape[0])

        # 抽取观察空间
        self.obs = self.data.iloc[self.i]

        # 抽取正确的决策
        self.label = int(self.obs.pop("label"))

        return self.obs.values

    # 用于下一次决策
    def step(self, action):
        # 是否成功
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # 判断是否结束此次训练
        info = {}
        done = False
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            info["reason_to_stop"] = "episode_limit"

        if self.label == 1 and action == 0:
            done = True
            info["reason_to_stop"] = "attack_missed"

        # 分析奖励
        reward = self.rewards[(self.label, action)]

        # 计算下一次状态
        if not done:
            self.i += 1
            if self.i >= self.data.shape[0]:
                self.i = 0

            self.obs = self.data.iloc[self.i]
            self.label = int(self.obs.pop("label"))

        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            # 如果直接退出训练，则报警
            if self.steps_beyond_done == 0:
                gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0

        # gym这个接口需要实现的函数
        # 1. 新的观察空间,
        # 2. 奖励,
        # 3. 结束训练,
        # 4. 可选.
        return self.obs.values, reward, done, info

# 搭建训练环境
train_data = pd.read_csv("processed_data/train.csv")
train_data.head()

env = IDS_Env(train_data)

from stable_baselines.common.env_checker import check_env

check_env(env, warn=True)

# 测试随机的数据
all_rewards = []
all_lengths = []
for i in range(10):
    print("-" * 50)
    print(f"Episode {i+1:02}")
    print("-" * 50)

    ep_reward = 0
    obs = env.reset()
    done = False
    while not done:
        t_label = env.label
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)  # perform random action

        print(
            f"true_label: {t_label}, action: {action}, reward: {rew:>2}, done: {done}, {info}"
        )
        ep_reward += rew
    print(f"\n>>Episode reward: {ep_reward}")
    print(f">>Episode length: {env.current_step}")
    print("-" * 50)
    all_rewards.append(ep_reward)
    all_lengths.append(env.current_step)

print("\nRewards", all_rewards)
print(f"Mean Episodes Reward: {np.mean(all_rewards)}")
print("\nLengths", all_lengths)
print(f"Mean Episodes Length: {np.mean(all_lengths)}")

# 测试全是正常数据
all_rewards = []
all_lengths = []
for i in range(10):
    print("-" * 50)
    print(f"Episode {i+1:02}")
    print("-" * 50)

    ep_reward = 0
    obs = env.reset()
    done = False
    while not done:
        t_label = env.label
        action = 0
        obs, rew, done, info = env.step(action)  # always normal traffic

        print(
            f"true_label: {t_label}, action: {action}, reward: {rew:>2}, done: {done}, {info}"
        )
        ep_reward += rew
    print(f"\n>>Episode reward: {ep_reward}")
    print(f">>Episode length: {env.current_step}")
    print("-" * 50)
    all_rewards.append(ep_reward)
    all_lengths.append(env.current_step)

print("\nRewards", all_rewards)
print(f"Mean Episodes Reward: {np.mean(all_rewards)}")
print("\nLengths", all_lengths)
print(f"Mean Episodes Length: {np.mean(all_lengths)}")

# 测试全是攻击的数据
all_rewards = []
all_lengths = []

for i in range(10):
    print("-" * 50)
    print(f"Episode {i+1:02}")
    print("-" * 50)

    ep_reward = 0
    obs = env.reset()
    done = False
    while not done:
        t_label = env.label
        action = 1
        obs, rew, done, info = env.step(action)  # always attack

        print(
            f"true_label: {t_label}, action: {action}, reward: {rew:>2}, done: {done}, {info}"
        )
        ep_reward += rew
    print(f"\n>>Episode reward: {ep_reward}")
    print(f"\n>>Episode length: {env.current_step}")
    print("-" * 50)
    all_rewards.append(ep_reward)
    all_lengths.append(env.current_step)

print("\nRewards", all_rewards)
print(f"Mean Episodes Reward: {np.mean(all_rewards)}")
print("\nLengths", all_lengths)
print(f"Mean Episodes Length: {np.mean(all_lengths)}")