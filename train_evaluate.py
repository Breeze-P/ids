import gym
import pandas as pd
import numpy as np

# 这部分和之前一样
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

# 开始训练
train_data = pd.read_csv("processed_data/train.csv")

from stable_baselines.common.vec_env import DummyVecEnv

n_envs = 16  # 这个数字比较合适
env = DummyVecEnv([lambda: IDS_Env(train_data)] * n_envs)
from stable_baselines import PPO2

from stable_baselines.common.policies import FeedForwardPolicy


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args,
            **kwargs,
            net_arch=[128, 64, 32],  # 三层网络
            act_fun=tf.nn.relu,
            feature_extraction="mlp"
        )

model = PPO2(
    CustomPolicy,
    env,
    gamma=0.9,
    n_steps=512,
    ent_coef=1e-05,
    learning_rate=lambda progress: progress
    * 0.0021,  # progress decreases from 1 to 0 -> lr decreasesb from 0.0021 to 0
    vf_coef=0.6,
    max_grad_norm=0.8,
    lam=0.8,
    nminibatches=16,
    noptepochs=55,
    cliprange=0.2,
    verbose=0,
    tensorboard_log="log_25",
)

from stable_baselines.common.callbacks import BaseCallback
from sklearn.metrics import accuracy_score, f1_score
import swifter

class AccF1Callback(BaseCallback):
    def __init__(self, train, val, eval_freq):
        super().__init__()
        self.train_data = train
        self.val_data = val
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:

        # 每次step()以后回调
        # 如果这个函数False，说明训练提前结束

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()

            # 计算训练数据
            predicted = self.train_data.drop(columns=["label"]).swifter.apply(
                lambda x: self.model.predict(x, deterministic=True)[0], axis=1
            )
            accuracy = accuracy_score(self.train_data["label"], predicted)
            f1 = f1_score(self.train_data["label"], predicted)

            print("-" * 60)
            print(f"timesteps: {self.num_timesteps}")
            print(f"Training   >>> accuracy: {accuracy:.4f}, f1-score: {f1:.4f}")

            # 计算验证数据
            predicted = self.val_data.drop(columns=["label"]).swifter.apply(
                lambda x: self.model.predict(x, deterministic=True)[0], axis=1
            )
            accuracy = accuracy_score(self.val_data["label"], predicted)
            f1 = f1_score(self.val_data["label"], predicted)
            print(f"Validation >>> accuracy: {accuracy:.4f}, f1-score: {f1:.4f}")
            print("-" * 60)

        return True

val_data = pd.read_csv("processed_data/val.csv")
eval_callback = AccF1Callback(train_data, val_data, eval_freq=20000 // n_envs)
model.learn(6000000, callback=eval_callback)
model.save('ids_drl_model')

# 验证评估，这里是和之前define_environment里那个随机的对照结果做比较
env = env.envs[0]
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
        action = model.predict(obs, deterministic=True)[0]
        obs, rew, done, info = env.step(action)

        print(
            f"true_label: {t_label}, action: {action}, reward: {rew:>2}, done: {done}, {info}"
        )
        ep_reward += rew
    print(f"\n>>Episode reward: {ep_reward}")
    print("-" * 50)
    all_rewards.append(ep_reward)
    all_lengths.append(env.current_step)

print("\nRewards", all_rewards)
print(f"Mean Episodes Reward: {np.mean(all_rewards)}")
print("\nLengths", all_lengths)
print(f"Mean Episodes Length: {np.mean(all_lengths)}")

from sklearn.metrics import (
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns

# 在训练集的结果
pred = train_data.drop(columns=["label"]).swifter.apply(
    lambda x: model.predict(x, deterministic=True)[0], axis=1
)

print("accuarcy:", accuracy_score(train_data["label"], pred))
print("recall:", recall_score(train_data["label"], pred))
print("precision:", precision_score(train_data["label"], pred))
print("f1-score:", f1_score(train_data["label"], pred))
print("balanced accuarcy:", balanced_accuracy_score(train_data["label"], pred))

print("\nConfusion matrix:")
cm = confusion_matrix(train_data["label"], pred)
df_cm = pd.DataFrame(cm, columns=["normal", "attack"], index=["normal", "attack"])
df_cm.index.name = "Actual"
df_cm.columns.name = "Predicted"

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, cmap="Blues", annot=True, fmt="g")

# 在验证集的结果
pred = val_data.drop(columns=["label"]).swifter.apply(
    lambda x: model.predict(x, deterministic=True)[0], axis=1
)

print("accuarcy:", accuracy_score(val_data["label"], pred))
print("recall:", recall_score(val_data["label"], pred))
print("precision:", precision_score(val_data["label"], pred))
print("f1-score:", f1_score(val_data["label"], pred))
print("balanced accuarcy:", balanced_accuracy_score(val_data["label"], pred))

print("\nConfusion matrix:")
cm = confusion_matrix(val_data["label"], pred)
df_cm = pd.DataFrame(cm, columns=["normal", "attack"], index=["normal", "attack"])
df_cm.index.name = "Actual"
df_cm.columns.name = "Predicted"

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, cmap="Blues", annot=True, fmt="g")

# 测试集下的结果
test_data = pd.read_csv("processed_data/test.csv")
pred = test_data.drop(columns=["label"]).swifter.apply(
    lambda x: model.predict(x, deterministic=True)[0], axis=1
)

print("accuarcy:", accuracy_score(test_data["label"], pred))
print("recall:", recall_score(test_data["label"], pred))
print("precision:", precision_score(test_data["label"], pred))
print("f1-score:", f1_score(test_data["label"], pred))
print("balanced accuarcy:", balanced_accuracy_score(test_data["label"], pred))

print("\nConfusion matrix:")
cm = confusion_matrix(test_data["label"], pred)
df_cm = pd.DataFrame(cm, columns=["normal", "attack"], index=["normal", "attack"])
df_cm.index.name = "Actual"
df_cm.columns.name = "Predicted"

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, cmap="Blues", annot=True, fmt="g")








# ROC和PR曲线评估
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
scores = test_data.drop(columns=["label"]).swifter.apply(
    lambda x: model.action_probability(x)[1], axis=1
)
# ROC 结果
fpr, tpr, _ = roc_curve(test_data["label"], scores)
roc_auc = auc(fpr, tpr)

sns.set_style("white")
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
plt.plot(fpr, tpr, color="darkorange", lw=3, label="ROC-AUC = %0.2f" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Curve on test data")
plt.legend(loc="lower right")

ax.spines["bottom"].set_color("#828282")
ax.spines["left"].set_color("#828282")
ax.spines["right"].set_color("#828282")
ax.spines["top"].set_color("#828282")

ax.yaxis.label.set_color("#828282")
ax.xaxis.label.set_color("#828282")
plt.show()


# PR 结果
precisions, recalls, thresholds = precision_recall_curve(test_data["label"], scores)
average_prec = average_precision_score(test_data["label"], scores)

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
plt.step(
    recalls,
    precisions,
    where="post",
    lw=3,
    color="darkorange",
    label="AUC-PR={0:0.2f}".format(average_prec),
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR-Curve on test data")

plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend()

ax.spines["bottom"].set_color("#828282")
ax.spines["left"].set_color("#828282")
ax.spines["right"].set_color("#828282")
ax.spines["top"].set_color("#828282")
ax.yaxis.label.set_color("#828282")
ax.xaxis.label.set_color("#828282")

plt.show()