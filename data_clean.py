import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)

dataset_folder = "../dataset"

# 获取所有文件的文件名
files = os.listdir(dataset_folder)

# 读数据集
df = pd.read_csv(os.path.join(dataset_folder, files[0]))
for file in os.listdir(dataset_folder)[1:]:
    print(file)
    # 从多个文件中，分别读取每个文件
    d = pd.read_csv(os.path.join(dataset_folder, file))
    df = pd.concat([df, d], axis=0)

# 释放文件内存
del d

# 给文件重置主键
df.reset_index(drop=True, inplace=True)

# 所有数据的字段信息
df.head()

# 将原来字段都是有空格的，替换成下划线
df.columns = df.columns.map(lambda x: "_".join(x.lower().strip().split(" ")))
# 删除目标端口信息
df.drop(columns=["destination_port"], inplace=True)
# 删除和fwd_header_length重复的字段
df.drop(columns=["fwd_header_length.1"], inplace=True)
# 删除为零的列
stats = df.describe()
zero_columns = stats.loc[:, (stats.loc["min"] == 0) & ((stats.loc["max"] == 0))]
# 删除值为NaN和无限的列
df["flow_bytes/s"].replace(np.inf, np.nan, inplace=True)
df["flow_packets/s"].replace(np.inf, np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)

# 分析数据集各种label占比
counts = df["label"].value_counts()
print(f"We have {len(counts.index)} different class:\n{counts.index.values}")
print(f"\nTheir distribution is as follows:\n{counts}")
print(f"\nAnd with percentages is:\n{counts / df.shape[0]}")
plt.figure(figsize=(14, 6))
sns.barplot(y=counts.index, x=counts.values, orient="h", palette="Paired")

# 只分成异常和正常包的情况
df["label"] = df["label"].map(
    lambda x: 0 if x == "BENIGN" else 1
)  # map labels to binary, 'BENIGN' <- 0  and 'Attack' <- 1

bi_counts = df["label"].value_counts()
print(f"Counts:\n{bi_counts}")
print(f"\nPercentage:\n{bi_counts/ df.shape[0]}")

plt.figure(figsize=(7, 7))
sns.barplot(x=bi_counts.index, y=bi_counts.values, orient="v", palette="Paired")

# 选择需要的特征向量
features = [
    "bwd_packet_length_min",
    "subflow_fwd_bytes",
    "total_length_of_fwd_packets",
    "fwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_iat_min",
    "fwd_iat_min",
    "flow_iat_mean",
    "flow_duration",
    "flow_iat_std",
    "active_min",
    "active_mean",
    "bwd_iat_mean",
    "fwd_iat_mean",
    "init_win_bytes_forward",
    "ack_flag_count",
    "fwd_psh_flags",
    "syn_flag_count",
    "fwd_packets/s",
    "init_win_bytes_backward",
    "bwd_packets/s",
    "psh_flag_count",
    "packet_length_mean",
]

# 训练集，验证集，测试集分割
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df["label"], test_size=0.2, shuffle=True)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

# 用自己和最小值的差比最大值和最小值的差，改变值的范围并保存

scaler = MinMaxScaler().fit(X_train.iloc[:, :])
X_train.iloc[:, :] = scaler.transform(X_train.iloc[:, :])
X_val.iloc[:, :] = scaler.transform(X_val.iloc[:, :])
X_test.iloc[:, :] = scaler.transform(X_test.iloc[:, :])

pd.concat([X_train, y_train], axis=1).to_csv("processed_data/train.csv", index=None)
pd.concat([X_test, y_test], axis=1).to_csv("processed_data/test.csv", index=None)
pd.concat([X_val, y_val], axis=1).to_csv("processed_data/val.csv", index=None)