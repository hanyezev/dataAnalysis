import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
# from pytorchtools import EarlyStopping
import pdb
# # 设置中文和负号正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.read_excel(r"/root/test2/服务器性能数据.xlsx", index_col=0)
df2 = df.copy(deep=True)

# Draw Plot-----绘图函数
def plot_df(df, x, y, title="", xlabel='Index', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

# k近邻法
def knn_mean(ts, n):
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            n_by_2 = np.ceil(n / 2)
            lower = np.max([0, int(i - n_by_2)])
            upper = np.min([len(ts) + 1, int(i + n_by_2)])
            ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
            out[i] = np.nanmean(ts_near)
    return out
# 使用k-近邻法填补缺失值
df2["主机CPU平均负载"] = knn_mean(df2["主机CPU平均负载"], 24)
# df2["主机CPU平均负载"].astype(int)
# plot_df(df2, x=df2.index, y= df2["主机CPU平均负载"], title='')

device = torch.device("cuda")

best_score = 1
index = 0
score_dict = {}
for num_hour in [24,48,72,168,336]:# 3
    for learning_rate in [1e-2,1e-3,1e-4]:#2
        for epoch_n in [2000]: #3
            # 25
            for h_size in range(2, 20, 2):
                data_1 = df2["主机CPU平均负载"]
                dataframe_1 = pd.DataFrame()
                for i in range(num_hour - 1, 0, -1):
                    dataframe_1['t-' + str(i)] = data_1.shift(i)
                dataframe_1['t'] = data_1.values
                for i in range(1, 13):
                    dataframe_1['t+' + str(i)] = data_1.shift(periods=-i, axis=0)

                np.random.seed(2)
                all_data = dataframe_1[num_hour - 1:-12]

                var1 = int(len(all_data) * 0.6)
                var2 = int(len(all_data) * 0.8)
                train_data = all_data.iloc[0:var1]
                train_truth = all_data.iloc[0:var1, -12:]
                validate_data = all_data.iloc[var1:var2]
                validate_truth = all_data.iloc[var1:var2, -12:]
                test_data = all_data.iloc[var2:]
                test_truth = all_data.iloc[var2:, -12:]

                max_value = max(train_data.max().values)
                min_value = min(train_data.min().values)
                ch = max_value - min_value

                train_data_normalized = scaler.fit_transform(train_data.values.reshape(-1, 1))
                train_data_normalized = train_data_normalized.reshape(-1, num_hour + 12)
                # print(train_data_normalized.shape)
                validate_data_normalized = scaler.transform(validate_data.values.reshape(-1, 1))
                validate_data_normalized = validate_data_normalized.reshape(-1, num_hour + 12)
                # print(validate_data_normalized.shape)
                test_data_normalized = scaler.transform(test_data.values.reshape(-1, 1))
                test_data_normalized = test_data_normalized.reshape(-1, num_hour + 12)
                # print(test_data_normalized.shape)

                train_X = torch.Tensor(train_data_normalized[:, 0:-12].reshape(-1, 1, num_hour))
                train_Y = torch.Tensor(train_data_normalized[:, -12:].reshape(-1, 1, 12))
                # print(train_X.shape)
                validate_X = torch.Tensor(validate_data_normalized[:, 0:-12].reshape(-1, 1, num_hour))
                validate_Y = torch.Tensor(validate_data_normalized[:, -12:].reshape(-1, 1, 12))
                test_X = torch.Tensor(test_data_normalized[:, 0:-12].reshape(-1, 1, num_hour))
                test_Y = torch.Tensor(test_data_normalized[:, -12:].reshape(-1, 1, 12))

                # VAE模型
                start = time.time()


                class VAE(nn.Module):
                    def __init__(self):
                        super(VAE, self).__init__()
                        self.encoder = nn.Sequential(
                            nn.Linear(num_hour, 256),
                            nn.ReLU(),
                            nn.Linear(256, 64),
                            nn.ReLU(),
                            nn.Linear(64, h_size),
                            nn.ReLU()
                        )
                        self.decoder = nn.Sequential(
                            nn.Linear(int(h_size / 2), 64),
                            nn.ReLU(),
                            nn.Linear(64, 256),
                            nn.ReLU(),
                            nn.Linear(256, num_hour),
                            nn.Sigmoid()
                        )

                        self.criteon = nn.MSELoss()

                    def forward(self, x):
                        """
                        :param x: [b, 1, 336]
                        :return:
                        """
                        batchsz = x.size(0)
                        # flatten
                        x = x.view(batchsz, num_hour)
                        # encoder
                        h_ = self.encoder(x)
                        # [b, 20] => [b, 10] and [b, 10]
                        mu, sigma = h_.chunk(2, dim=1)
                        # reparametrize trick, epison~N(0, 1)
                        h = mu + sigma * torch.randn_like(sigma)  # 914,10
                        # decoder
                        x_hat = self.decoder(h)
                        # reshape
                        x_hat = x_hat.view(batchsz, 1, num_hour)

                        kld = 0.5 * torch.sum(
                            torch.pow(mu, 2) +
                            torch.pow(sigma, 2) -
                            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                        ) / (batchsz * num_hour)
                        return x_hat, kld


                model_VAE = VAE().to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model_VAE.parameters(), lr=learning_rate)

                ep = []
                losses = []
                lr_list = []
                for e in range(1, epoch_n + 1):
                    var_x = Variable(train_X).to(device)
                    #     var_y = Variable(train_Y)
                    # 前向传播
                    x_hat, kld = model_VAE(var_x)
                    loss = criterion(x_hat, var_x)

                    if kld is not None:
                        elbo = - loss - 1.0 * kld
                        loss = - elbo

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if e % 1000 == 0:  # 每 1000 次输出结果
                        print('Epoch: {}, Loss: {:.8f},kld: {:.8f}'.format(e, loss.item(), kld.item()))
                    ep.append(e)
                    losses.append(loss.item())

                if losses[-1] < best_score:
                    best_score = losses[-1]
                    best_parameters = {'num_hour': num_hour, "learning_rate": learning_rate,
                                       "epoch_n": epoch_n, "h_size": h_size}
                index += 1
                print(f"已搜索{index}次")
                score_dict[index] = {"best_score": best_score, 'num_hour': num_hour,
                                     "learning_rate": learning_rate,"epoch_n": epoch_n,
                                     "h_size": h_size}

print(best_score)
print(best_parameters)
print(score_dict)



# # 参数列表
# # 平滑前,滞后阶数
# num_hour = 336
# learning_rate = 1e-3
# epoch_n = 5000
# h_size = 16
print('Finished Training')

plot_df(df, x=ep, y= losses, title='LOSS')