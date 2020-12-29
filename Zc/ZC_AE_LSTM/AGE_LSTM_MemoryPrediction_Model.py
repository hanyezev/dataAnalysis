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
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from pytorchtools import EarlyStopping
import math
import os

start = time.time()

# 归一化到0~1
scaler = MinMaxScaler(feature_range=(0, 1))

def get_data_dir(dataname):
    father_dir = os.path.abspath('..')
    data_dir = father_dir+f'/{dataname}'
    return data_dir

def read_file(filename):
    temp = filename.split(".")
    if temp[1] == "xlsx" or temp[1] == "xls":
        return pd.read_excel(get_data_dir(filename), index_col=0)
    elif temp[1] == "csv":
        return pd.read_csv(get_data_dir(filename), index_col=0)
    else:
        return "文件不存在或格式不符"

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

def com_test(arr1, arr2):
    MSEs = []
    errors = []
    MAPEs = []
    num_temp = len(arr1)

    for i in range(num_temp):
        error = abs(arr1[i] - arr2[i])
        errors.append(error)
        MSEs.append(error ** 2)
        MAPEs.append((error / arr2[i]) * 100)

    MSE = np.mean(MSEs)
    MAPE = np.mean(MAPEs)

    return MSE, MAPE, MAPEs, errors

def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """
    y_true = y_true.reshape(-1).detach().cpu().numpy()
    y_pred = y_pred.reshape(-1).detach().cpu().numpy()
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

class AE(nn.Module):
    def __init__(self,input_size,hidden_size,hat_size,num_hour):
        super(AE, self).__init__()
        self.GRU_layer1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output_linear1 = nn.Linear(in_features=hidden_size, out_features=hat_size)

        self.GRU_layer2 = nn.GRU(input_size=int(hat_size/(num_hour/input_size)), hidden_size=hidden_size, batch_first=True)
        self.output_linear2 = nn.Linear(in_features=hidden_size, out_features=num_hour)
        self.hidden = None

    def forward(self, x):
        x, self.hidden = self.GRU_layer1(x)
        b1, s1, h1 = x.size()
        x = x[:, [s1-1], :]
        x = x.view(-1,h1)
        x = self.output_linear1(x)
        x_hat = x.view(b1, s1, -1)

        x, self.hidden = self.GRU_layer2(x_hat)
        b2, s2, h2 = x.size()
        x = x[:, [s2 - 1], :]
        x = x.view(-1, h2)
        x = self.output_linear2(x)
        x = x.view(b2, s2, -1)
        return x,x_hat

class lstm(nn.Module):
    def __init__(self, input_size=16, hidden_size=100, output_size=1, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = x[:, [13], :]
        b, s, h = x.size()
        x = x.view(-1, h)
        x = self.layer2(x)
        x = x.view(b, s, -1)
        return x

# 参数设置
filename = "服务器性能数据.xlsx"
KPI = "CPU平均负载"
num_hour = 336      # 历史数据个数
pred_h = 12     # 预测步数

device = torch.device("cuda")

# 1.读取data数据
df_original = read_file(filename)
# 2.在数据中建立深拷贝副本
df = df_original.copy(deep=True)

# 3.使用k-近邻法填补缺失值
df["主机CPU平均负载"] = knn_mean(df["主机CPU平均负载"], 24)

# 4.建立自回归预测矩阵
data = df["主机CPU平均负载"]
dataframe = pd.DataFrame()
for i in range(num_hour-1,0,-1):
    dataframe['t-'+str(i)] = data.shift(i)
dataframe['t'] = data.values
for i in range(1,pred_h+1):
    dataframe['t+'+str(i)] = data.shift(periods=-i, axis=0)
print(dataframe)
# 5.划分测试集和训练集
np.random.seed(113)
all_data = dataframe[num_hour:-pred_h]
all_data = shuffle(all_data)

var1 = int(len(all_data)*0.6)
var2 = int(len(all_data)*0.8)
train_data = all_data[0:var1]
train_truth = all_data.iloc[0:var1,-pred_h:]
validate_data = all_data[var1:var2]
validate_truth = all_data.iloc[var1:var2,-pred_h:]
test_data = all_data[var2:]
test_truth = all_data.iloc[var2:,-pred_h:]

max_value = max(train_data.max().values)
min_value = min(train_data.min().values)
ch = max_value - min_value

# 6.归一化
train_data_normalized = scaler.fit_transform(train_data.values.reshape(-1, 1))
train_data_normalized = train_data_normalized.reshape(-1,num_hour+pred_h)

validate_data_normalized = scaler.transform(validate_data.values.reshape(-1, 1))
validate_data_normalized = validate_data_normalized.reshape(-1,num_hour+pred_h)

test_data_normalized = scaler.transform(test_data.values.reshape(-1, 1))
test_data_normalized = test_data_normalized.reshape(-1,num_hour+pred_h)

# 7.转Tensor
train_X = torch.Tensor(train_data_normalized[:,0:-pred_h].reshape(-1, int(num_hour/24), 24)).to(device)
train_Y = torch.Tensor(train_data_normalized[:,-pred_h:].reshape(-1, 1, pred_h)).to(device)

validate_X = torch.Tensor(validate_data_normalized[:,0:-pred_h].reshape(-1, int(num_hour/24), 24)).to(device)
validate_Y = torch.Tensor(validate_data_normalized[:,-pred_h:].reshape(-1, 1, pred_h)).to(device)

test_X = torch.Tensor(test_data_normalized[:,0:-pred_h].reshape(-1, int(num_hour/24), 24)).to(device)
test_Y = torch.Tensor(test_data_normalized[:,-pred_h:].reshape(-1, 1, pred_h)).to(device)

# 8.建模以及模型参数
model_AE = AE(24,256,hat_size=84,num_hour=num_hour).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_AE.parameters(), lr=1e-2)
epoch_n = 3000

# 9.开始训练
ep_AE = []
losses_AE = []
lr_list_AE = []
for e in range(1, epoch_n + 1):
    var_x = Variable(train_X).to(device)
    #     var_y = Variable(train_Y)
    # 前向传播
    out, out_hat = model_AE(var_x)
    loss = criterion(out, var_x)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 10 == 0:  # 每 10 次输出结果
        print('Epoch: {}, Loss: {:.8f}'.format(e, loss.item()))
    ep_AE.append(e)
    losses_AE.append(loss.item())

# 10.保存AE模型
torch.save(model_AE, 'AE.pkl')
# 11.加载AE模型
model_AE2 = torch.load('AE.pkl')

# 12.生成隐变量
outputs, outputs_hat = model_AE2(train_X)
train_X2 = outputs_hat.to(device)
outputs, outputs_hat = model_AE2(validate_X)
validate_X2 = outputs_hat.to(device)
outputs, outputs_hat = model_AE2(test_X)
test_X2 = outputs_hat.to(device)

# 13.建立LSTM模型
model_lstm = lstm(6, 64, 12, 2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=1e-2)
epoch_n = 1000
patience = 20
early_stopping = EarlyStopping(patience, verbose=True)

# 14.开始训练
ep_Ls = []
losses_Ls = []
lr_list_LS = []
for e in range(1, epoch_n + 1):
    var_x = Variable(train_X2).to(device)
    var_y = Variable(train_Y).to(device)
    # 前向传播
    out = model_lstm(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    valid_output = model_lstm(validate_X2)
    valid_loss = criterion(valid_output, validate_Y)

    if e % 10 == 0:  # 每 10 次输出结果
        print('Epoch: {}, Loss: {:.8f}, VA_Loss: {:.8f}'.format(e, loss.item(), valid_loss.item()))
    ep_Ls.append(e)
    losses_Ls.append(loss.item())

    early_stopping(valid_loss, model_lstm)

    # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break
#     break
#     if (e+1)%120 == 0:
#         for p in optimizer.param_groups:
#             p['lr'] *= 0.1
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

# 15.结束并保存
print('Finished Training')
end = time.time()
time = end - start
print(f'运行时长为:{int(time)}s')
torch.save(model_lstm, 'LSTM.pkl')
model_lstm2 = torch.load('LSTM.pkl')

# 16.验证
outputs = model_lstm2(test_X2)
predict = outputs*(max_value-min_value) + min_value
predict = predict.to(device)
truth = torch.Tensor(test_truth.values.reshape(-1, 1, pred_h)).to(device)
print(f"测试集整体MSE: {criterion(predict, truth)}")
print(f"测试集整体MAPE: {mape(truth, predict)}")

outputs = model_lstm2(train_X2)
predict2 = outputs*(max_value-min_value) + min_value
predict2 = predict2.to(device)
truth2 = torch.Tensor(train_truth.values.reshape(-1, 1, pred_h)).to(device)
print(f"训练集整体MSE: {criterion(predict2, truth2)}")
print(f"训练集整体MAPE: {mape(truth2, predict2)}")
