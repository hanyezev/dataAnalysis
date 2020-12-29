import pandas as pd
import numpy as np
import torch
from dateutil.parser import parse
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from datetime import datetime
import time
from scipy.interpolate import interp1d
import statsmodels.api as sm
import itertools
import seaborn as sns
import math
from numpy import np

# Draw Plot-----绘图函数
def plot_df(df, x, y, title="", xlabel='Index', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

# knn_mean-----k近邻插值法
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

seed = 2
print('GPU:', torch.cuda.is_available())
torch.manual_seed(seed) # 为CPU设置种子用于生成随机数，以使得结果是确定的

df = pd.read_excel(r"F:\实验室\电网项目\服务器性能数据.xlsx", index_col=0)





if __name__ == "main":
    print("1")
