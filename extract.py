import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


# 读取服务器数据,返回指定列
def readFile(filepath, num):
    df = pd.read_excel(filepath)
    # print(df.head())
    col_num = num
    X = df.iloc[:col_num, 1]
    Y1 = df.iloc[:col_num, 3]
    Y2 = df.iloc[:col_num, 4]
    # Y1[Y1 == 0] = np.nan
    # Y2[Y2 == 0] = np.nan
    return X, Y1, Y2


# 测试序列种类
def d1_ADF(series):
    old = ts.adfuller(series)

    _series = pd.Series(data=series)  # 获取data过程省略
    diff1 = dta = _series.diff(1)[1:]  # dta[0] is nan
    diff1.plot()
    # plt.savefig('./diff_1.jpg')
    plt.show()
    new = ts.adfuller(dta)
    return old, new, dta


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


if __name__ == "__main__":
    path = r'C:/Users/16526/Desktop/电网项目/服务器性能数据.xlsx'
    df = pd.read_excel(path, encoding="utf-8", index_col=0, header=0)
    plt.plot(range(72), df["内存负载"][:72])
    plt.show()
    # 将内存负载列的0值转化为Nan
    print(df.isnull().any(axis=0))
    df["内存负载"] = df["内存负载"].replace(0, np.nan)
    print(df.isnull().any(axis=0))

    # 使用k-近邻法填补缺失值
    df["内存负载"] = knn_mean(df["内存负载"], 24)

    # ADF检验,然后一阶差分
    # old, new, d1_series = d1_ADF(df["内存负载"])
    # print('Y1_ram(无缺失值)差分前:' + str(old))
    # print('Y1_ram(无缺失值)差分后:' + str(new))

    # 利用ACF和PACF判断模型阶数
    # plot_acf(d1_series, lags=40)  # 延迟数
    # plot_pacf(d1_series, lags=40)
    # plt.show()

    # plt.plot(df["日期"], df["内存负载"])
    # plt.show()
