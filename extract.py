import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.tsa.stattools as ts


# 读取服务器数据,返回指定列
def readFile(filepath, num):
    df = pd.read_excel(filepath)
    # print(df.head())
    col_num = num
    X = df.iloc[:col_num, 1]
    Y1 = df.iloc[:col_num, 3]
    Y2 = df.iloc[:col_num, 4]
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
    return old, new


if __name__ == "__main__":
    path = r'C:/Users/16526/Desktop/电网项目/服务器性能数据.xlsx'
    X_time, Y1_ram, Y2_cpu = readFile(path, 720)
    old, new = d1_ADF(Y1_ram)
    print('差分前:' + str(old))
    print('差分后:' + str(new))