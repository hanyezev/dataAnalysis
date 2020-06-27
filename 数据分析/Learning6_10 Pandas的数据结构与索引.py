import pandas as pd
import numpy as np

# fixme 1.Series数据结构与索引
'''
Numpy是基于数组格式构建的一个数组运算工具，而Pandas是基于Numpy构建的结构化数据处理工具。
Pandas 常用的数据结构有两种：Series和 DataFrame。
其中 Series 是一个带有名称和索引的一维数组，而 DataFrame则是用来表示多维的数组结构。
'''
# Series数据结构
stock_seri = pd.Series([3.5, 7.2, 12.6, 4.7, 8.2],
                       index=["300773", "600751", "300405", "002937", "601615"],
                       name="Stock")
print(stock_seri)
print(type(stock_seri))
print(stock_seri.name)
print(stock_seri.index)
print(stock_seri.values)

# Series的常用操作
# Series的索引
print(stock_seri["300773"])
print(stock_seri[["002937", "300773", "601615"]])
print(stock_seri[[True, False, True, False, True]])
print(stock_seri["300773":"300405"])  # 注意:pandas是左闭右闭的区间

# Numpy常用的运算，在Series里也是适用的。
print(stock_seri[stock_seri > 8])
print(stock_seri * 2)
print(np.average(stock_seri))

# 查看Series数据集中是否有空值
print(stock_seri.isnull())
print(stock_seri.notnull())

# 修改index
stock_seri.index = ["拉卡拉", "海航科技", "科隆股份", "兴瑞科技", "明阳智能"]
# 更新兴瑞科技的股票价格
stock_seri["兴瑞科技"] = 6.8

# 字典来创建Series
stock_price_dict = {
    "2012": 1,
    "2013": 3
}
pd.Series(stock_price_dict)

# fixme 2.DataFrame数据结构与索引
'''
DataFrame是Series在水平方向的扩展，是一个多列表格型的数据结构
'''
frame0 = pd.DataFrame(np.arange(6).reshape(2, 3),
                      index=[2000, 2001], columns=["A", "B", "C"])
print(frame0)
# 利用字典格式，生成DataFrame数据结构
data = {"A": [0, 3], "B": [1, 4], "C": [2, 5]}
temp = pd.DataFrame(data, index=[2000, 2001])
print(temp)
print(temp["A"])
# 同时对多列索引，需要以列表的形式传入序列
print(temp[["A", "B", "C"]])

# 筛选A值为0的行
print(frame0[frame0["A"] == 0])
# 乘法运算
print(frame0 * 2)

# fixme 3.DataFrame索引的索引
'''
通过两种方式进行索引，分别是标签索引和位置索引
'''
# a. loc 标签索引
# Pandas自带的loc方法，通过具体的标签名字来索引行或者列。
# 选择索引为2000的行
print(frame0.loc[2000])
# 选择“A”列
print(frame0.loc[:, "A"])
# 选择索引为2001，BC列的元素
print(frame0.loc[2001, "B":"C"])  # 左闭右闭区间

# b. iloc函数
# 通过位置，选择索引为2000的行
print(frame0.iloc[0])  # 第0行
# 通过位置，选择“A”列
print(frame0.iloc[:, 0])  # 第0列

# c. 索引对象初窥
# index对象可以像列表一样被灵活切片
print(frame0.index[:1])
print(frame0.reindex([2000, 2001, 2002]))
print(frame0.reindex([2000, 2001, 2002], fill_value=20))
# 列索引的基本格式和行索引一直，也是Pandas自带的Index格式
print(frame0.columns)

# 修改列名，有2种方法
# 方法一：利用rename函数
print(frame0.rename(columns = {"A": "col_01", "B": "col_02", "C": "col_03"}))
# 方法二：重新指定
frame0.columns = ["col_04", "col_05", "col_06"]
print(frame0)
