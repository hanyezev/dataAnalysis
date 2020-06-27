import numpy as np

# fixme 1.创建等差数组

# arange()方法
# 指定 start、stop、以及step。arange和range一样，是左闭右开的区间。
arr_uniform0 = np.arange(1, 10, 1)
print(arr_uniform0)
# 也可以只传入一个参数，这种情况下默认start=0，step=1
arr_uniform1 = np.arange(10)
print(arr_uniform1)
arr_uniform2 = np.arange(1.2, 3.8, 0.3)
print(arr_uniform2)

# linspace()方法
# np.linspace(start, stop[, num=50[, endpoint=True[, retstep=False[, dtype=None]]]]])
# start、stop参数，和arange()中一致；
# num为待创建的数组中的元素的个数，默认为50
# endpoint=True，则为左闭右闭区间，默认为True；endpoint=False，则为左闭右开区间
# retstep用来控制返回值的形式。默认为False，返回数组；若为True，则返回由数组和步长组成的元祖
arr_uniform3 = np.linspace(1, 99, 11)
print(arr_uniform3)
arr_uniform4 = np.linspace(1, 99, 11, retstep=True)
print(arr_uniform4)
arr_uniform5 = np.linspace(1, 99, 11, endpoint=False)
print(arr_uniform5)

# fixme 2.创建等比数组

# geomspace()方法，创建指数等比数列
arr_geo0 = np.geomspace(2, 16, 4)
print(arr_geo0)

# logspace()方法，创建对数等比数列
# logspace（start, stop, num=50, endpoint=True, base=10.0, dtype=None）
# start：区间起始值为base的start次方
# stop：区间终止值为base的stop次方（是否取得到，需要设定参数endpoint）
# num：为待生成等比数列的长度。按照对数，即start和stop值进行等分。默认值为50
# endpoint：若为True（默认），则可以取到区间终止值，即左闭右闭区间，规则同上
arr_geo1 = np.logspace(1, 4, num=4, base=3)
print(arr_geo1)

# fixme 3.创建随机数数组

# 创建[0, 1)之间的均匀分布的随机数组
arr_rand0 = np.random.rand(3, 2)
print(arr_rand0)

# 创建[low, high)之间的均匀分布的随机数组
# numpy.random.uniform(low=0.0, high=1.0, size=None)
arr_rand1 = np.random.uniform(1, 10, (3, 2))
print(arr_rand1)

# 创建服从标准正态分布的数组（均值为0，方差为1）
arr_rand2 = np.random.randn(3, 2)
print(arr_rand2)

# 创建服从μ=loc，σ=scale的正态分布的数组
# numpy.random.normal(loc=0.0, scale=1.0, size=None)
# loc:指定均值 μ; scale:指定标准差 σ
arr_rand3 = np.random.normal(5, 10, (3, 2))
print(arr_rand3)

# 在指定区间[low, high)中离散均匀抽样的数组
# numpy.random.randint(low, high=None, size=None, dtype=np.int64)
# 有放回抽样6次
arr_rand4 = np.random.randint(1, 5, (3, 2))
print(arr_rand4)

# 对具体样本进行有放回或者无放回的抽样
# numpy.random.choice(a, size=None, replace=True, p=None)
# 从样本a中进行抽样，a可以为数组、列表或者整数，若为整数，表示[0,a)的离散抽样；
# replace为False，表示无放回抽样；replace为True，表示有放回抽样
# size为生成样本的大小
# p为给定数组中元素出现的概率
shoot_lst = np.random.choice(["命中", "未命中"], size=10, replace=True, p=[0.65, 0.35])
print(shoot_lst)
