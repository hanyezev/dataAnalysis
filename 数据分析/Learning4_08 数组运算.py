import numpy as np
from numpy.linalg import inv
from numpy.linalg import solve

# fixme 1.数组与标量之间的运算
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr + 5)
print(arr * 2)
print(1 / arr)
print(arr ** 2)

# fixme 2.数组的通用函数运算

# 能够对数组中的每个元素进行微操，也就是元素级的函数运算
# 对应位置相加减,非矩阵运算
print(arr - arr)
print(arr * arr)
# add, subtract, multiply, divide	算术四则运算，分别对应加、减、乘、除
print(np.multiply(arr, arr))

# 一元函数
arr_rnd = np.random.normal(5, 10, (3, 4))
# 对数组进行四舍五入运算
arr_temp = np.rint(arr_rnd)
print(arr_temp)

# todo 1.常用的一元函数列举如下，供大家查阅：
# abs，fabs	计算绝对值，对于非负数值，可以使用更快的fass
# sqrt，square，exp	求个元素的平方根、平方、指数ex
# log，log10，log2，log1p	分别计算自然对数（底数为e）、底数为10的log、底数为2的log、log(1+x)
# sign	计算各元素的正负号：1（整数）、0（零）、-1（负数）
# ceil	计算各元素的、大于等于该值的最小整数
# floor	计算各元素的、大于等于该值的最大整数
# rint	将各元素值四舍五入到最接近的整数，并保留dtype
# modf	把数组的小数和整数部分以两个独立的数组分别返回
# isnan	判断各元素是否为空NaN，返回布尔型
# cos，cosh，sin，sinh，tan，tanh	普通型和双曲型三角函数

# 二元函数
x = np.random.normal(5, 10, (3, 1))
y = np.random.normal(5, 10, (3, 1))
print(x)
print(y)
# 计算，比较元素级的最大值
temp = np.maximum(x, y)
print(temp)
# 计算，执行元素级的比较
temp = np.greater(x, y)
print(temp)

# todo 2.常用的二元函数列举如下，供大家查阅：
'''
但是在使用之前，大家千万要注意数组中是否有空值，
空值的存在可能会导致运算结果错误甚至是报错。判断
数组是否存在空值，需要使用isnan函数。
'''
# maximum，fmax	计算元素级的最大值，fmax自动忽略空值NaN
# minimum，fmin	计算元素级的最小值，fmin自动忽略空值NaN
# greater，greater_equal	执行元素级的比较，生产布尔型数组。效果相当于>，≥
# less，less_equal	执行元素级的比较，生产布尔型数组。效果相当于＜，≤
# equal，not_equal	执行元素级的比较，生产布尔型数组。效果相当于==，!=
# logical_and，logical_or，logic_xor	执行元素级的逻辑运算，相当于执行运算符&、|、^

# fixme 3.数组的线性代数运算

# 矩阵的乘法，输入的2个数组的维度需要满足矩阵乘法的要求，否则会报错；
# arr.T表示对arr数组进行转置
# np.dot表示对输入的两个数组进行矩阵乘法运算
arr = np.random.randn(3, 1)
temp = np.dot(arr, arr.T)
print(temp)

# numpy.linalg工具
# 封装了一组标准的矩阵分解运算以及诸如逆运算、行列式等功能

# 利用inv函数，求解矩阵的逆矩阵（注意：矩阵可变，首先必须是方阵）
arr_lg = np.array([[0, 1, 2], [1, 0, 3], [4, -3, 8]])
arr_inv = inv(arr_lg)
print(arr_inv)
print(np.dot(arr_lg, arr_inv))

# numpy.linalg中的函数solve可以求解形如 Ax = b 的线性方程组，其中 A 为矩阵，b 为一维数组，x 是未知变量。
A = np.array([[1, -2, 1], [0, 2, -8], [-4, 5, 9]])
b = np.array([0, 8, -9])
X = solve(A, b)
print(np.equal(np.dot(A, X), b))

# todo 3.numpy.linalg中还封装了一些其它函数，这里就不一一列举了，大家可以参考下表，根据需要选择合适的函数：
# diag	以一维数组的形式返回方阵的对角线（或非对角线）元素，或将一维数组转换为方阵
# trace	计算对角线元素的和
# det	计算矩阵行列式
# eig	计算方阵的本征值和本征向量
# inv	计算方阵的逆
# pinv	计算矩阵的Moore-Penrose伪逆
# qr	计算QR分解
# svd	计算奇异值分解（SVD）
# solve	解线性方程
# lstsq	计算Ax=b的最小二乘解

# fixme 4.数组的聚合函数运算
'''
聚合函数是指对一组值（比如一个数组）进行操作，返回一个单一值
作为结果的函数，比如求数组所有元素之和就是聚合函数。常见的聚合
函数有：求和，求最大最小，求平均，求标准差，求中位数等。
'''
# sum	求和运算
# cumsum	累积求和运算
# min	求最小值
# max	求最大值
# mean	求均值
# median	求中位数
# var	求方差
# std	求标准差
arr = np.random.randn(3, 4)
temp = np.max(arr)
print(arr)
print(temp)
'''
把二维数组的垂直方向定义为axis 0轴，水平方向为axis 1轴。
当我们在对Numpy进行运算时，我们把axis=0指定为垂直方向的
计算，axis=1指定为水平方向的运算。
'''
temp = np.max(arr, axis=0)
print(temp)

# fixme 5.小试牛刀

# 利用函数解决一些问题
# 利用自带的随机数生成函数生成5位选手的评委打分结果，一共有7位评委。打分结果用5×7大小的数组表示
votes = np.random.randint(1, 10, (5, 7))
print(votes)
result = (np.sum(votes, axis=1) - np.max(votes, axis=1) - np.min(votes, axis=1)) / 5
print(result)

# 利用Numpy实现条件判断
# where 函数中输入3个参数，分别是判断条件、为真时的值，为假时的值
# 在Numpy中，空值是一种新的数据格式，我们用np.nan产生空值
temp = np.where(arr_rnd < 5, np.nan, arr_rnd)
print(temp)


# fixme 6.自定义函数np.frompyfunc
# 定义函数，购买x件订单，返回订单金额
def order(x):
    if x >= 100:
        return 20 * 0.6 * x
    if x >= 50:
        return 20 * 0.8 * x
    if x >= 10:
        return 20 * 0.9 * x
    return 20 * x


# frompyfunc函数有三个输入参数，分别是待转化的函数、函数的输入参数的个数、函数的返回值的个数
income = np.frompyfunc(order, 1, 1)
# order_lst 为5位顾客的下单量
order_lst = [600, 300, 5, 2, 85]
# 计算当天的营业额
result = np.sum(income(order_lst))
print(result)



