import numpy as np

# fixme 1.ndarray对象与创建

data0 = [2, 4, 6.5, 8]
arr0 = np.array(data0)
print(str(type(arr0)) + str(arr0))

# 可以用isinstance函数来判断是否是ndarray类型
print(isinstance(arr0, np.ndarray))

# 创建多维数组
data1 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr1 = np.array(data1)

# 利用dtype关键字，传入合适的数据类型，显式地定义
arr2 = np.array(data1, dtype=np.float32)
# 查看类型
print(arr2.dtype)
# 查看维度
print(arr2.shape)

# 通过整形1和0，定义布尔类型的数组
data3 = [[1, 0], [0, 1]]
arr3 = np.array(data3, dtype=np.bool)
print(arr3)
data4 = [["a", "b"], ["c", ""]]
arr4 = np.array(data4, dtype=np.bool)
print(arr4)
arr5 = np.array(data4, dtype=np.string_)
print(arr5.dtype)   # |S1 元素的固定长度为1

# fixme 2.数据类型的更改与格式化输出

# 更改ndarray的数据类型, 用astype函数来对数组进行操作
data6 = [[1.230, 2.670], [1.450, 6.000]]
arr6 = np.array(data6, np.float32)
print(arr6.dtype)
arr6_ = arr6.astype(np.float16)
print(arr6.dtype)
print(arr6_.dtype)

# 数组的格式化输出
# precision: 默认保留8位位有效数字，后面不会补0；supress: 对很大/小的数不使用科学计数法 (True)
np.set_printoptions(precision=3, suppress=True)
arr7 = np.array([[3.141592653], [9.8]], dtype=np.float16)	    # 定义一个2维数组
np.set_printoptions(precision=3, suppress=True)
print(arr7)

# fixme 3.创建数组的其它方式
# zeros函数，创建指定维度的全为0的数组
# 创建一个大小为10的全0数组
np.zeros(10, dtype=np.int8)         # 大小为10的全0数组
np.zeros((2, 3), dtype=np.float16)  # 大小为2×3的全0数组
# ones函数，创建指定维度的全为1的数组
np.ones((2,3), dtype=np.float16)
# empty函数，创建一个空数组，只分配内存空间，但是不填充任何值
np.empty((2,3), dtype=np.int8)
# identity函数，创建一个大小为n×n的单位矩阵（对角线为1，其余为0）
temp = np.identity(3, dtype=np.int8)
print(temp)
# eye函数，identity的升级版本
# 如果同时指定N和M，则输出大小为N×M的矩形矩阵。K为调节值，调节为1的对角线的位置偏离度。
# 创建3×4的矩形矩阵
temp = np.eye(N=3, M=4, dtype=np.int8)
print(temp)
# 创建3×4的矩形矩阵，并且为1的对角线向右偏移1个单位。
temp = np.eye(N=3, M=4, k=1, dtype=np.int8)
print(temp)