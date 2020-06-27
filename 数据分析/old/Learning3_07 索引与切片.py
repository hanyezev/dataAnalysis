import numpy as np

# fixme 1.数组索引与切片
# 一维数组
arr1d = np.arange(1, 10, 1, dtype=np.float32)
print(arr1d)
print(arr1d[6])
print(arr1d[5:8])  # 左闭右开

# 二维数组
arr2d = np.arange(9, dtype=np.float32).reshape(3, 3)
print(arr2d)
print(arr2d[1])
print(arr2d[1, 1])
print(arr2d[0:2, 1:3])

# 三维数组
arr3d = np.arange(1, 19, 1).reshape(3, 2, 3)
print(arr3d)

# 布尔型索引
cities = np.array(["hz", "sh", "hz", "bj", "wh", "sh", "sz"])
arr_rnd = np.random.randn(7, 4)
print(arr_rnd)
arr_temp = cities == "hz"  # 生成一个布尔类型的数组
print(arr_temp)
print(arr_rnd[cities == "hz"])  # 取第一行和第三行
print(arr_rnd[cities == "hz", :3])
# 把数组中的负数都筛选出来,变为0
arr_rnd[arr_rnd < 0] = 0
print(arr_rnd)

# fixme 2.花式索引
arr_demo01 = np.arange(24).reshape(4, 6)
# 方法1：分别将4个角的元素索引出来，然后把取出来的4个元素，重新组成一个新2×2的数组
arr_method1 = np.array([[arr_demo01[0, 0], arr_demo01[0, -1]],
                        [arr_demo01[-1, 0], arr_demo01[-1, -1]]])
# 方法2：利用布尔索引，可以同时索引不连续的行。分别对axis 0方向和axis 1方向进行索引。但是需要注意的是，得分2次索引；
arr_method2 = arr_demo01[[True, False, False, True]][:, [True, False, False, False, False, True]]
# 方法3：分别传入4个角的坐标，请朋友们注意观察传入的2个整数数组的规律
arr_temp = arr_demo01[[0, 0, -1, -1], [0, -1, 0, -1]]
print(arr_temp)
# 方法4：利用花式索引和切片混用，整体思路和方法2很相似。也是通过连续2次索引，得到一个矩形状的区域
arr_temp = arr_demo01[[0, -1]][:, [0, -1]]
print(arr_temp)
