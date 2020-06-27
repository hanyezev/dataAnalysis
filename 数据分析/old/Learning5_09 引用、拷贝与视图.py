import copy
import numpy as np

# fixme 1.Python篇

# Python 引用
a = ["a", "b", "c"]
b = a
print(a is b)

# Python的深拷贝与浅拷贝
# 1.深拷贝
m = ["Jack", "Tom", "Brown"]
n = copy.deepcopy(m)
print(m == n)
print(m is n)
# 改变m首位的元素，发现n并无变化，说明二者互不影响
m[0] = "Helen"
print(m)
print(n)
# 2.浅拷贝
m = ["Jack", "Tom", "Brown"]
n = copy.copy(m)
print(m == n)
print(m is n)
# 特殊情况:嵌套列表
# 对于嵌套列表里的可变元素（深层次的数据结构），浅拷贝并没有进行拷贝，只是对其进行了引用
students = ["Class 1", ["Jack", 178, 120], ["Tom", 174, 109]]
students_c = copy.copy(students)
print(students[1] is students_c[1])
# a.尝试更改students中某位学生的信息，通过测试更改后的students和students_c
students[1][1] = 180
# students_c[1][1]会变
print(students)
print(students_c)
# b.尝试更改students中的班级信息
students[0] = "Class 2"
# students_c[0]不会变
print(students)
print(students_c)

# 切片与浅拷贝
# 切片其实就是对源列表进行部分元素的浅拷贝
students = ["Class 1", ["Jack", 178, 120], ["Tom", 174, 109]]
students_silce = students[:2]
students_silce[-1][1] = 185
print(students_silce)
print(students)

# fixme 2.Numpy篇
# 对于Numpy来讲，我们主要甄别两个概念，即视图与副本
'''
视图view是对数据的引用，通过该引用，可以方便地访问、操作原有数据，但原有数据不会产生拷贝。
如果我们对视图进行修改，它会影响到原始数据，因为它们的物理内存在同一位置。
副本是对数据的完整拷贝（Python中深拷贝的概念），如果我们对副本进行修改，
它不会影响到原始数据，它们的物理内存不在同一位置。
'''
# view视图
arr_0 = np.arange(12).reshape(3, 4)
view_0 = arr_0.view()
print(id(arr_0) is view_0)
# 更改视图的元素，则原始数据会产生联动效果
view_0[1, 1] = 100
print(arr_0)
print(view_0)
# 视图的纬度更改并不会传递到原始数组
view_0.shape = (4, 3)
print(arr_0.shape)
print(view_0.shape)
# 副本
'''
Numpy建立副本的方法稍有不同。
方法一，是利用Numpy自带的copy函数；
方法二，是利用deepcopy（）函数。
'''
arr_2 = np.array([[1, 2, 3], [4, 5, 6]])
copy_2 = arr_2.copy()
copy_2[1, 1] = 500
print(arr_2)
print(copy_2)
