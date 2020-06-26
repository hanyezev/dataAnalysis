import datetime
from matplotlib import pyplot as plt
import random
import time

str_startTime = "2020-6-11 10:00"
startTime = datetime.datetime.strptime(str_startTime, '%Y-%m-%d %H:%M:%S')
x = []
y = []

for i in range(120):
    y.append(random.randint(20, 35))

# 设置展示大小
plt.figure(figsize=(3, 2), dpi=200)

# 绘制图片
plt.plot(x, y)

# 设置x轴的刻度
plt.xticks(range(1, 120, 1))

# 保存图片
# plt.savefig("./01.jpg")

plt.show()