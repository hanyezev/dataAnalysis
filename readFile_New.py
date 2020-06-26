import xlrd  # 引入模块
import time
from pandas import DataFrame
import pandas as pd

data = pd.read_csv("C:/Users/16526/Desktop/电网项目/主机10.217.14.100.csv", encoding="ANSI")
print(len(data))


def strToTimeStamp(str_):
    new_str = str_[0:4] + "-" + str_[4:6] + "-" + str_[6:8] + " " + str_[8:10] + ":" + str_[10:12] + ":" + str_[12:]
    timeArray = time.strptime(new_str, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


def timeStampToStr(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


date = []
TimeStamp = []
OldDate = data["DATA_DT"]
RAM = data["内存负载"]
CPU = data["主机UPC平均负负载"]

for i in OldDate:
    date_temp = timeStampToStr(strToTimeStamp(str(int(float(i)))))
    date.append(date_temp)
    TimeStamp.append(strToTimeStamp(str(int(float(i)))))

# print(date)       '2020-03-01 00:00:00'元素
# print(TimeStamp)  1582992000

reduceS = 1590937200 - 1582995600  # 2020-03-01 01:00:00 至 2020-05-31 23:00:00的秒数
print((reduceS / 300) == len(data))
reduceH = reduceS / (60 * 60)   # 2206

X_date = []
X_timeStamp = []
Y1_ram = []
Y2_cpu = []
startT = 1582995600

for i in range(int(reduceH)+1):
    X_timeStamp.append(startT + i * 60 * 60)

def searchAvg(startTime):
    temp_r = []
    temp_c = []
    flag = 0
    for i in range(len(TimeStamp)):
        if (TimeStamp[i] <= startTime) & (TimeStamp[i] > startTime - 60 * 60):
            flag = 1
            ram_ = RAM[i]
            cpu_ = CPU[i]
            temp_r.append(ram_)
            temp_c.append(cpu_)
    if flag == 0:
        return 0, 0
    return format(sum(temp_r) / len(temp_r), '.4f'), format(sum(temp_c) / len(temp_c), '.4f')


for i in range(len(X_timeStamp)):
    X_date.append(timeStampToStr(X_timeStamp[i]))
    y1, y2 = searchAvg(X_timeStamp[i])

    Y1_ram.append(y1)
    Y2_cpu.append(y2)


# print(X_date[0] + " " + str(Y1_ram[0]) + " " + str(Y2_cpu[0]))
# print(len(X_date))

# 生成数据表
# data = {
#     '日期': X_date,
#     'DATA_DT': X_timeStamp,
#     '内存负载': Y1_ram,
#     '主机CPU平均负载': Y2_cpu
# }
# df = DataFrame(data)
# df.to_excel("C:/Users/16526/Desktop/电网项目/服务器性能数据.xlsx")

print("success")