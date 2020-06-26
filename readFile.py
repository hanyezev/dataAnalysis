import xlrd  # 引入模块
import time
from pandas import DataFrame

# 打开文件，获取excel文件的workbook（工作簿）对象
workbook1 = xlrd.open_workbook("C:/Users/16526/Desktop/电网项目/服务器性能数据.xls")  # 文件路径
workbook2 = xlrd.open_workbook("C:/Users/16526/Desktop/电网项目/服务器性能数据2.xls")
'''对workbook对象进行操作'''


# 获取所有sheet的名字
# names = workbook.sheet_names()
# print(names)  # ['各省市', '测试表']  输出所有的表名，以列表的形式

def strToTimeStamp(str_):
    new_str = str_[0:4] + "-" + str_[4:6] + "-" + str_[6:8] + " " + str_[8:10] + ":" + str_[10:12] + ":" + str_[12:]
    timeArray = time.strptime(new_str, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


def timeStampToStr(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


worksheet1 = workbook1.sheet_by_index(0)
worksheet2 = workbook2.sheet_by_index(0)

nrows1 = worksheet1.nrows  # 获取该表总行数
print(nrows1)

ncols1 = worksheet1.ncols  # 获取该表总列数
print(ncols1)

nrows2 = worksheet2.nrows  # 获取该表总行数
print(nrows2)

ncols2 = worksheet2.ncols  # 获取该表总列数
print(ncols2)

date = []
RAM = []
CPU = []
for i in range(1, nrows1):
    date.append(strToTimeStamp(str(int(worksheet1.row_values(i)[2]))))
    RAM.append(float(worksheet1.row_values(i)[3]))
    CPU.append(int(worksheet1.row_values(i)[5]))

for i in range(1, nrows2):
    ram = worksheet2.row_values(i)[7]
    cpu = worksheet2.row_values(i)[9]
    if ram == '':
        continue
    date.append(strToTimeStamp(str(int(worksheet2.row_values(i)[2]))))
    RAM.append(float(ram))
    CPU.append(int(cpu))

reduceS = 1566529200 - 1559322000  # 2019-06-01 01:00:00 至 2019-08-23 11:00:00的秒数
reduceH = reduceS / (60 * 60)

# print(date)
X_time = []
X_time_str = []
Y1_ram = []
Y2_cpu = []
startT = 1559322000

for i in range(int(reduceH)+1):
    X_time.append(startT+i*60*60)


def searchAvg(startTime):
    temp_r = []
    temp_c = []
    flag = 0
    for i in range(len(date)):
        if (date[i] <= startTime) & (date[i] > startTime-60 * 60):
            flag = 1
            ram_ = RAM[i]
            cpu_ = CPU[i]
            temp_r.append(ram_)
            temp_c.append(cpu_)
    if flag == 0:
        return 0, 0
    return format(sum(temp_r) / len(temp_r), '.4f'), format(sum(temp_c) / len(temp_c), '.4f')


for i in range(len(X_time)):
    X_time_str.append(timeStampToStr(X_time[i]))
    y1, y2 = searchAvg(X_time[i])
    Y1_ram.append(y1)
    Y2_cpu.append(y2)


# txt = open("C:/Users/16526/Desktop/电网项目/服务器性能数据.xlsx", "w")
# txt.write('DATA_DT'+";"+'内存负载'+";"+'主机CPU平均负载')
# txt.write("\n")
# for i in range(len(X_time)):  # 循环打印每一行
#     txt.write(str(X_time[i])+";"+str(Y1_ram[i])+";"+str(Y2_cpu[i]))
#     txt.write("\n")
# txt.close()

# data = {
#     '日期': X_time_str,
#     'DATA_DT': X_time,
#     '内存负载': Y1_ram,
#     '主机CPU平均负载': Y2_cpu
# }
# df = DataFrame(data)
# df.to_excel("C:/Users/16526/Desktop/电网项目/服务器性能数据.xlsx")

print("success")
