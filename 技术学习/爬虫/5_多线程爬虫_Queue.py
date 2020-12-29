from queue import Queue
import time
import threading

# q = Queue(4)

# for i in range(4):
#     q.put(i)
#
# for i in range(4):
#     print(q.get())
def set_value(q):
    index = 0
    while True:
        q.put(index)
        index += 1
        time.sleep(1)

def get_value(q):
    while True:
        print(q.get())


def main():
    q = Queue(4)
    t1 = threading.Thread(target=set_value,args=[q])
    t2 = threading.Thread(target=get_value, args=[q])

    t1.start()
    t2.start()
    # 若没有值,会阻塞
    # q.get(block=True)

if __name__ == "__main__":
    main()
