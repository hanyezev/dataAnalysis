import threading
import random
import time

gMoney = 1000
gCondition = threading.Condition()
gTotalTimes = 10
gTimes = 0

class Producer(threading.Thread):
    def run(self):
        global gMoney
        global gTimes
        while True:
            money = random.randint(100,1000)
            gCondition.acquire()
            if gTimes >= gTotalTimes:
                gCondition.release()
                break
            gMoney += money
            print(f"{threading.current_thread()}生产了{money}元钱,剩余{gMoney}元钱")
            gTimes += 1
            gCondition.notifyAll()
            gCondition.release()
            time.sleep(0.5)

class Consumer(threading.Thread):
    def run(self):
        global gMoney
        while True:
            money = random.randint(100,1000)
            gCondition.acquire()
            while gMoney < money:
                if gTimes >= gTotalTimes:
                    gCondition.release()
                    return
                print(f"{threading.current_thread()}准备消费{money}元钱,剩余{gMoney}元钱,余额不足.")
                gCondition.wait()
            gMoney -= money
            print(f"{threading.current_thread()}消费了{money}元钱,剩余{gMoney}元钱")
            gCondition.release()
            time.sleep(0.5)

def main():
    for i in range(3):
        t = Consumer()
        t.start()
    for i in range(5):
        t = Producer()
        t.start()
if __name__ == "__main__":
    main()