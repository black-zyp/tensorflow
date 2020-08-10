#
# txt = open("./songs.txt","r",encoding="utf8")
# x = list(txt.read().split("！"))
# for i in x:
#     print(i)
#
# import threading
#
# import time
#
# x = 0
#
# mutex = threading.Lock()
#
# def work1(num):
#     global x
#     for i in range(num):
#         mutex.acquire()
#         x = x + 1
#         mutex.release()
#
#     print("work1---- x=:", x)
#
#
# def work2(num):
#     global x
#     for i in range(num):
#         mutex.acquire()
#         x = x + 1
#         mutex.release()
#     print("work2---- x=:", x)
#
# def main():
#     work_one = threading.Thread(target=work1,args=(1000000,))
#     work_one.start()
#     work_two = threading.Thread(target=work2,args=(1000000,))
#     work_two.start()
#     while len(threading.enumerate()) != 1:
#         time.sleep(1)
#
# if __name__ == '__main__':
#     main()


import os
from multiprocessing import Process
import time

num = [11,22]

def work1(number):
    for i in range(number):
        num.append(i)
        time.sleep(1)
        print("进程:work1,pid={},num={}".format(os.getpid(),num))


def work2():
    print("进程:work2,pid={},num={}".format(os.getpid(), num))

def main():
    number1 = Process(target=work1,args=(3,))
    number1.start()

    time.sleep(5)

    number2 = Process(target=work2)
    number2.start()

if __name__ == '__main__':
    main()

