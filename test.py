'''
时间：2021.8.11
作者：手可摘星辰不去高声语
名称：03-进程中执行带有参数的任务.py
'''
 
# 1.导入包和模块
import multiprocessing
import time
 
 
def sing(num, name):
    for i in range(num):
        print(name)
        print("---i am sing ooo~")
        time.sleep(0.5)
 
 
def dance(num, name):
    for i in range(num):
        print(name)
        print("i am dance lll~")
        time.sleep(0.5)
 
 
if __name__ == '__main__':
    # 2.使用进程类创建进程对象
    # target:指定进程执行的函数名，不加括号
    # args:使用元组方式给指定任务传参，顺序一致(参数顺序)
    # kwargs:使用字典的方式给指定任务传参，名称一致(参数名称)
    sing_process = multiprocessing.Process(target=sing, args=(3, "猪猪"))
    dance_process = multiprocessing.Process(target=dance, kwargs={"name": "珊珊", "num": 2})
 
    # 3. 使用进程对象启动进程执行指定任务
    sing_process.start()
    dance_process.start()