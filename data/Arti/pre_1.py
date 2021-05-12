import time
import re
import pandas as pd

import numpy as np
import pickle
import time,os
from multiprocessing import Pool

filepath = '/home/dkb/workspace/Code/Code/pretrain/Arti/data/Arti'
savepath = '/home/dkb/workspace/Code/Code/pretrain/Arti/data/val'
import json

def wash(name):
    # print("read source file as csv")
    filename = os.path.join(filepath, name)
    # print(filename)
    train_data_x = pd.read_csv(filename, encoding="utf-8", header=None)
    train_data_x = train_data_x.fillna('')
    # print("train_data_x:", train_data_x.shape)
    ''''
    index, "label", "des", "tort","contract","marriage"
    '''
    i = 0
    for index, row in train_data_x.iterrows():
        # print(row)
        i += 1
        # print(row[3])
        # print(row[4])
        # print(row[5])
        # print("+++++++")
        if row[3] == '' and row[4] == '' and row[5] == '':
            train_data_x.drop(index = index,inplace=True)
            # if i < 10:
            #     print(row)

        # if i % 5000 == 0:
        #     print(i)

    print("washed:", train_data_x.shape)

    filename = os.path.join(savepath, name)
    train_data_x.to_csv(filename, header=None, index=False,encoding="utf-8")

    # train_data_x.to_csv('../all_Article.csv', header=None, encoding="utf-8")


if __name__=='__main__':
    n = 0
    for file in os.listdir(filepath):
        n += 1
        print(n)
        wash(file)
    #开启进程，与逻辑核保持一致

    # t1 = time.time()
    # pro_num = 1 #进程数
    # n = 0
    # pool = Pool(processes = pro_num)
    # job_result = []
    # #遍历文件夹读取所有文件
    # for file in os.listdir(filepath):
    #     n += 1
    #     res = pool.apply_async(wash, (file,))
    #     job_result.append(n)
    #
    # pool.close() #关闭进程池
    # pool.join()
    # print(job_result)


