#  #!/usr/bin/env python3
#  # -*- coding:utf-8 -*-
#  # @FileName :Test.py
#  # @Software PyCharm
#
# import os
# import pandas as pd
#
# # filename为文件路径，file_num为拆分后的文件行数
# # 根据是否有表头执行不同程序，默认有表头的
# def Data_split(filename,file_num,header=True):
#     if header:
#         # 设置每个文件需要有的行数,初始化为1000W
#         chunksize=10000
#         data1=pd.read_table(filename,chunksize=chunksize,encoding='utf-8')
#          # print(data1)
#          # num表示总行数
#         num=0
#         for chunk in data1:
#              num += len(chunk)
#         # print(num)
#         # chunksize表示每个文件需要分配到的行数
#         chunksize=round(num/file_num+1)
#         # print(chunksize)
#         # 分离文件名与扩展名os.path.split(filename)
#         head,tail=os.path.split(filename)
#         data2=pd.read_table(filename,chunksize=chunksize,sep=',',encoding='gbk')
#         i=0
#         for chunk in data2:
#             chunk.to_csv('{0}_{1}{2}'.format(head,i,tail),header=None,index=False)
#             print('保存第{0}个数据'.format(i))
#             i+=1
#     else:
#        # 获得每个文件需要的行数
#         chunksize=10000
#         data1=pd.read_table(filename,chunksize=chunksize,header=None,sep=',')
#         num=0
#         for chunk in data1:
#             num+=len(chunk)
#             chunksize=round(num/file_num+1)
#
#             head,tail=os.path.split(filename)
#             data2=pd.read_table(filename,chunksize=chunksize,header=None,sep=',')
#             i=0
#             for chunk in data2:
#                 chunk.to_csv('{0}_{1}{2}'.foemat(head,i,tail),header=None,index=False)
#                 print('保存第{0}个数据'.format(i))
#                 i+=1
#
# filename='文件路径'
# #num为拆分为的文件个数
# Data_split(filename,num,header=True)


# -*- coding:utf-8 -*-
# author:chenpeng
# date: 2017-11-07
# 作用：根据需要拆分的文件数，拆分文件
# 备注：可以拆分csv格式文件和txt格式文件，返回的数据均是没有表头
import os
import pandas as pd


def file_split(filename, file_num, header=False):
    # 根据是否有表头执行不同程序，默认是否表头的
    if header:
        # 获得每个文件需要有的行数
        chunksize = 1000000  # 先初始化的chunksize是100W
        data1 = pd.read_table(filename, chunksize=chunksize, sep=',', encoding='utf-8')
        num = 0
        for chunk in data1:
            num += len(chunk)
        chunksize = round(num / file_num + 1)

        # 需要存的file
        head, tail = os.path.splitext(filename)
        data2 = pd.read_table(filename, chunksize=chunksize, sep=',', encoding='utf-8')
        i = 0  # 定文件名
        for chunk in data2:
            chunk.to_csv('{0}_{1}{2}'.format(head, i, tail), header=None)
            print('保存第{0}个数据'.format(i))
            i += 1
    else:
        # 获得每个文件需要有的行数
        chunksize = 1000000  # 先初始化的chunksize是100W
        data1 = pd.read_csv(filename, chunksize=chunksize, header=None, sep=',',encoding='utf-8')
        num = 0
        for chunk in data1:
            num += len(chunk)
        chunksize = round(num / file_num + 1)

        # 需要存的file
        head, tail = os.path.splitext(filename)
        data2 = pd.read_csv(filename, chunksize=chunksize, header=None, sep=',')
        i = 0  # 定文件名
        for chunk in data2:
            chunk.to_csv('{0}_{1}{2}'.format(head, i, tail), header=None, index=False)
            print('保存第{0}个数据'.format(i))
            i += 1


if __name__ == '__main__':
    filename = '/home/dkb/workspace/Code/pretrain/Arti/Article_des.csv'
    file_split(filename, 500, header=False)