import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
import random
import numpy as np
import json
import h5py
import pickle
from sklearn.model_selection import train_test_split

'''
################################################################################
########################## 第一种，构造二三级的数据集 ##########################
取spl个三级案由，记录他们对应的二级案由

################################################################################
'''
# 构造 数据集D中的D1
spl = 90
# f = open('/home/dkb/workspace/Code/pretrain/Hie_dic.json', encoding='utf-8')  # 打开文件
f = open('/home/dkb/workspace/Code/pretrain/Hie/Hie_dic.json', encoding='utf-8')  # 打开文件
Hie_dic = json.load(f)  # 把json串变成python的数据类型：字典，传一个文件对象，它会帮你读文件，不需要再单独读文件

fr1 = open("/home/dkb/workspace/Code/pretrain/Hie/first2index.txt", 'r+')
one_dic = eval(fr1.read())  # 读取的str转换为字典
fr1.close()
one_dic = {v: k for k, v in one_dic.items()}

fr2 = open("/home/dkb/workspace/Code/pretrain/Hie/second2index.txt", 'r+')
two_dic = eval(fr2.read())  # 读取的str转换为字典
fr2.close()
two_dic = {v: k for k, v in two_dic.items()}

fr3 = open("/home/dkb/workspace/Code/pretrain/Hie/third2index.txt", 'r+')
three_dic = eval(fr3.read())  # 读取的str转换为字典
fr3.close()
three_dic = {v: k for k, v in three_dic.items()}

fr4 = open("/home/dkb/workspace/Code/pretrain/Hie/forth2index.txt", 'r+')
four_dic = eval(fr4.read())  # 读取的str转换为字典
fr4.close()
four_dic = {v: k for k, v in four_dic.items()}

'''
读取数据
'''
print("read source file as csv")
base_path = '/home/dkb/workspace/Code/pretrain/No_level_pre/'
names = ['index', 'label', 'des', 'tort', 'contract', 'marriage']

train_data_x = pd.read_csv('/home/dkb/workspace/Code/pretrain/all_Art_washed.csv', encoding="utf-8", header=None,
                           dtype=object)
#  test mode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# shkiprows=10000,  # 跳过前十行
# , nrows=20000)  # 只取前10000行
#  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train_data_x.astype(str)

train_data_x = train_data_x.fillna('')
print("train_data_x:", train_data_x.shape)
#  test mode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# train_data_x = train_data_x[:10000]
#  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


'''
处理 label
'''
print("generate labels list, and save to file system")

# step1，清洗label
c_labels = Counter()
for index, row in train_data_x.iterrows():
    label_name = row[1]
    # topic_list=topic_ids.split(',')
    if '其他' == label_name:  # 处理噪音：'其他'
        continue
    if Hie_dic.get(label_name) is None:  # 处理找不到的label
        print(label_name)
        continue
    c_labels.update([label_name])
# class_list = c_labels.most_common(spl)  # select!!!!!!!!
class_list = c_labels.most_common()  # all

# step2，根据层次筛选label，三四层合并
class2index = {}

label_2 = Counter()
label_3 = Counter()

# label_target_object = open(base_path + '3_level_class_seq_' + str('all') + '.txt', 'w')

# 根据class整理label
label_2_dic = {}
label_3_dic = {}
for i, class_freq in enumerate(class_list):
    classname, freq = class_freq  # label:name of label
    class2index[classname] = i  # 记录全局class list

    # start 统计每层label数量，构建二三层，每层上的label索引second2ind，third2ind
    if Hie_dic[classname][1] == -1:  # 一级案由
       continue

    elif Hie_dic[classname][2] == -1:  # 二级案由
        # 收录二级案由
        if classname == "不当得利纠纷":
            classname = classname + "2"
        label_2.update(classname = freq)
        # label_2_dic[classname] += freq


    elif Hie_dic[classname][3] == -1:
        # 收录三级案由
        label_3.update([classname])
        label_3_dic[classname] += freq
        # 收录三级案由的二级案由
        label_name_2 = two_dic[Hie_dic[classname][1]]
        if label_name_2 == "不当得利纠纷":
            label_name_2 = label_name_2 + "2"
        label_2.update([label_name_2])
        label_2_dic[classname] += freq

    else: # 四级案由
        # 收录四级案由的二级案由
        label_name_2 = two_dic[Hie_dic[classname][1]]
        if label_name_2 == "不当得利纠纷":
            label_name_2 = label_name_2 + "2"
        label_2.update([label_name_2])
        label_2_dic[classname] += freq
        # 收录四级案由的三级案由
        label_name_3 = three_dic[Hie_dic[classname][2]]
        label_3.update(label_name_3=freq)
        # label_3_dic[classname] += freq
    # end
    # label_target_object.write(classname + "|" + str(freq) + "\n")

    # 将label按频率记录进文件


# label_target_object.close()

# 总label按照频率记录
all_label = label_2 + label_3
label_writ = open(base_path + 'label_all' + '.txt', 'w')
# 不同层次有重名，这里处理了90类上的：不当得利纠纷（2，3），侵权责任纠纷（1，2），人格权纠纷（1，2）
for i, label in enumerate(all_label):
    # label:name of label
    label_writ.write(label + "\n")
label_writ.close()

# 将各个层次上的label按索引记录
# firs2ind = {}
seco2ind = {}
thir2ind = {}
# fort2ind = {}

for i, label in enumerate(label_2):
    # label:name of label
    seco2ind[label] = i
for i, label in enumerate(label_3):
    # label:name of label
    thir2ind[label] = i

f1 = open(base_path + "level_index.txt", 'w')

f1.write(str(seco2ind) + "\n")
f1.write(str(thir2ind) + "\n")

f1.close()

print("generate label dict successful...")

'''
划分集合
'''
print("generate training/validation/test data")
#  get X,Y---> shuffle and split data----> save to file system.
# all_label_size = len(all_label)
# print("label number：" + str(all_label_size))
label_1_size = len(label_1)
print("label_1 number：" + str(label_1_size))
label_2_size = len(label_2)
print("label_2 number：" + str(label_2_size))
label_3_size = len(label_3)
print("label_3 number：" + str(label_3_size))
label_4_size = len(label_4)
print("label_4 number：" + str(label_4_size))

all_label_size = label_4_size + label_3_size + label_2_size + label_1_size
checksize = len(all_label)
print("checksize：" + str(checksize))
print("all_label_size：" + str(all_label_size))

text_train_path = base_path + 'text_train'
label_train_path = base_path + 'label_train'

text_val_path = base_path + 'text_val'
label_val_path = base_path + 'label_val'

text_test_path = base_path + 'text_test'
label_test_path = base_path + 'label_test'

append_train_path = base_path + 'append_train'
append_test_path = base_path + 'append_test'
append_val_path = base_path + 'append_val'

max_sentence_length = 200

'''
get XY
'''
datanum1 = 0
datanum2 = 0
datanum3 = 0
datanum4 = 0

ii = 0
for index, row in train_data_x.iterrows():
    # if index==0: continue
    topic_ids = row[1]
    desc_char = row[2]
    tort = row[3]
    contract = row[4]
    marriage = row[5]
    ii += 1
    if ii % 1000000 == 0:
        print(ii)

    # 处理class
    if topic_ids in class2index:
        # if Hie_dic.get(topic_ids) is None: # 处理找不到的label
        #     continue
        '''
        firs2ind={}
        seco2ind={}
        thir2ind={}
        fort2ind={}
        '''

        #  start 分级别处理案由的标签，转化为one-hot，即：input_y_1，input_y_2，...
        #  同时处理label——multi

        if Hie_dic[topic_ids][1] == -1:
            #  一级案由 ：input_y_1，input_y_2，...
            if topic_ids == "人格权纠纷" or topic_ids == "侵权责任纠纷" or topic_ids == "海事海商纠纷":
                topic_ids = topic_ids + "1"
            train_data_x.loc[index, 1] = topic_ids
            datanum1 += 1

        elif Hie_dic[topic_ids][2] == -1:
            #  二级案由
            # 二级案由的一级案由label:input_y_1
            Hie1_name = one_dic[Hie_dic[topic_ids][0]]
            if Hie1_name == "人格权纠纷" or Hie1_name == "侵权责任纠纷" or Hie1_name == "海事海商纠纷":
                Hie1_name = Hie1_name + "1"
            if topic_ids == "不当得利纠纷":
                topic_ids = topic_ids + "2"
            train_data_x.loc[index, 1] = Hie1_name + " " + topic_ids

            # 计数
            datanum1 += 1
            datanum2 += 1

        elif Hie_dic[topic_ids][3] == -1:
            # 三级案由
            # 三级案由的一级案由label:input_y_1
            Hie1_name = one_dic[Hie_dic[topic_ids][0]]
            if Hie1_name == "人格权纠纷" or Hie1_name == "侵权责任纠纷" or Hie1_name == "海事海商纠纷":
                Hie1_name = Hie1_name + "1"
            # 三级案由的二级案由label:input_y_2
            Hie2_name = two_dic[Hie_dic[topic_ids][1]]
            if Hie2_name == "不当得利纠纷":
                Hie2_name = Hie2_name + "2"

            train_data_x.loc[index, 1] = Hie1_name + " " + Hie2_name + " " + topic_ids
            # 计数
            datanum1 += 1
            datanum2 += 1
            datanum3 += 1
        else:
            # 四级案由
            # 四级案由的一级案由label:input_y_1
            Hie1_name = one_dic[Hie_dic[topic_ids][0]]
            if Hie1_name == "人格权纠纷" or Hie1_name == "侵权责任纠纷" or Hie1_name == "海事海商纠纷":
                Hie1_name = Hie1_name + "1"
            # 四级案由的二级案由label:input_y_2
            Hie2_name = two_dic[Hie_dic[topic_ids][1]]
            if Hie2_name == "不当得利纠纷":
                Hie2_name = Hie2_name + "2"
            # 四级案由的三级案由label:input_y_3
            Hie3_name = three_dic[Hie_dic[topic_ids][2]]
            # 四级案由的四级案由label:input_y_4
            train_data_x.loc[index, 1] = Hie1_name + " " + Hie2_name + " " + Hie3_name + " " + topic_ids

            # 计数
            datanum1 += 1
            datanum2 += 1
            datanum3 += 1
            datanum4 += 1
        # end

        #  处理Article—list
        Arti = tort + ',' + contract + ',' + marriage
        train_data_x.loc[index, 3] = Arti

    else:
        #  不是截取得到的label
        train_data_x.drop(index=index, inplace=True)
        continue

print("*****first level data num:" + str(datanum1) + "*****")
print("*****second level data num:" + str(datanum2) + "*****")
print("*****third level data num:" + str(datanum3) + "*****")
print("*****forth level data num:" + str(datanum4) + "*****")

train_data_x.drop(columns=4, inplace=True)
train_data_x.drop(columns=5, inplace=True)

# 2： shuffle, split,
train_data_x = shuffle(train_data_x)
print("shuffle done")

############# new version（2020-03-03 update）
# base on ：
# https://www.wandouip.com/t5i398001/
# https://github.com/kikizxd/Data_preprocessing
# https://blog.csdn.net/samsam2013/article/details/80702582


train_X, X_combine, train_Ycombine, Y_combine = train_test_split(train_data_x.iloc[:, 2], train_data_x.iloc[:, [1, 3]],
                                                                 test_size=0.02, random_state=0,
                                                                 stratify=train_data_x.iloc[:, 1])

print("start to split it again")

Y_s = Y_combine.iloc[:, 0]
# combine_Y, combine_Z = zip(*Yall_combine)
# combine_Y = np.array(combine_Y)
# combine_Z = np.array(combine_Z)
#
test_X, valid_X, test_Yall, valid_Yall = train_test_split(X_combine, Y_combine, test_size=0.5, random_state=0,
                                                          stratify=Y_s)

print("split done")

train_Y = train_Ycombine.iloc[:, 0]
train_Append = train_Ycombine.iloc[:, 1]

test_Y = test_Yall.iloc[:, 0]
test_Append = test_Yall.iloc[:, 1]

valid_Y = valid_Yall.iloc[:, 0]
valid_Append = valid_Yall.iloc[:, 1]

print("start to save")
train_X.to_csv(text_train_path, index=False, header=None, encoding="utf-8")
train_Y.to_csv(label_train_path, index=False, header=None, encoding="utf-8")

test_X.to_csv(text_test_path, index=False, header=None, encoding="utf-8")
test_Y.to_csv(label_test_path, index=False, header=None, encoding="utf-8")

valid_X.to_csv(text_val_path, index=False, header=None, encoding="utf-8")
valid_Y.to_csv(label_val_path, index=False, header=None, encoding="utf-8")

train_Append.to_csv(append_train_path, index=False, header=None, encoding="utf-8")
test_Append.to_csv(append_test_path, index=False, header=None, encoding="utf-8")
valid_Append.to_csv(append_val_path, index=False, header=None, encoding="utf-8")

print("Test")
srcF = open(text_test_path)
tgtF = open(label_test_path)
i = 0
while True:
    ssline = srcF.readline()
    tline = tgtF.readline()
    if ssline == "" and tline == "":
        break
    if i < 10:
        ssline = ssline.strip()
        print(ssline)
        print("++")
        i += 1
        tline = tline.strip()
        print(tline)
        print("--")

print("Done")
# train_Y, train_Z = zip(*train_Yall)
#
# train_X = np.array(train_X)
# train_Y = np.array(train_Y)
# train_Z = np.array(train_Z)
#
# test_Y, test_Z = zip(*test_Yall)
#
# test_X = np.array(test_X)
# test_Y = np.array(test_Y)
# test_Z = np.array(test_Z)
#
# valid_Y, valid_Z = zip(*valid_Yall)
#
# vaild_X = np.array(vaild_X)
# valid_Y = np.array(valid_Y)
# valid_Z = np.array(valid_Z)

# step 3: save to file system
#  change:level_index

# print("save cache files to file system successfully!")
#
# print("num_examples:", num_examples, ";X.shape:", X.shape, ";Y.shape:", Y.shape)
# print("train_X:", train_X.shape, ";train_Y:", train_Y.shape, ";vaild_X.shape:", vaild_X.shape, ";valid_Y:",
#       valid_Y.shape, ";test_X:", test_X.shape, ";test_Y:", test_Y.shape,
#       ";train_Z:", train_Z.shape, ";test_Z:", test_Z.shape, ";valid_Z:", valid_Z.shape)

del train_X, train_Y, valid_X, \
    valid_Y, test_X, test_Y, train_Append, test_Append, valid_Append
