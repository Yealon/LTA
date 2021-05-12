# coding=utf-8

import pandas as pd
from collections import Counter
from tflearn.data_utils import pad_sequences
import random
import numpy as np
import json
import h5py
import pickle
from sklearn.model_selection import train_test_split

# {"id": "1", "title": ["tokens"], "abstract": ["tokens"], "section": [1, 2], "subsection": [1, 2, 3, 4],
# "group": [1, 2, 3, 4], "labels": [1, 2, 1+N, 2+N, 3+N, 4+N, 1+N+M, 2+N+M, 3+N+M, 4+N+M]}

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


def transform_multilabel_as_multihot(label_list, label_size, Hieflag):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result = np.zeros(label_size)
    if Hieflag == 0:
        result[label_list] = 1
    else:
        result[0] = 0  # pass
    return result


def get_X_Y(train_data_x, all_label_size, label_1_size, label_2_size, label_3_size, label_4_size):
    """
    get X and Y given input and labels
    input:
    train_data_x:
    train_data_y:
    label_size: number of total unique labels(e.g. 1999 in this task)
    output:
    X,Y
    """

    X = []
    Y = []
    Z = []
    datanum1 = 0
    datanum2 = 0
    datanum3 = 0
    datanum4 = 0

    train_data_x_tiny_test = train_data_x
    data_pre_dic = {}

    for index, row in train_data_x_tiny_test.iterrows():
        # if index==0: continue
        topic_ids = row[1]
        desc_char = row[2]
        tort = row[3]
        contract = row[4]
        marriage = row[5]


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
                first_Hie = transform_multilabel_as_multihot(firs2ind[topic_ids], label_1_size, 0)  # 0：在标签位置置1
                second_Hie = transform_multilabel_as_multihot(0, label_2_size, 1)  # 1：全0
                third_Hie = transform_multilabel_as_multihot(0, label_3_size, 1)
                forth_Hie = transform_multilabel_as_multihot(0, label_4_size, 1)
                # 一级案由：input_y
                label_list_dense = [firs2ind[topic_ids]]
                label_list_sparse = transform_multilabel_as_multihot(label_list_dense, all_label_size, 0)
                Y.append(label_list_sparse)  # addY
                # 计数
                datanum1 += 1

            elif Hie_dic[topic_ids][2] == -1:
                #  二级案由
                # 二级案由的一级案由label:input_y_1
                Hie1_name = one_dic[Hie_dic[topic_ids][0]]
                first_Hie = transform_multilabel_as_multihot(firs2ind[Hie1_name], label_1_size, 0)
                # 二级案由的label:input_y_2，...
                second_Hie = transform_multilabel_as_multihot([seco2ind[topic_ids]], label_2_size, 0)
                third_Hie = transform_multilabel_as_multihot(0, label_3_size, 1)
                forth_Hie = transform_multilabel_as_multihot(0, label_4_size, 1)
                # 二级案由：input_y
                label_list_dense = [firs2ind[Hie1_name], len(firs2ind) + seco2ind[topic_ids]]
                label_list_sparse = transform_multilabel_as_multihot(label_list_dense, all_label_size, 0)
                Y.append(label_list_sparse)  # addY
                # 计数
                datanum1 += 1
                datanum2 += 1
            elif Hie_dic[topic_ids][3] == -1:
                # 三级案由
                # 三级案由的一级案由label:input_y_1
                Hie1_name = one_dic[Hie_dic[topic_ids][0]]
                first_Hie = transform_multilabel_as_multihot(firs2ind[Hie1_name], label_1_size, 0)
                # 三级案由的二级案由label:input_y_2
                Hie2_name = two_dic[Hie_dic[topic_ids][1]]
                second_Hie = transform_multilabel_as_multihot(seco2ind[Hie2_name], label_2_size, 0)
                # 三级案由的label:input_y_3，.....
                third_Hie = transform_multilabel_as_multihot(thir2ind[topic_ids], label_3_size, 0)
                forth_Hie = transform_multilabel_as_multihot(0, label_4_size, 1)
                # 三级案由：input_y
                label_list_dense = [firs2ind[Hie1_name],
                                    len(firs2ind) + seco2ind[Hie2_name],
                                    len(firs2ind) + len(seco2ind) + thir2ind[topic_ids]
                                    ]
                label_list_sparse = transform_multilabel_as_multihot(label_list_dense, all_label_size, 0)
                Y.append(label_list_sparse)  # addY
                # 计数
                datanum1 += 1
                datanum2 += 1
                datanum3 += 1
            else:
                # 四级案由
                # 四级案由的一级案由label:input_y_1
                Hie1_name = one_dic[Hie_dic[topic_ids][0]]
                first_Hie = transform_multilabel_as_multihot(firs2ind[Hie1_name], label_1_size, 0)
                # 四级案由的二级案由label:input_y_2
                Hie2_name = two_dic[Hie_dic[topic_ids][1]]
                second_Hie = transform_multilabel_as_multihot(seco2ind[Hie2_name], label_2_size, 0)
                # 四级案由的三级案由label:input_y_3
                Hie3_name = three_dic[Hie_dic[topic_ids][2]]
                third_Hie = transform_multilabel_as_multihot(thir2ind[Hie3_name], label_3_size, 0)
                # 四级案由的四级案由label:input_y_4
                forth_Hie = transform_multilabel_as_multihot([fort2ind[topic_ids]], label_4_size, 0)
                # 四级案由：input_y
                label_list_dense = [firs2ind[Hie1_name],
                                    len(firs2ind) + seco2ind[Hie2_name],
                                    len(firs2ind) + len(seco2ind) + thir2ind[Hie3_name],
                                    len(firs2ind) + len(seco2ind) + len(thir2ind) + fort2ind[topic_ids]
                                    ]
                label_list_sparse = transform_multilabel_as_multihot(label_list_dense, all_label_size, 0)
                Y.append(label_list_sparse)  # addY
                # 计数
                datanum1 += 1
                datanum2 += 1
                datanum3 += 1
                datanum4 += 1
            # end

            #  存入input_y_1，input_y_2，...
            # 循环结束再把字典存入文件，在训练预加载时读取
            labels_tuple = (first_Hie.tolist(), second_Hie.tolist(), third_Hie.tolist(), forth_Hie.tolist())
            data_pre_dic[str(label_list_sparse)] = labels_tuple
            # end

            #  处理描述文本
            desc_char_list = desc_char.split(" ")
            desc_char_id_list = [word2index.get(x, UNK_ID) for x in desc_char_list if x.strip()]
            X.append(desc_char_id_list)
            if index < 3: print(index, desc_char_id_list)

            #  处理Article—list
            Arti = tort+','+contract+','+marriage
            Z.append(Arti)
        else:
            #  不是截取得到的label
            continue

    # 结束循环，把字典存入文件，在训练预加载时读取
    res2 = json.dumps(data_pre_dic, indent=8, ensure_ascii=False)
    # print(res2)
    with open(base_path+"train_label_tuple.json", 'w', encoding='utf-8') as f:
        f.write(res2)

    print("*****first level data num:" + str(datanum1) + "*****")
    print("*****second level data num:" + str(datanum2) + "*****")
    print("*****third level data num:" + str(datanum3) + "*****")
    print("*****forth level data num:" + str(datanum4) + "*****")

    return X, Y, Z


def save_data(cache_file_h5py, cache_file_pickle, word2index,firs2ind,seco2ind,thir2ind,fort2ind, train_X, train_Y, vaild_X, valid_Y, test_X,
              test_Y,train_Z,test_Z,valid_Z):
    # train/valid/test data using h5py
    f = h5py.File(cache_file_h5py, 'w')
    f['train_X'] = train_X
    f['train_Y'] = train_Y
    f['vaild_X'] = vaild_X
    f['valid_Y'] = valid_Y
    f['test_X'] = test_X
    f['test_Y'] = test_Y
    train_Z_ansc = []
    for j in train_Z:
        train_Z_ansc.append(j.encode())
    test_Z_ansc = []
    for j in test_Z:
        test_Z_ansc.append(j.encode())
    vaild_Z_ansc = []
    for j in valid_Z:
        vaild_Z_ansc.append(j.encode())
    f['train_Z'] = train_Z_ansc
    f['test_Z'] = test_Z_ansc
    f['valid_Z'] = vaild_Z_ansc

    f.close()
    # save word2index, label2index
    with open(cache_file_pickle, 'ab') as target_file:
        pickle.dump((word2index, firs2ind,seco2ind,thir2ind,fort2ind), target_file)


'''
读取数据
'''
print("read source file as csv")
base_path = '/home/dkb/workspace/Code/pretrain/Arti/pre/'
names = ['index', 'label', 'des', 'tort', 'contract', 'marriage']

train_data_x = pd.read_csv('/home/dkb/workspace/Code/pretrain/all_Art_washed.csv', encoding="utf-8", header=None,
                            dtype = object)

train_data_x.astype(str)

train_data_x = train_data_x.fillna('')
print("train_data_x:", train_data_x.shape)
#  test mode!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# train_data_x = train_data_x[:10000]
#  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


'''
构建词表
'''
print("create vocabulary_dict")
# create vocabulary of charactor token by read word_embedding.txt
word_embedding_object = open('/home/dkb/workspace/Code/pretrain/embedding.txt')
lines_wv = word_embedding_object.readlines()
word_embedding_object.close()
char_list = []
char_list.extend(['PAD', 'UNK', 'CLS', 'SEP', 'unused1', 'unused2', 'unused3', 'unused4', 'unused5'])
PAD_ID = 0
UNK_ID = 1
for i, line in enumerate(lines_wv):
    if i == 0: continue
    char_embedding_list = line.split(" ")
    char_token = char_embedding_list[0]
    char_list.append(char_token)

vocab_path = base_path + 'vocab-Hie-' + str(spl) + '.txt'
vocab_char_object = open(vocab_path, 'w')

word2index = {}
for i, char in enumerate(char_list):
    if i < 10: print(i, char)
    word2index[char] = i
    vocab_char_object.write(char + "\n")
vocab_char_object.close()
print("vocabulary of char generated....")

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
label_list = c_labels.most_common(spl)  # select!!!!!!!!

# step2，根据层次筛选label，三四层合并
class2index = {}

label_1 = Counter()
label_2 = Counter()
label_3 = Counter()
label_4 = Counter()

label_target_object = open(base_path + 'class_seq_' + str(spl) + '.txt', 'w')

# 根据class整理label
for i, label_freq in enumerate(label_list):
    label, freq = label_freq  # label:name of label

    class2index[label] = i  # 记录全局label list

    # start 统计每层label数量，构建每层上的label索引firs2ind，seco2ind，thir2ind，fort2ind
    if Hie_dic[label][1] == -1:  # 一级案由
        # 收录一级案由
        label_1.update([label])

    elif Hie_dic[label][2] == -1:  # 二级案由
        # 收录二级案由
        label_2.update([label])
        # 收录二级案由的一级案由
        label_name_1 = one_dic[Hie_dic[label][0]]
        label_1.update([label_name_1])

    elif Hie_dic[label][3] == -1:
        # 收录三级案由
        label_3.update([label])
        # 收录三级案由的一级案由
        label_name_1 = one_dic[Hie_dic[label][0]]
        label_1.update([label_name_1])
        # 收录三级案由的二级案由
        label_name_2 = two_dic[Hie_dic[label][1]]
        label_2.update([label_name_2])
    else:
        # 收录四级案由
        label_4.update([label])
        # 收录四级案由的一级案由
        label_name_1 = one_dic[Hie_dic[label][0]]
        label_1.update([label_name_1])
        # 收录四级案由的二级案由
        label_name_2 = two_dic[Hie_dic[label][1]]
        label_2.update([label_name_2])
        # 收录四级案由的三级案由
        label_name_3 = three_dic[Hie_dic[label][2]]
        label_3.update([label_name_3])
    # end

    # 将label按频率记录进文件
    label_target_object.write(label + "|" + str(freq) + "\n")

label_target_object.close()

# 总label按照频率记录
# all_label = label_1 + label_2 + label_3 + label_4
# label_writ = open(base_path + 'label_no_use' + str(spl) + '.txt', 'w')
# # 因为不同层次有重名，这里被合并了，不可以作为数据来源
# for i, label in enumerate(all_label):
#     # label:name of label
#     label_writ.write(label + "\n")
# label_writ.close()

# 将各个层次上的label按索引记录
firs2ind = {}
seco2ind = {}
thir2ind = {}
fort2ind = {}
for i, label in enumerate(label_1):
    # label:name of label
    firs2ind[label] = i
for i, label in enumerate(label_2):
    # label:name of label
    seco2ind[label] = i
for i, label in enumerate(label_3):
    # label:name of label
    thir2ind[label] = i
for i, label in enumerate(label_4):
    # label:name of label
    fort2ind[label] = i
f1 = open(base_path + "level_index.txt", 'w')
f1.write(str(firs2ind) + "\n")
f1.write(str(seco2ind) + "\n")
f1.write(str(thir2ind) + "\n")
f1.write(str(fort2ind) + "\n")
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

all_label_size = label_4_size+label_3_size+label_2_size+label_1_size

cache_path_h5py = base_path + 'dataHie-' + str(spl) + '.h5'
cache_path_pickle = base_path + 'vocab_labelHie-' + str(spl) + '.pik'
max_sentence_length = 200

# 1: get (X,y)
X, Y, Z = get_X_Y(train_data_x, all_label_size, label_1_size, label_2_size, label_3_size, label_4_size)

print("get done!")
# pad and truncate to a max_sequence_length
X = pad_sequences(X, maxlen=max_sentence_length, value=0.)  # padding to max length
print("pad done")

# 2： shuffle, split,
xyz = list(zip(X, Y, Z))
random.Random(5869134).shuffle(xyz)
X, Y, Z = zip(*xyz)
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
num_examples = len(X)
print("shuffle done")
############## old version（2020-03-03 delete）
# num_valid=30
# num_valid=30
# num_valid=int(num_examples*0.15)    # 5869134*0.15 total.txt
# num_valid=int(num_examples*0.15)   #total.txt!!!!!!!!!!!!!!!!!!!!!!!!!!!
# num_train=num_examples-(num_valid+num_valid)
# train_X, train_Y=X[0:num_train], Y[0:num_train]
# vaild_X, valid_Y=X[num_train:num_train+num_valid], Y[num_train:num_train+num_valid]
# test_X, test_Y=X[num_train+num_valid:], Y[num_train+num_valid:]


############# new version（2020-03-03 update）
# base on ：
# https://www.wandouip.com/t5i398001/
# https://github.com/kikizxd/Data_preprocessing
# https://blog.csdn.net/samsam2013/article/details/80702582

yall = list(zip(Y, Z))

train_X, X_combine, train_Yall, Yall_combine = train_test_split(X, yall, test_size=0.02, random_state=0, stratify=Y)

print("start to split it again")
combine_Y, combine_Z = zip(*Yall_combine)
combine_Y = np.array(combine_Y)
combine_Z = np.array(combine_Z)


test_X, vaild_X, test_Yall, valid_Yall = train_test_split(X_combine, Yall_combine, test_size=0.5, random_state=0,
                                                    stratify=combine_Y)

print("split done")

train_Y, train_Z = zip(*train_Yall)

train_X = np.array(train_X)
train_Y = np.array(train_Y)
train_Z = np.array(train_Z)

test_Y, test_Z = zip(*test_Yall)

test_X = np.array(test_X)
test_Y = np.array(test_Y)
test_Z = np.array(test_Z)

valid_Y, valid_Z = zip(*valid_Yall)

vaild_X = np.array(vaild_X)
valid_Y = np.array(valid_Y)
valid_Z = np.array(valid_Z)

# step 3: save to file system
#  change:level_index

save_data(cache_path_h5py, cache_path_pickle, word2index, firs2ind,seco2ind,thir2ind,fort2ind, train_X, train_Y, vaild_X, valid_Y, test_X,
          test_Y,train_Z,test_Z,valid_Z)
print("save cache files to file system successfully!")

print("num_examples:", num_examples, ";X.shape:", X.shape, ";Y.shape:", Y.shape)
print("train_X:", train_X.shape, ";train_Y:", train_Y.shape, ";vaild_X.shape:", vaild_X.shape, ";valid_Y:",
      valid_Y.shape, ";test_X:", test_X.shape, ";test_Y:", test_Y.shape,
      ";train_Z:", train_Z.shape,";test_Z:", test_Z.shape,";valid_Z:", valid_Z.shape)

del X, Y, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y,train_Z,test_Z,valid_Z
