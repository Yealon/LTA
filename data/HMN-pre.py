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
#  注意！！！为了测试，改变了row[1],row[2]为 row[0],row[1]
spl = 146
# f = open('/home/dkb/workspace/Code/pretrain/Hie_dic.json', encoding='utf-8')  # 打开文件
f = open('Hie/Hie_dic.json', encoding='utf-8')  # 打开文件
Hie_dic = json.load(f)  # 把json串变成python的数据类型：字典，传一个文件对象，它会帮你读文件，不需要再单独读文件


def transform_multilabel_as_multihot(label_list,label_size,Hieflag):
    """
    convert to multi-hot style
    :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
    :param label_size: e.g.199
    :return:e.g.[1,1,0,1,0,0,........]
    """
    result=np.zeros(label_size)
    if Hieflag == 0:
        result[label_list] = 1
    else:
        result[0] = 0 # pass
    return result

def get_X_Y(train_data_x,label_size,firstnum,secondnum,thirdnum,forthnum,test_mode=False):
    """
    get X and Y given input and labels
    input:
    train_data_x:
    train_data_y:
    label_size: number of total unique labels(e.g. 1999 in this task)
    output:
    X,Y
    """
    # f = open('/home/dkb/workspace/Code/pretrain/Hie_dic.json', encoding='utf-8')  # 打开文件

    X=[]
    Y=[]
    datanum1 = 0
    datanum2 = 0
    datanum3 = 0
    datanum4 = 0

    train_data_x_tiny_test = train_data_x
    data_pre_dic = {}

    for index, row in train_data_x_tiny_test.iterrows():
        #if index==0: continue
        topic_ids=row[1]
        desc_char = row[2]
        #topic_id_list=topic_ids.split(",")#multi-label


        # 处理label
        if topic_ids in label2index:
            # if Hie_dic.get(topic_ids) is None: # 处理找不到的label
            #     continue
            '''
            firs2ind={}
            seco2ind={}
            thir2ind={}
            fort2ind={}
            '''
            #  首先处理label的层次级别  处理-1
            #######  start统分别统计有多少个一类，二类，三类案由
            if Hie_dic[topic_ids][1] == -1:
                first_Hie = transform_multilabel_as_multihot([firs2ind[topic_ids]], firstnum, 0)
                second_Hie = transform_multilabel_as_multihot(0, secondnum, 1)
                third_Hie = transform_multilabel_as_multihot(0, thirdnum, 1)
                forth_Hie = transform_multilabel_as_multihot(0, forthnum, 1)
                datanum1 += 1
            elif Hie_dic[topic_ids][2] == -1:
                first_Hie = transform_multilabel_as_multihot(0, firstnum, 1)
                second_Hie = transform_multilabel_as_multihot([seco2ind[topic_ids]], secondnum, 0)
                third_Hie = transform_multilabel_as_multihot(0, thirdnum, 1)
                forth_Hie = transform_multilabel_as_multihot(0, forthnum, 1)
                datanum2 += 1
            elif Hie_dic[topic_ids][3] == -1:
                first_Hie = transform_multilabel_as_multihot(0, firstnum, 1)
                second_Hie = transform_multilabel_as_multihot(0, secondnum, 1)
                third_Hie = transform_multilabel_as_multihot([thir2ind[topic_ids]], thirdnum, 0)
                forth_Hie = transform_multilabel_as_multihot(0, forthnum, 1)
                datanum3 += 1
            else:
                first_Hie = transform_multilabel_as_multihot(0, firstnum, 1)
                second_Hie = transform_multilabel_as_multihot(0, secondnum, 1)
                third_Hie = transform_multilabel_as_multihot(0, thirdnum, 1)
                forth_Hie = transform_multilabel_as_multihot([fort2ind[topic_ids]], forthnum, 0)
                datanum4 += 1
            #######  end分别统计有多少个一类，二类，三类案由，相加等于sql

            # ### old erro
            # first_Hie = transform_multilabel_as_multihot(Hie_dic[topic_ids][0], firstnum, 0)
            # if Hie_dic[topic_ids][1] == -1:
            #     second_Hie = transform_multilabel_as_multihot(0, 43, 1)
            # else:
            #     second_Hie = transform_multilabel_as_multihot(Hie_dic[topic_ids][1], 43, 0)
            #
            # if Hie_dic[topic_ids][2] == -1:
            #     third_Hie = transform_multilabel_as_multihot(0, 425, 1)
            # else:
            #     third_Hie = transform_multilabel_as_multihot(Hie_dic[topic_ids][2], 425, 0)
            #
            # if Hie_dic[topic_ids][3] == -1:
            #     forth_Hie = transform_multilabel_as_multihot(0, 374, 1)
            # else:
            #     forth_Hie = transform_multilabel_as_multihot(Hie_dic[topic_ids][3], 374, 0)

            labels_tuple = (first_Hie.tolist(), second_Hie.tolist(), third_Hie.tolist(), forth_Hie.tolist())


            ###处理label的向量转换 start
            label_list_dense = [label2index[topic_ids]]
            label_list_sparse = transform_multilabel_as_multihot(label_list_dense, label_size,0)
            Y.append(label_list_sparse)  # addY
            # if index % 1000000 == 0:
            #     print(index, ";label_list_dense:", label_list_dense)
            ###处理label的向量转换 end

            # 存入字典
            data_pre_dic[str(label_list_sparse)] = labels_tuple
            #再把字典存入文件，在训练预加载时读取

        #  处理描述文本
            desc_char_list = desc_char.split(" ")
            desc_char_id_list = [word2index.get(x, UNK_ID) for x in desc_char_list if x.strip()]
            X.append(desc_char_id_list)
            if index < 3: print(index, desc_char_id_list)
            # if index % 1000000 == 0: print(index, desc_char_id_list)

        else:
            continue

        # print(data_pre_dic)

        #label_list_dense=[label2index[l] for l in topic_id_list if l.strip()]#label2id
        #label_list_sparse=transform_multilabel_as_multihot(label_list_dense,label_size)#label2one-hot
        # Y.append(label_list_sparse)# addY
        # if index%1000000==0: print(index,";label_list_dense:",label_list_dense)
    # print(data_pre_dic)
    res2 = json.dumps(data_pre_dic, indent=8, ensure_ascii=False)
    # print(res2)
    with open("Hie/train_label_tuple.json", 'w', encoding='utf-8') as f:
        f.write(res2)

    print("*****first level data num:" + str(datanum1) + "*****")
    print("*****second level data num:" + str(datanum2) + "*****")
    print("*****third level data num:" + str(datanum3) + "*****")
    print("*****forth level data num:" + str(datanum4) + "*****")

    return X,Y

def save_data(cache_file_h5py,cache_file_pickle,word2index,label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y):
    # train/valid/test data using h5py
    f = h5py.File(cache_file_h5py, 'w')
    f['train_X'] = train_X
    f['train_Y'] = train_Y
    f['vaild_X'] = vaild_X
    f['valid_Y'] = valid_Y
    f['test_X'] = test_X
    f['test_Y'] = test_Y
    f.close()
    # save word2index, label2index
    with open(cache_file_pickle, 'ab') as target_file:
        pickle.dump((word2index,label2index), target_file)

print ("read source file as csv")
base_path='./'

train_data_x=pd.read_csv(base_path+'all.csv', encoding="utf-8")

train_data_x=train_data_x.fillna('')

print("train_data_x:",train_data_x.shape)
#print("train_data_y:",train_data_y.shape)
#print("valid_data_x:",valid_data_x.shape)



print ("compute average length of title_char, title_word, desc_char, desc_word")
#dict_length_columns={'title_char':0,'title_word':0,'desc_char':0,'desc_word':0}
dict_length_columns={'desc_word':0}
num_examples=len(train_data_x)
# print(num_examples)
#train_data_x_small=train_data_x.sample(frac=0.01)
for index, row in train_data_x.iterrows():
    #title_char_length=len(row['title_char'].split(","))
    #title_word_length=len(row['title_word'].split(","))
    #desc_char_length=len(row['desc_char'].split(","))
    desc_word_length=len(row[2].split(" "))
    #print(desc_word_length)
    #dict_length_columns['title_char']=dict_length_columns['title_char']+title_char_length
    #dict_length_columns['title_word']=dict_length_columns['title_word']+title_word_length
    #dict_length_columns['desc_char']=dict_length_columns['desc_char']+desc_char_length
    dict_length_columns['desc_word']=dict_length_columns['desc_word']+desc_word_length
dict_length_columns={k:float(v)/float(num_examples) for k,v in dict_length_columns.items()}
print("dict_length_columns:",dict_length_columns)


print ("create vocabulary_dict, label_dict, generate training/validation data, and save to some place")
# create vocabulary of charactor token by read word_embedding.txt
word_embedding_object = open(base_path + 'embedding.txt')
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

# write to vocab.txt under data/ieee_zhihu_cup
vocab_path = base_path + 'vocab-Hie-'+str(spl)+'.txt'
vocab_char_object = open(vocab_path, 'w')

word2index = {}
for i, char in enumerate(char_list):
    if i < 10: print(i, char)
    word2index[char] = i
    vocab_char_object.write(char + "\n")
vocab_char_object.close()
print("vocabulary of char generated....")


print("generate labels list, and save to file system")
c_labels=Counter()
# train_data_y_small=train_data_y[0:100000]#.sample(frac=0.1)
for index, row in train_data_x.iterrows():
    topic_ids=row[1]
    topic_list=topic_ids.split(',')
    if '其他' in topic_list:
        continue
    if Hie_dic.get(topic_list[0]) is None:  # 处理找不到的label
        print(topic_list[0])
        continue
    c_labels.update(topic_list)

label_list=c_labels.most_common(spl) # select!!!!!!
label2index={}
firstlevelnum = 0
secondlevelnum = 0
thirdlevelnum = 0
forthlevelnum = 0
firs2ind={}
seco2ind={}
thir2ind={}
fort2ind={}

label_target_object=open(base_path+'label_seqHie-'+str(spl)+'.txt','w')


for i, label_freq in enumerate(label_list):
    label,freq=label_freq
    label2index[label]=i
#######  start统计训练集中，每层输出的维度
    if Hie_dic[label][1] ==-1:
        firs2ind[label] = firstlevelnum
        firstlevelnum += 1
    else:
        if Hie_dic[label][2] == -1:
            seco2ind[label] = secondlevelnum
            secondlevelnum +=1
        else:
            if Hie_dic[label][3] == -1:
                thir2ind[label] = thirdlevelnum
                thirdlevelnum += 1
            else:
                fort2ind[label] = forthlevelnum
                forthlevelnum += 1
#######  end统计训练集中，每层输出的维度，也就是截取后有多少个一类，二类，三类案由，相加等于sql
    label_target_object.write(label+"|"+str(freq)+"\n")
    # if i<20: print(label,freq)
label_target_object.close()
###### start将各个层次数量记录进文件
label_Hie_object=open(base_path+'every_num-'+str(spl)+'.txt','w')

# if firstlevelnum ==1: firstlevelnum += 1
# if secondlevelnum ==1: secondlevelnum += 1
# if thirdlevelnum ==1: thirdlevelnum += 1
# if forthlevelnum ==1: forthlevelnum += 1

label_Hie_object.write(str(firstlevelnum)+"\n")
label_Hie_object.write(str(secondlevelnum)+"\n")
label_Hie_object.write(str(thirdlevelnum)+"\n")
label_Hie_object.write(str(forthlevelnum)+"\n")
label_Hie_object.close()
###### end将各个层次数量记录进文件


jso1 = json.dumps(firs2ind, indent=8, ensure_ascii=False)
jso1  = jso1+"\n"
jso2 = json.dumps(seco2ind, indent=8, ensure_ascii=False)
jso2  = jso2+"\n"
jso3 = json.dumps(thir2ind, indent=8, ensure_ascii=False)
jso3  = jso3+"\n"
jso4 = json.dumps(fort2ind, indent=8, ensure_ascii=False)
jso4  = jso4+"\n"
with open("Hie/every_name.json", 'w', encoding='utf-8') as f:
    f.write(jso1)
    f.write(jso2)
    f.write(jso3)
    f.write(jso4)


print("generate label dict successful...")

print ("generate training/validation/test data using source file and vocabulary/label set.")
#  get X,Y---> shuffle and split data----> save to file system.
test_mode=False
label_size=len(label2index)
print(label_size)

# save number of labels
# total_label_obj = open(base_path+'total_label_num-stratifyHie-'+str(spl)+'.txt','w')
# total_label_obj.write(str(label_size) + "\n")
# total_label_obj.close()

cache_path_h5py=base_path+'dataHie-'+str(spl)+'.h5'
cache_path_pickle=base_path+'vocab_labelHie-'+str(spl)+'.pik'
max_sentence_length=200

# step 1: get (X,y)
X,Y=get_X_Y(train_data_x,label_size,firstlevelnum, secondlevelnum, thirdlevelnum, forthlevelnum,test_mode=test_mode)

print("get done!")
# pad and truncate to a max_sequence_length
X = pad_sequences(X, maxlen=max_sentence_length, value=0.)  # padding to max length
print("pad done")

# step 2. shuffle, split,
xy=list(zip(X,Y))
random.Random(5869134).shuffle(xy)
X,Y=zip(*xy)
X=np.array(X); Y=np.array(Y)

num_examples=len(X)
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

train_X,X_combine, train_Y, Y_combine =train_test_split(X,Y,test_size=0.3, random_state=0,stratify=Y)
print("Combine size")
print (X_combine.shape)
print (Y_combine.shape)
print("start to split it again")

test_X,vaild_X, test_Y, valid_Y =train_test_split(X_combine,Y_combine,test_size=0.5, random_state=0,stratify=Y_combine)
print("split done")

print("num_examples:",num_examples,";X.shape:",X.shape,";Y.shape:",Y.shape)
print("train_X:",train_X.shape,";train_Y:",train_Y.shape,";vaild_X.shape:",vaild_X.shape,";valid_Y:",valid_Y.shape,";test_X:",test_X.shape,";test_Y:",test_Y.shape)

# step 3: save to file system
save_data(cache_path_h5py,cache_path_pickle,word2index,label2index,train_X,train_Y,vaild_X,valid_Y,test_X,test_Y)
print("save cache files to file system successfully!")



del X,Y,train_X, train_Y,vaild_X, valid_Y,test_X, test_Y






