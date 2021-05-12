import os
import re

import numpy as np
import json

'''
分析result结果，寻找target与output不同的数据
'''
# ref_dir ='/home/dkb/workspace/Code/Code/Train_eval/Seq2Seq_Beam'
# ref_file = os.path.join(ref_dir, "result12.txt")
# print(ref_file)
#
# f = open(ref_file)
# lines = f.readlines()
# count = 0
# target = []
# output = []
# for line in lines:
#     if count % 2 == 0:
#         target.append(line.strip().split('| ')[1])
#         # print(target)
#     if count % 2 == 1:
#         output.append(line.strip().split('| ')[1])
#         # print(target)
#     count += 1
# f.close()
#
# out_dir ='/home/dkb/workspace/Code/Code//Train_eval/Analyze'
# out_file = os.path.join(out_dir, "Seq2SeqBeam.txt")
#
# with open(out_file, mode='a') as filename:
#     for i in range(0, len(target)):
#         if target[i] != output[i]:
#             filename.write(" |target| " + "".join(str(target[i]).strip()) + '\n')
#             filename.write(" |output| " + "".join(str(output[i]).strip()) + '\n')
#
# print("Done")
#

# '''
# 分级别分析，各级别错误
# '''
# out_dir ='/home/dkb/workspace/Code/Code/analyze/new_SeqATT'
# out_file = os.path.join(out_dir, "result%d.txt" % epoch)
#
# f = open(out_file)
# lines = f.readlines()
# count = 0
# target = []
# output = []
# for line in lines:
#     if count % 2 == 0:
#         target.append(line.strip().replace('[','').replace(']','').replace('\'','').split('| ')[1].split(', '))
#         # print(target)
#     if count % 2 == 1:
#         output.append(line.strip().replace('[','').replace(']','').replace('\'','').split('| ')[1].split(', '))
#         # print(target)
#     count += 1
# f.close()
#
# # #############level 1#################
# # level_1_file = os.path.join(out_dir, "level-1-%d.txt" % epoch)
# # num_1 = 0
# # with open(level_1_file, mode='a') as filename:
# #     for i in range(0, len(target)):
# #         if target[i][0] != output[i][0]:
# #             # print(target[i])
# #             filename.write(" |target| " + "".join(str(target[i]).strip()) + '\n')
# #             filename.write(" |output| " + "".join(str(output[i]).strip()) + '\n')
# #             num_1 += 1
# # print("num1: %d" %(num_1))
#
# # ##############level 2#################
# # level_2_file = os.path.join(out_dir, "level-2-%d.txt" % epoch)
# # num_2 = 0
# # with open(level_2_file, mode='a') as filename:
# #     for i in range(0, len(target)):
# #         if target[i][0] == output[i][0] and target[i][1] != output[i][1]:
# #             # print(target[i])
# #             # print(output[i])
# #             filename.write(" |target| " + "".join(str(target[i]).strip()) + '\n')
# #             filename.write(" |output| " + "".join(str(output[i]).strip()) + '\n')
# #             num_2 += 1
# # print("num2: %d" %(num_2))
#
# ##############level 3#################
# level_3_file = os.path.join(out_dir, "level-3-len-new-%d.txt" % epoch)
# num_3 = 0
# num_3_wa = 0
# num_3_len = 0
# with open(level_3_file, mode='a') as filename:
#     for i in range(0, len(target)):
#         if target[i][0] == output[i][0] and target[i][1] == output[i][1]:
#             if (len(target[i]) == 2 or len(output[i]) == 2) and len(output[i]) != len(target[i]):
#                 print("level 3 length not match")
#                 print(target[i])
#                 print(output[i])
#                 # filename.write(" |target| " + "".join(str(target[i]).strip()) + '\n')
#                 # filename.write(" |output| " + "".join(str(output[i]).strip()) + '\n')
#                 num_3_len += 1
#                 num_3 += 1
#             elif len(target[i]) != 2 and len(output[i]) != 2:
#                 if target[i][2] != output[i][2]:
#                     print("level 3 wrong")
#                     print(target[i])
#                     print(output[i])
#                     # filename.write(" |target| " + "".join(str(target[i]).strip()) + '\n')
#                     # filename.write(" |output| " + "".join(str(output[i]).strip()) + '\n')
#                     num_3_wa += 1
#                     num_3 += 1
#                 else:continue
#             # filename.write(" |target| " + "".join(str(target[i]).strip()) + '\n')
#             # filename.write(" |output| " + "".join(str(output[i]).strip()) + '\n')
#
# print("num3: %d" %(num_3))
#
# # ##############level 4#################
# # level_4_file = os.path.join(out_dir, "level-4-wa-new-%d.txt" % epoch)
# # num_4 = 0
# # num_4_wa = 0
# # num_4_len = 0
# # with open(level_4_file, mode='a') as filename:
# #     for i in range(0, len(target)):
# #         if len(target[i]) >= 3 and len(output[i]) >= 3:
# #             if target[i][0] == output[i][0] \
# #                     and target[i][1] == output[i][1] \
# #                     and target[i][2] == output[i][2]:
# #                 if (len(target[i]) == 3 or len(output[i]) == 3) and len(output[i]) != len(target[i]):
# #                     print("level 3 length not match")
# #                     print(target[i])
# #                     print(output[i])
# #                     # filename.write(" |target| " + "".join(str(target[i]).strip()) + '\n')
# #                     # filename.write(" |output| " + "".join(str(output[i]).strip()) + '\n')
# #
# #                     num_4_len += 1
# #                     num_4 += 1
# #                 elif len(target[i]) != 3 and len(output[i]) != 3:
# #                     if target[i][3] != output[i][3]:
# #                         print("level 3 wrong")
# #                         print(target[i])
# #                         print(output[i])
# #                         num_4_wa += 1
# #                         num_4 += 1
# #                         # filename.write(" |target| " + "".join(str(target[i]).strip()) + '\n')
# #                         # filename.write(" |output| " + "".join(str(output[i]).strip()) + '\n')
# #
# #                     else:continue
# #             # filename.write(" |target| " + "".join(str(target[i]).strip()) + '\n')
# #             # filename.write(" |output| " + "".join(str(output[i]).strip()) + '\n')
# #
# # print("num4: %d" %(num_4))
#
# print("Done")


# '''
# 构建level之间关系
# '''
# '''
# 构建分层的字典
# 根据level-index文件
# 找出两个级别案由之间的切分关系
# 转化为【0，1，0，1，0，0，0】
# '''
#
# # base_path = '/home/dkb/workspace/Code/analyze/'
#
# f2 = open("/home/dkb/workspace/Code/nlp_data/jieba/level_index.txt", 'r+')
# onedic_name_label = eval(f2.readline())  # 读取的str转换为字典
# onedic_label_name = {v: k for k, v in onedic_name_label.items()}
# twodic_name_label = eval(f2.readline())  # 读取的str转换为字典
# twodic_label_name = {v: k for k, v in twodic_name_label.items()}
# threedic_name_label = eval(f2.readline())  # 读取的str转换为字典
# threedic_label_name = {v: k for k, v in threedic_name_label.items()}
# fourdic_name_label = eval(f2.readline())  # 读取的str转换为字典
# fourdic_label_name = {v: k for k, v in fourdic_name_label.items()}
# f2.close()
#
# f = open('/home/dkb/workspace/Code/pretrain/Hie/Hie_dic.json', encoding='utf-8')  # 打开文件
# Hie_dic = json.load(f)  # 把json串变成python的数据类型：字典，传一个文件对象，它会帮你读文件，不需要再单独读文件
# f.close()
# level_list = []
# for k, v in Hie_dic.items():
#     level_list.append(v)
# # 因为Hie.dic中的索引是根据下面的来的，因此需要通过first/second/..._dic的name作为中介寻找在Hie中的索引
#
# fr1 = open("/home/dkb/workspace/Code/pretrain/Hie/first2index.txt", 'r+')
# first_dic_name_index = eval(fr1.read())  # 读取的str转换为字典
# first_dic_index_name = {v: k for k, v in first_dic_name_index.items()}
# fr1.close()
#
# fr2 = open("/home/dkb/workspace/Code/pretrain/Hie/second2index.txt", 'r+')
# second_dic_name_index = eval(fr2.read())
# second_dic_index_name = {v: k for k, v in second_dic_name_index.items()}
# fr2.close()
#
# fr3 = open("/home/dkb/workspace/Code/pretrain/Hie/third2index.txt", 'r+')
# third_dic_name_index = eval(fr3.read())
# third_dic_index_name = {v: k for k, v in third_dic_name_index.items()}
# fr3.close()
#
# fr4 = open("/home/dkb/workspace/Code/pretrain/Hie/forth2index.txt", 'r+')
# forth_dic_name_index = eval(fr4.read())
# forth_dic_index_name = {v: k for k, v in forth_dic_name_index.items()}
# fr4.close()
#
# listdic = {}
# restdic = {}
# dic_1_2 = []
# # dic_2_3 = []
# # dic_3_4 = []
# # resdic_3_4 = []
# # resdic_2_3 = []
# # resdic_1_2 = []
# i = 0
# for key1 in twodic_name_label.keys():# 这里是实际索引
#     vec1 = Hie_dic[key1]
#     print("**********parent:" + str(twodic_name_label[key1]) + str(key1))
#     listchild = []
#     restchild = []
#     for key2 in third_dic_name_index.keys():# 这里是字典里面的索引
#         vec2 = Hie_dic[key2]
#         if (vec1[1] == vec2[1]):  # 1
#             try:
#                 listchild.append(threedic_name_label[key2])# 字典转化为实际索引
#                 print("-child:" + str(threedic_name_label[key2]) + str(key2))
#             except KeyError as e:
#                 continue
#
#         else:
#             try:
#                 restchild.append(threedic_name_label[key2])
#                 # print("+++rest:" + str(forth_dic_name_index[key2])+str(key2))
#             except KeyError as e:
#                 continue
#     dic_1_2.append(listchild)
#     listdic[i] = listchild
#     restdic[i] = restchild
#     print("child list:")
#     print(listchild)
#     print("rest list:")
#     print(restchild)
#     i += 1
#
# print("##")
#
#
# # fw = open("/home/dkb/workspace/Code/Code/pretrain/Arti/3-4.txt",'w+')
# # fw.write(str(dic_3_4)) #把字典转化为str
# # fw.close()
# # 补全，方便tensor
# # maxlen = 0
# # for v in resdic_1_2:
# #     if len(v)>maxlen:
# #         maxlen = len(v)
# #
# # k = 0
# # for v in resdic_3_4:
# #     while True:
# #         if len(v) < maxlen:
# #             resdic_3_4[k].append(resdic_3_4[k][0])
# #         else:
# #             # resdic_1_2[k] = np.array(resdic_1_2[k])
# #             k+=1
# #             break
#
# # aaa = np.array(resdic_1_2)
# aaa = np.ones((len(twodic_name_label),len(threedic_name_label)),dtype=int)
#
#
# for i, val in enumerate(dic_1_2):
#     if len(val) != 0:
#         aaa[i][np.array(val)] = 0
#
# print("hup")
# # fw = open("/home/dkb/workspace/Code/Code/pretrain/Arti/res-1-2.txt",'w+')
# # fw.write(str(resdic_1_2)) #把字典转化为str
# # fw.close()
#
# # base_path = '/home/dkb/workspace/Code/analyze/'
#
# np.savetxt("/home/dkb/workspace/Code/Code/analyze/np-zero-2-3.txt",aaa)
#
# print("Done")
#

'''
分析错误的标签出现在当前层级还是其他层级
'''
# f2 = open("/home/dkb/workspace/Code/nlp_data/jieba/level_index.txt", 'r+')
# onedic_name_label = eval(f2.readline())  # 读取的str转换为字典
# onedic_label_name = {v: k for k, v in onedic_name_label.items()}
# twodic_name_label = eval(f2.readline())  # 读取的str转换为字典
# twodic_label_name = {v: k for k, v in twodic_name_label.items()}
# threedic_name_label = eval(f2.readline())  # 读取的str转换为字典
# threedic_label_name = {v: k for k, v in threedic_name_label.items()}
# fourdic_name_label = eval(f2.readline())  # 读取的str转换为字典
# fourdic_label_name = {v: k for k, v in fourdic_name_label.items()}
# f2.close()
#
# dic3_4 = np.loadtxt("/home/dkb/workspace/Code/Code/analyze/np-zero-3-4.txt")
# dic3_4 = dic3_4.astype(np.int)
#
#
# out_dir ='/home/dkb/workspace/Code/Code/analyze/new_SeqATT'
# out_file = os.path.join(out_dir, "level-4-wa-8.txt")
#
# f = open(out_file)
# lines = f.readlines()
# count = 0
# target = []
# output = []
# for line in lines:
#     if count % 2 == 0:
#         target.append(line.strip().replace('[','').replace(']','').replace('\'','').split('| ')[1].split(', '))
#         # print(target)
#     if count % 2 == 1:
#         output.append(line.strip().replace('[','').replace(']','').replace('\'','').split('| ')[1].split(', '))
#         # print(target)
#     count += 1
# f.close()
#
# out_range = 0
# in_range = 0
# for i in range(0,len(output)):
#     wrong = output[i][3]
#     previous = output[i][2]
#     wrongid = fourdic_name_label[wrong]
#     previousid = threedic_name_label[previous]
#     if dic3_4[previousid][wrongid] == 1:
#         out_range += 1
#     else:
#         in_range += 1


# '''
# 分析一级案由之间哪些出现错误比较多
# '''
# in_dir ='/home/dkb/workspace/Code/Code/Test_eval/Transformer/'
# in_file = os.path.join(in_dir, "result2.txt")
# # in_dir ='/home/dkb/workspace/Code/Code/Train_eval/Seq2Seq_Beam/'
# # in_file = os.path.join(in_dir, "result12.txt")
# level = 2  # 统计第几级的错误（start 0）
#
# f = open(in_file)
# lines = f.readlines()
# count = 0
# target = []
# output = []
# cur_target = None
# cur_output = None
# for line in lines:
#     if count % 2 == 0:
#         cur_target = line.strip().replace('[','').replace(']','').replace('\'','').split('| ')[1].split(', ')
#
#         # target.append(line.strip().replace('[','').replace(']','').replace('\'','').split('| ')[1].split(', '))
#         # print(target)
#     if count % 2 == 1:
#         cur_output = line.strip().replace('[','').replace(']','').replace('\'','').split('| ')[1].split(', ')
#         if cur_output != cur_target:
#             target.append(cur_target)
#             output.append(cur_output)
#         # output.append(line.strip().replace('[','').replace(']','').replace('\'','').split('| ')[1].split(', '))
#         # print(target)
#     count += 1
# f.close()
#
# total = 0
# dic_seq = {}
# for i in range(0,len(output)):
#     try:
#         output_be = output[i][level]
#         should_be = target[i][level]
#     except Exception as e:
#         output_be = output[i]
#         should_be = target[i]
#     combin = str(should_be)+'  -->  '+str(output_be)
#     if combin in dic_seq.keys():
#         dic_seq[combin] += 1
#     else:
#         dic_seq[combin] = 1
#     total += 1
#
#     # sort_dic_Seq = sorted(dic_seq, key=dic_seq.__getitem__)
#
# for k in sorted(dic_seq,key=dic_seq.__getitem__):
#     print(k, dic_seq[k])
#
# print('Done')

# '''
# 分析
# 与公司、证券、保险、票据等有关的民事纠纷  -->  合同、无因管理、不当得利纠纷
# 之间的案由
# '''
#
# punctuation = '!;:?"\'、，；'
# def removePunctuation(text):
#     text = re.sub(r'[{}]+'.format(punctuation),' ',text)
#     return text.strip()
#
# out_dir ='/home/dkb/workspace/Code/nlp_data/jieba'
# out_file = os.path.join(out_dir, "valid.json")
#
# f = open(out_file,'r',encoding='utf-8')
# data = []
#
# for line in f:
#     data.append(json.loads(line))
#
# f.close()
# print("read done")
# long_1 = set()
# long_2 = set()
# long_3 = set()
# for a in data:
#     if '侵权责任纠纷1' in a['label']:
#         law_this = removePunctuation(a['law'])
#         law_list = law_this.split(',')
#         law_1 = law_list[0].split('|')
#         law_2 = law_list[1].split('|')
#         law_3 = law_list[2].split('|')
#         if '' not in law_1:
#             long_1.update(law_1)
#             # long_1.append(law_1)
#         if '' not in law_2:
#             long_2.update(law_2)
#             # long_2.append(law_2)
#         if '' not in law_3:
#             long_3.update(law_3)
#             # long_3.append(law_3)
#         # print("one")
# print("long done")
#
# short_1 = set()
# short_2 = set()
# short_3 = set()
# for a in data:
#     if '人格权纠纷1' in a['label']:
#         law_this = removePunctuation(a['law'])
#         law_list = law_this.split(',')
#         law_1 = law_list[0].split('|')
#         law_2 = law_list[1].split('|')
#         law_3 = law_list[2].split('|')
#         if '' not in law_1:
#             short_1.update(law_1)
#             # long_1.append(law_1)
#         if '' not in law_2:
#             short_2.update(law_2)
#             # long_2.append(law_2)
#         if '' not in law_3:
#             short_3.update(law_3)
#             # long_3.append(law_3)
#         # print("one")
#
# print("short done")
#
# long_short_1 = long_1-short_1
# long_short_2 = long_2 - short_2
# long_short_3 = long_3 - short_3
#
# short_long_1 = short_1-long_1
# short_long_2 = short_2-long_2
# short_long_3 = short_3-long_3
#
# print("set calculate done")
#
# print(long_short_1)
# print(long_short_2)
# print(long_short_3)
#
# print(short_long_1)
# print(short_long_2)
# print(short_long_3)
#

'''
统计错误案例，一级
'''
# import pandas as pd
# import openpyxl as xl
#
# in_str = '侵权责任纠纷1  -->  物权纠纷 21'
# in_arr = in_str.split(" ")
#
# should_be = in_arr[0]
# out_be = in_arr[4]
#
#
# # hy_dic = {}
# out_file ='/home/dkb/workspace/Code/Code/2_json'
# # with open(out_file, 'r', encoding="utf-8") as load_f:
# #     hy_dic = json.load(load_f)
#
# f = open(out_file,'r',encoding='utf-8')
# data = []
#
# for line in f:
#     data.append(json.loads(line))
# f.close()
# print("read done")
#
# store_data = pd.DataFrame(columns = ["text", "correct", "wrong"]) #创建一个空的dataframe
#
# count = 0
# for a in data:
#     if out_be == a['output'][0] and should_be == a['target'][0]:
#         new = pd.DataFrame({'text': a['text'],
#                             'correct': str(a['target']),
#                             'wrong': str(a['output']),
#                             },index=[1])
#         store_data = store_data.append(new, ignore_index=True)
#         count += 1
# print(count)
# store_data.to_excel('/home/dkb/workspace/Code/Code/analyze/one_new_Seq'+'/|correct|:'+should_be+'|wrong|:'+out_be+'.xlsx', index=0)
# # store_data.to_csv('/home/dkb/workspace/Code/Code/analyze/one_new_Seq'+'/'+should_be+'.csv',mode='w', encoding = "utf-8")
#
# print("Done")