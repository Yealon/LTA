import json
import re

import h5py
import jieba
import numpy as np

# base_path = '/home/dkb/workspace/Code/nlp_data/jieba/'
#
# PAD = 0
# UNK = 1
# BOS = 2
# EOS = 3
#
# PAD_WORD = '<pad>'
# UNK_WORD = '<unk>'
# BOS_WORD = '<s>'
# EOS_WORD = '</s>'
#
# f2 = open(base_path + 'trg_level.dic', 'r+')
# onedic = eval(f2.readline())  # 读取的str转换为字典
#
# ff = open(base_path + 'trg.dic', 'r+')
# ff.write(PAD_WORD +" "+str(PAD))
# ff.write('\n')
#
# ff.write(UNK_WORD +" "+str(UNK))
# ff.write('\n')
#
# ff.write(BOS_WORD +" "+str(BOS))
# ff.write('\n')
#
# ff.write(EOS_WORD +" "+str(EOS))
# ff.write('\n')
#
# ii = 0
# for di in onedic:
#         line = di + " " + str(int(onedic[di]) + 3)
#         ff.write(line)
#         ff.write('\n')
#
#
# print("Done")
from utils.dict_utils import Dict

'''
构造基本target词典
'''
#
# f = open("/home/dkb/workspace/Code/nlp_data/jieba/trg_level.dic", 'r+')
# i = -4
# dic_json={}
#
# for line in f:
#     if i >= 0:
#         (key, value) = line.strip().split()
#         dic_json[key] = i
#         i += 1
#     else:
#         i += 1
#         continue
#
# dict_sorted=sorted(dic_json.items(), key=lambda d:d[1])
# results=[key for key,value in dict_sorted]
#
# results = np.array(results)
#
#
# dic_json = sorted(dic_json.items(),key=lambda x:x[1],reverse = False)
#
# dic_out = {}
# for i in range(0,len(dic_json)):
#     dic_out[dic_json[i][0]] = i
#
# file_name = '/home/dkb/workspace/Code/nlp_data/jieba/trg_label.json'
# json_str = json.dumps(dic_out, ensure_ascii=False)
#
# with open(file_name, 'w', encoding='utf-8') as f:
#     f.write(json_str)
# print("写入json文件：", dic_out)
'''
清洗 合同法
'''
#
# import re
# import string
# import jieba
#
#
#
# i = 0
#
# common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
#                             '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
# common_used_numerals = {}
# for key in common_used_numerals_tmp:
#     common_used_numerals[key] = common_used_numerals_tmp[key]
#
#
# def chinese2digits(uchars_chinese):
#     total = 0
#     r = 1  # 表示单位：个十百千...
#     for i in range(len(uchars_chinese) - 1, -1, -1):
#         val = common_used_numerals.get(uchars_chinese[i])
#         if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
#             if val > r:
#                 r = val
#                 total = total + val
#             else:
#                 r = r * val
#                 # total =total + r * x
#         elif val >= 10:
#             if val > r:
#                 r = val
#             else:
#                 r = r * val
#         else:
#             total = total + r * val
#     return total
#
#
# num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九',
#                         '十']
# more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']
#
#
# def changeChineseNumToArab(oriStr):
#     lenStr = len(oriStr)
#     aProStr = ''
#     if lenStr == 0:
#         return aProStr
#
#     hasNumStart = False
#     numberStr = ''
#     for idx in range(lenStr):
#         if oriStr[idx] in num_str_start_symbol:
#             if not hasNumStart:
#                 hasNumStart = True
#
#             numberStr += oriStr[idx]
#         else:
#             if hasNumStart:
#                 if oriStr[idx] in more_num_str_symbol:
#                     numberStr += oriStr[idx]
#                     continue
#                 else:
#                     numResult = str(chinese2digits(numberStr))
#                     numberStr = ''
#                     hasNumStart = False
#                     aProStr += numResult
#
#             aProStr += oriStr[idx]
#             pass
#
#     if len(numberStr) > 0:
#         resultNum = chinese2digits(numberStr)
#         aProStr += str(resultNum)
#
#     return aProStr
#
#
# def format_string(s):
#     return s.replace("\t", " ")
#
#
# # def detect_nowalk(dir_path,i):
# #     files = os.listdir(dir_path)
# #     for filename in files:
# #         if '0' in filename:
# #             print("get")
# #             i+=1
# #             print(i)
# #         print ("file:%s\n" % filename)
# #         next = os.path.join(dir_path, filename)
# #         print(next)
# #         if os.path.isdir(next):
# #             print ("file folds:%s\n" % filename)
# #             detect_nowalk(next,i)
# #
# # def process(our_data):
# #     m1 = map(lambda s: s.replace(' ', ''), our_data)
# #     return list(m1)
# # def is_chinese(uchar):
# #     if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
# #         return True
# #     else:
# #         return False
# # def format_str(content):
# #     content_str = ''
# #     for ii in content:
# #         if is_chinese(ii):
# #             content_str = content_str + ii
# #     return content_str
#
# # punctuation = '！，；：？!,;:?"\''
# # def removePunctuation(text):
# #     text = re.sub(r'[{}]+'.format(punctuation),'',text)
# #     return text.strip().lower()
# def remove_punctuation(line):
#     rule = re.compile(r"([^a-zA-Z0-9\u4e00-\u9fa5])")
#     line = rule.sub('', line)
#     return line
#
#
# def seg_depart(sentence):
#     # sentence = process(sentence.strip())
#     # sentence = format_str(sentence)
#
#     # 对文档中的每一行进行中文分词
#     sentence = sentence.strip()
#     sentence = remove_punctuation(sentence)
#     # sentence = sentence.translate(str.maketrans("", ""), string.punctuation)
#     # sentence = removePunctuation(sentence)
#
#     # print("正在分词")
#     sentence_depart = jieba.cut(sentence)
#     # 创建一个停用词列表
#     # stopwords = stopwordslist()
#     # 输出结果为outstr
#     outstr = ''
#     # 去停用词
#     i = 0
#     for word in sentence_depart:
#         # if word not in stopwords:
#         if word != '\t' and word != '\n':
#             i += 1
#             if i > 500:
#                 break
#             outstr += word
#             outstr += " "
#     # print(i)
#     return outstr
#
#
# '''
# 打开文件
# '''
# out_file = '/home/dkb/workspace/Code/nlp_data/qq'
# f = open(out_file)
# lines = f.readlines()
# count = 0
# target = []
# output = []
# law_num = 0
# dic1_law = {}
# for line in lines:
#
#     if line.startswith('第'):
#         rexagain = re.compile(r"(第[\u4E00-\u9FA5\\s]+条)")
#         law_list_chinese = re.findall(rexagain, line)
#         for lawindex, law_chinese in enumerate(law_list_chinese):
#             law_chinese.strip()
#             # re.sub('[\r\n\t]', '', law_chinese)
#             law_chinese = law_chinese.replace('第', '')
#             law_chinese = law_chinese.replace('条', '')
#             law_number_str = changeChineseNumToArab(law_chinese)
#             # 数据中有law_number = '114和248'情况，再次提取数字
#             number_list = re.findall(r'(\d+)', law_number_str)
#             for law_number in number_list:
#                 law_num = law_number
#             continue
#
#     else:
#         outstr = seg_depart(line)
#         dic1_law[law_num] = outstr
#         law_num = -1
#
# file_name = '/home/dkb/workspace/Code/nlp_data/qq.json'
# json_str = json.dumps(dic1_law, ensure_ascii=False,indent=4)
#
# with open(file_name, 'w', encoding='utf-8') as f:
#     f.write(json_str)
# print("写入json文件：", dic1_law)
# # 验证
# # with open(file_name, 'r', encoding="utf-8") as load_f:
# #     text = json.load(load_f)

'''
law 词典
'''
# import argparse
# from formatter.Seq2Seq import dict as dict
# import os
# # basepath = '/home/dkb/workspace/Code/pt/data/data/'
#
# # **Preprocess Options**
# parser = argparse.ArgumentParser(description='preprocess.py')
#
# parser.add_argument('-config', help="Read options from this file")
#
# parser.add_argument('-train_src',
#                     default=['/home/dkb/workspace/Code/nlp_data/vocab_law/ht.json',
#                              '/home/dkb/workspace/Code/nlp_data/vocab_law/hy.json',
#                              '/home/dkb/workspace/Code/nlp_data/vocab_law/qq.json'],
#                     help="Path to the training source data")
#
# parser.add_argument('-save_data',
#                     default='/home/dkb/workspace/Code/nlp_data/vocab_law/save_test',
#                     help="Output file for the prepared data")
#
# parser.add_argument('-src_vocab_size', type=int, default=3000000,
#                     help="Size of the source vocabulary")
#
# parser.add_argument('-src_vocab',
#                     default=None,
#                     help="Path to an existing source vocabulary")
#
# parser.add_argument('-src_length', type=int, default=200,
#                     help="Maximum source sequence length")
# parser.add_argument('-tgt_length', type=int, default=1000,
#                     help="Maximum target sequence length")
# parser.add_argument('-seed',       type=int, default=10,
#                     help="Random seed")
#
# parser.add_argument('-lower', default=False,
#                     action='store_true', help='lowercase data')
# parser.add_argument('-char', default=False,
#                     action='store_true', help='replace unk with char')
# parser.add_argument('-share', default=False, action='store_true',
#                     help='share the vocabulary between source and target')
#
# parser.add_argument('-report_every', type=int, default=1000000,
#                     help="Report status every this many sentences")
#
# opt = parser.parse_args()
#
#
# def makeVocabulary(filename, size, char=False):
#     vocab = dict.Dict([dict.PAD_WORD, dict.UNK_WORD,
#                        dict.BOS_WORD, dict.EOS_WORD], lower=opt.lower)
#     if char:
#         vocab.addSpecial(dict.SPA_WORD)
#
#     lengths = []
#     words_len=[]
#
#     if type(filename) == list:
#         for _filename in filename:
#             with open(_filename) as f:
#                 temp_dic = json.load(f)
#                 for sent in temp_dic.values():
#                     count = 0
#                     for word in sent.strip().split():
#                         lengths.append(len(word))
#                         count += 1
#                         if char:
#                             for ch in word:
#                                 vocab.add(ch)
#                         else:
#                             vocab.add(word + " ")
#                     words_len.append(count)
#     else:
#         with open(filename) as f:
#             # for sent in f:
#             for sent in f.readlines():
#                 for word in sent.strip().split():
#                     lengths.append(len(word))
#                     if char:
#                         for ch in word:
#                             vocab.add(ch)
#                     else:
#                         vocab.add(word+" ")
#
#     print('max: %d, min: %d, avg: %.2f' %
#           (max(lengths), min(lengths), sum(lengths)/len(lengths)))
#
#     print('word: max: %d, min: %d, avg: %.2f' %
#           (max(words_len), min(words_len), sum(words_len)/len(words_len)))
#
#     originalSize = vocab.size()
#     vocab = vocab.prune(size)
#     print('Created dictionary of size %d (pruned from %d)' %
#           (vocab.size(), originalSize))
#
#     return vocab
#
#
# def initVocabulary(name, dataFile, vocabFile, vocabSize, char=False):
#     vocab = None
#     if vocabFile is not None:
#         # If given, load existing word dictionary.
#         print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
#         vocab = dict.Dict()
#         vocab.loadFile(vocabFile)
#         print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')
#
#     if vocab is None:
#         # If a dictionary is still missing, generate it.
#         print('Building ' + name + ' vocabulary...')
#         vocab = makeVocabulary(dataFile, vocabSize, char=char)
#
#     return vocab
#
#
# def saveVocabulary(name, vocab, file):
#     print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
#     # vocab.writeFile(file)
#
#
#
# def main():
#
#     dicts = {}
#
#     dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,# opt.src_vocab
#                                   opt.src_vocab_size)
#     # saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
#
#     if opt.src_vocab is None:
#         saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
#
#     print('Saving data to \'' + opt.save_data + '.train.pt\'...')
#
# if __name__ == "__main__":
#     main()
#
# print("Done")
'''
清洗label meaning
# '''
# l = []
# # list 转成Json格式数据
# def listToJson(lst):
#     import json
#     import numpy as np
#     keys = [str(x) for x in np.arange(len(lst))]
#     list_json = dict(zip(keys, lst))
#     str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
#     return str_json
#
# def is_chinese(uchar):
#     if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
#         return True
#     else:
#         return False
#
# def format_str(content):
#     content_str = ''
#     for i in content:
#         if is_chinese(i):
#             content_str = content_str + i
#     return content_str
#
# def save_func(label,meaning,manage,law_range,notice):
#     # todo: save json
#     if len(label) != 0:
#         print("标签 -- >" + label)
#         print("【释义】"+meaning)
#         print("【管辖】"+manage)
#         print("【法律适用】"+law_range)
#         print("【确定该案由应当注意的问题】"+notice)
#         dic1_law = {}
#         # dic1_law['label'] = format_str(label)
#         # dic1_law['meaning'] = format_str(meaning)
#         # dic1_law['manage'] = format_str(manage)
#         # dic1_law['law_range'] = format_str(law_range)
#         # dic1_law['notice'] = format_str(notice)
#
#         dic1_law['label'] = label
#         dic1_law['meaning'] = meaning
#         dic1_law['manage'] = manage
#         dic1_law['law_range'] = law_range
#         dic1_law['notice'] = notice
#
#         l.append(dic1_law)
#         # file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/result.json'
#         # json_str = json.dumps(dic1_law, ensure_ascii=False)
#         # with open(file_name, 'a', encoding='utf-8') as f:
#         #     f.write(json_str)
#         # print("写入json文件：", dic1_law)
#     else:
#         return
#
# out_file = '/home/dkb/workspace/Code/nlp_data/labe_mean/out'
# f = open(out_file)
# lines = f.readlines()
# count = 1
# target = []
# output = []
#
# meaning_flag = 0
# manage_flag = 0
# law_range_flag = 0
# notice_flag = 0
# label = ''
# meaning = ''
# manage = ''
# law_range = ''
# notice = ''
#
#
# for line in lines:
#
#     if count < 10:
#         if str(count) == line[0] and '.' == line[1]:
#             count += 1
#             if count == 2 : pass
#             else: save_func(label,meaning,manage,law_range,notice)
#             label = line
#             # label = format_str(line)
#             meaning = ''
#             manage = ''
#             law_range = ''
#             notice = ''
#         else:
#             if line.startswith('【释'):
#                 meaning_flag = 1
#                 manage_flag = 0
#                 law_range_flag = 0
#                 notice_flag = 0
#                 continue
#
#             elif line.startswith('【管'):
#                 meaning_flag = 0
#                 manage_flag = 1
#                 law_range_flag = 0
#                 notice_flag = 0
#                 continue
#
#             elif line.startswith('【法'):
#                 meaning_flag = 0
#                 manage_flag = 0
#                 law_range_flag = 1
#                 notice_flag = 0
#                 continue
#
#             elif line.startswith('【确定'):
#                 meaning_flag = 0
#                 manage_flag = 0
#                 law_range_flag = 0
#                 notice_flag = 1
#                 continue
#             else:
#                 if line.startswith('【第') and len(label) != 0:
#                     save_func(label, meaning, manage, law_range, notice)
#                     meaning_flag = 0
#                     manage_flag = 0
#                     law_range_flag = 0
#                     notice_flag = 0
#                     label = ''
#                     meaning = ''
#                     manage = ''
#                     law_range = ''
#                     notice = ''
#                     continue
#                 elif meaning_flag == 1:
#                     meaning += line
#                 elif manage_flag == 1:
#                     manage += line
#                 elif law_range_flag == 1:
#                     law_range += line
#                 elif notice_flag == 1:
#                     notice += line
#
#     elif count < 100:
#         if str(int(count / 10)) == line[0] and str(count % 10) == line[1] and '.' == line[2]:
#             count += 1
#             save_func(label,meaning,manage,law_range,notice)
#             label = line
#             # label = format_str(line)
#             meaning = ''
#             manage = ''
#             law_range = ''
#             notice = ''
#         else:
#             if line.startswith('【释'):
#                 meaning_flag = 1
#                 manage_flag = 0
#                 law_range_flag = 0
#                 notice_flag = 0
#                 continue
#
#             elif line.startswith('【管'):
#                 meaning_flag = 0
#                 manage_flag = 1
#                 law_range_flag = 0
#                 notice_flag = 0
#                 continue
#
#             elif line.startswith('【法'):
#                 meaning_flag = 0
#                 manage_flag = 0
#                 law_range_flag = 1
#                 notice_flag = 0
#                 continue
#
#             elif line.startswith('【确定'):
#                 meaning_flag = 0
#                 manage_flag = 0
#                 law_range_flag = 0
#                 notice_flag = 1
#                 continue
#             else:
#                 if line.startswith('【第') and len(label) != 0:
#                     save_func(label, meaning, manage, law_range, notice)
#                     meaning_flag = 0
#                     manage_flag = 0
#                     law_range_flag = 0
#                     notice_flag = 0
#                     label = ''
#                     meaning = ''
#                     manage = ''
#                     law_range = ''
#                     notice = ''
#                     continue
#                 elif meaning_flag == 1:
#                     meaning += line
#                 elif manage_flag == 1:
#                     manage += line
#                 elif law_range_flag == 1:
#                     law_range += line
#                 elif notice_flag == 1:
#                     notice += line
#
#     elif count < 1000 and len(line) >= 4:
#         if str(int(count / 100)) == line[0] and str(int(count / 10) % 10) == line[1] and str(count % 10) == line[2] and '.' == line[3]:
#             count += 1
#             save_func(label,meaning,manage,law_range,notice)
#             # label = format_str(line)
#             label = line
#             meaning = ''
#             manage = ''
#             law_range = ''
#             notice = ''
#         else:
#             if line.startswith('【释'):
#                 meaning_flag = 1
#                 manage_flag = 0
#                 law_range_flag = 0
#                 notice_flag = 0
#                 continue
#
#             elif line.startswith('【管'):
#                 meaning_flag = 0
#                 manage_flag = 1
#                 law_range_flag = 0
#                 notice_flag = 0
#                 continue
#
#             elif line.startswith('【法'):
#                 meaning_flag = 0
#                 manage_flag = 0
#                 law_range_flag = 1
#                 notice_flag = 0
#                 continue
#
#             elif line.startswith('【确定'):
#                 meaning_flag = 0
#                 manage_flag = 0
#                 law_range_flag = 0
#                 notice_flag = 1
#                 continue
#             else:
#                 if line.startswith('【第') and len(label) != 0:
#                     save_func(label, meaning, manage, law_range, notice)
#                     meaning_flag = 0
#                     manage_flag = 0
#                     law_range_flag = 0
#                     notice_flag = 0
#                     label = ''
#                     meaning = ''
#                     manage = ''
#                     law_range = ''
#                     notice = ''
#                     continue
#                 elif meaning_flag == 1:
#                     meaning += line
#                 elif manage_flag == 1:
#                     manage += line
#                 elif law_range_flag == 1:
#                     law_range += line
#                 elif notice_flag == 1:
#                     notice += line
#
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/nowash.json'
# l_str = listToJson(l)
# json_str = l_str
# with open(file_name, 'a', encoding='utf-8') as f:
#     f.write(json_str)
# print("写入json文件：", json_str)

'''
二级案由
'''
# l = []
# # list 转成Json格式数据
# def listToJson(lst):
#     import json
#     import numpy as np
#     keys = [str(x) for x in np.arange(len(lst))]
#     list_json = dict(zip(keys, lst))
#     str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
#     return str_json
# common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
#                             '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
# common_used_numerals = {}
# for key in common_used_numerals_tmp:
#     common_used_numerals[key] = common_used_numerals_tmp[key]
#
#
# def chinese2digits(uchars_chinese):
#     total = 0
#     r = 1  # 表示单位：个十百千...
#     for i in range(len(uchars_chinese) - 1, -1, -1):
#         val = common_used_numerals.get(uchars_chinese[i])
#         if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
#             if val > r:
#                 r = val
#                 total = total + val
#             else:
#                 r = r * val
#                 # total =total + r * x
#         elif val >= 10:
#             if val > r:
#                 r = val
#             else:
#                 r = r * val
#         else:
#             total = total + r * val
#     return total
#
#
# num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九',
#                         '十']
# more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']
#
#
# def changeChineseNumToArab(oriStr):
#     lenStr = len(oriStr)
#     aProStr = ''
#     if lenStr == 0:
#         return aProStr
#
#     hasNumStart = False
#     numberStr = ''
#     for idx in range(lenStr):
#         if oriStr[idx] in num_str_start_symbol:
#             if not hasNumStart:
#                 hasNumStart = True
#
#             numberStr += oriStr[idx]
#         else:
#             if hasNumStart:
#                 if oriStr[idx] in more_num_str_symbol:
#                     numberStr += oriStr[idx]
#                     continue
#                 else:
#                     numResult = str(chinese2digits(numberStr))
#                     numberStr = ''
#                     hasNumStart = False
#                     aProStr += numResult
#
#             aProStr += oriStr[idx]
#             pass
#
#     if len(numberStr) > 0:
#         resultNum = chinese2digits(numberStr)
#         aProStr += str(resultNum)
#
#     return aProStr
#
#
# def is_chinese(uchar):
#     if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
#         return True
#     else:
#         return False
#
# def format_str(content):
#     content_str = ''
#     for i in content:
#         if is_chinese(i):
#             content_str = content_str + i
#     return content_str
#
# def save_func(label,meaning,manage,law_range,notice):
#     # todo: save json
#     if len(label) != 0:
#         print("标签 -- >" + label)
#         print("【释义】"+meaning)
#         # print("【管辖】"+manage)
#         # print("【法律适用】"+law_range)
#         # print("【确定该案由应当注意的问题】"+notice)
#         dic1_law = {}
#         # dic1_law['label'] = format_str(label)
#         # dic1_law['meaning'] = format_str(meaning)
#         # dic1_law['manage'] = format_str(manage)
#         # dic1_law['law_range'] = format_str(law_range)
#         # dic1_law['notice'] = format_str(notice)
#
#         dic1_law['label'] = label
#         dic1_law['meaning'] = meaning
#         # dic1_law['manage'] = manage
#         # dic1_law['law_range'] = law_range
#         # dic1_law['notice'] = notice
#
#         l.append(dic1_law)
#         # file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/result.json'
#         # json_str = json.dumps(dic1_law, ensure_ascii=False)
#         # with open(file_name, 'a', encoding='utf-8') as f:
#         #     f.write(json_str)
#         # print("写入json文件：", dic1_law)
#     else:
#         return
#
# out_file = '/home/dkb/workspace/Code/nlp_data/labe_mean/out'
# f = open(out_file)
# lines = f.readlines()
# count = 1
# target = []
# output = []
#
# meaning_flag = 0
#
# label = ''
# meaning = ''
# manage = ''
# law_range = ''
# notice = ''
#
#
# for line in lines:
#     line = changeChineseNumToArab(line)
#     if count < 10:
#         if str(count) == line[0] and '、' == line[1]:
#             count += 1
#             if count == 2 : pass
#             else: save_func(label,meaning,manage,law_range,notice)
#             label = format_str(line)
#             meaning_flag = 1
#             # label = format_str(line)
#             meaning = ''
#
#         else:
#             if meaning_flag == 1:
#                 meaning = format_str(line)
#                 meaning_flag = 0
#                 continue
#
#             else:
#                 continue
#
#     elif count < 100:
#         if str(int(count / 10)) == line[0] and str(count % 10) == line[1] and '、' == line[2]:
#             count += 1
#             save_func(label,meaning,manage,law_range,notice)
#             label = format_str(line)
#             meaning_flag = 1
#             # label = format_str(line)
#             meaning = ''
#         else:
#             if meaning_flag == 1:
#                 meaning = format_str(line)
#                 meaning_flag = 0
#                 continue
#             else:
#                 continue
# save_func(label, meaning, manage, law_range, notice)
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/second.json'
# l_str = listToJson(l)
# json_str = l_str
# with open(file_name, 'a', encoding='utf-8') as f:
#     f.write(json_str)
# print("写入json文件：", json_str)
# #


'''
第一级案由
'''
# l = []
# # list 转成Json格式数据
# def listToJson(lst):
#     import json
#     import numpy as np
#     keys = [str(x) for x in np.arange(len(lst))]
#     list_json = dict(zip(keys, lst))
#     str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
#     return str_json
#
# def is_chinese(uchar):
#     if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
#         return True
#     else:
#         return False
#
# def format_str(content):
#     content_str = ''
#     for i in content:
#         if is_chinese(i):
#             content_str = content_str + i
#     return content_str
#
# def save_func(label,meaning,manage,law_range,notice):
#     # todo: save json
#     if len(label) != 0:
#         print("标签 -- >" + label)
#         print("【释义】"+meaning)
#         # print("【管辖】"+manage)
#         # print("【法律适用】"+law_range)
#         # print("【确定该案由应当注意的问题】"+notice)
#         dic1_law = {}
#         # dic1_law['label'] = format_str(label)
#         # dic1_law['meaning'] = format_str(meaning)
#         # dic1_law['manage'] = format_str(manage)
#         # dic1_law['law_range'] = format_str(law_range)
#         # dic1_law['notice'] = format_str(notice)
#
#         dic1_law['label'] = label
#         dic1_law['meaning'] = meaning
#         # dic1_law['manage'] = manage
#         # dic1_law['law_range'] = law_range
#         # dic1_law['notice'] = notice
#
#         l.append(dic1_law)
#         # file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/result.json'
#         # json_str = json.dumps(dic1_law, ensure_ascii=False)
#         # with open(file_name, 'a', encoding='utf-8') as f:
#         #     f.write(json_str)
#         # print("写入json文件：", dic1_law)
#     else:
#         return
#
# out_file = '/home/dkb/workspace/Code/nlp_data/labe_mean/data'
# f = open(out_file)
# lines = f.readlines()
# count = 1
# target = []
# output = []
#
# meaning_flag = 0
#
# label = ''
# meaning = ''
# manage = ''
# law_range = ''
# notice = ''
#
#
# for line in lines:
#     count += 1
#     if count % 2 == 0 :
#         label = format_str(line)
#     else:
#         meaning = format_str(line)
#         save_func(label,meaning,manage,law_range,notice)
#
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/first.json'
# l_str = listToJson(l)
# json_str = l_str
# with open(file_name, 'a', encoding='utf-8') as f:
#     f.write(json_str)
# print("写入json文件：", json_str)
# #
#

'''
测试 读json
# '''
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/result.json'
#
# with open(file_name,'r',encoding='utf8')as fp:
#     json_data = json.load(fp)
#

'''
合并词典
'''
## list 转成Json格式数据
# def listToJson(lst):
#     import json
#     import numpy as np
#     keys = [str(x) for x in np.arange(len(lst))]
#     list_json = dict(zip(keys, lst))
#     str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
#     return str_json
#
# first_dic = {}
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/first.json'
# with open(file_name, 'r', encoding="utf-8") as load_f:
#     first_dic = json.load(load_f)
#
# second_dic = {}
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/second.json'
# with open(file_name, 'r', encoding="utf-8") as load_f:
#     second_dic = json.load(load_f)
#
# third_dic = {}
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/result.json'
# with open(file_name, 'r', encoding="utf-8") as load_f:
#     third_dic = json.load(load_f)
# l = []
#
#
# for k in first_dic.keys() :
#     merge_dic = {}
#     merge_dic['label'] = first_dic[k]['label']
#     merge_dic['meaning'] = first_dic[k]['meaning']
#     l.append(merge_dic)
#
#
# for k in second_dic.keys() :
#     merge_dic = {}
#     merge_dic['label'] = second_dic[k]['label']
#     merge_dic['meaning'] = second_dic[k]['meaning']
#     l.append(merge_dic)
#
#
# for k in third_dic.keys() :
#     merge_dic = {}
#     merge_dic['label'] = third_dic[k]['label']
#     merge_dic['meaning'] = third_dic[k]['meaning']
#     l.append(merge_dic)

# print(listToJson(l))
#
# def is_chinese(uchar):
#     if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
#         return True
#     else:
#         return False
#
# def format_str(content):
#     content_str = ''
#     for i in content:
#         if is_chinese(i):
#             content_str = content_str + i
#     return content_str
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
# l = []
# count = 0
#
# first_dic = {}
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/first.json'
# with open(file_name, 'r', encoding="utf-8") as load_f:
#     first_dic = json.load(load_f)
#
# for k in onedic_name_label.keys() :
#     for k1 in first_dic.keys():
#         if format_str(k) == first_dic[k1]['label']:
#             merge_dic = {}
#             merge_dic['index'] = count
#             merge_dic['label'] = k
#             merge_dic['meaning'] = first_dic[k1]['meaning']
#             count += 1
#             l.append(merge_dic)
#
# second_dic = {}
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/second.json'
# with open(file_name, 'r', encoding="utf-8") as load_f:
#     second_dic = json.load(load_f)
#
# for k in twodic_name_label.keys() :
#     for k1 in second_dic.keys():
#         if format_str(k) == second_dic[k1]['label']:
#             merge_dic = {}
#             merge_dic['index'] = count
#             merge_dic['label'] = k
#             merge_dic['meaning'] = second_dic[k1]['meaning']
#             count += 1
#             l.append(merge_dic)
#
# third_dic = {}
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/result.json'
# with open(file_name, 'r', encoding="utf-8") as load_f:
#     third_dic = json.load(load_f)
#
# for k in threedic_name_label.keys():
#     for k1 in third_dic.keys():
#         if k == third_dic[k1]['label']:
#             merge_dic = {}
#             merge_dic['index'] = count
#             merge_dic['label'] = k
#             merge_dic['meaning'] = third_dic[k1]['meaning']
#             count += 1
#             l.append(merge_dic)
# file_save = '/home/dkb/workspace/Code/nlp_data/labe_mean/washed_total.json'
#
# l_str = listToJson(l)
# json_str = l_str
# with open(file_save, 'a', encoding='utf-8') as f:
#     f.write(json_str)
# print("写入json文件：", json_str)

'''
统计label描述长度
'''
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/washed_total.json'
#
# with open(file_name,'r',encoding='utf8')as fp:
#     json_data = json.load(fp)
# count = 0
# length = 0
# for k in json_data:
#     count += 1
#     length += len(json_data[k]['meaning'])
#
# average = length / count


'''
对label的描述分词，统计邻接矩阵
'''
#
# # def seg_depart(sentence):
# #     # sentence = process(sentence.strip())
# #     # sentence = format_str(sentence)
# #
# #     # 对文档中的每一行进行中文分词
# #     sentence = sentence.strip()
# #
# #     # sentence = sentence.translate(str.maketrans("", ""), string.punctuation)
# #     # sentence = removePunctuation(sentence)
# #
# #     # print("正在分词")
# #     sentence_depart = jieba.cut(sentence)
# #     # 创建一个停用词列表
# #     # stopwords = stopwordslist()
# #     # 输出结果为outstr
# #     outstr = ''
# #     # 去停用词
# #     i = 0
# #     for word in sentence_depart:
# #         # if word not in stopwords:
# #         if word != '\t' and word != '\n':
# #             i += 1
# #             if i > 500:
# #                 break
# #             outstr += word
# #             outstr += " "
# #     # print(i)
# #     return outstr
# #
# #
# # file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/washed_total.json'
# #
# # with open(file_name,'r',encoding='utf8')as fp:
# #     json_data = json.load(fp)
# #
# # dic1_law = {}
# # dic2_label = {}
# # law_num = 0
# # for k in json_data:
# #     outstr = seg_depart(json_data[k]['meaning'])
# #     dic1_law[law_num] = outstr
# #     dic2_label[law_num] = json_data[k]['label']
# #     law_num += 1
# #
# # file_name = '/home/dkb/workspace/Code/nlp_data/vocab_label/label_meaning'
# # file_name_1 = '/home/dkb/workspace/Code/nlp_data/vocab_label/index_label'
# #
# #
# # json_str = json.dumps(dic1_law, ensure_ascii=False,indent=4)
# #
# # json_str1 = json.dumps(dic2_label, ensure_ascii=False,indent=4)
# #
# # with open(file_name, 'w', encoding='utf-8') as f:
# #     f.write(json_str)
# # print("写入json文件：", dic1_law)
# #
# # with open(file_name_1, 'w', encoding='utf-8') as f:
# #     f.write(json_str1)
# # print("写入json文件：", dic2_label)
# #
#
#
# # 验证
# # with open(file_name, 'r', encoding="utf-8") as load_f:
# #     text = json.load(load_f)
#
#
# import argparse
# from formatter.Seq2Seq import dict as dict
# import os
# # basepath = '/home/dkb/workspace/Code/pt/data/data/'
#
# # **Preprocess Options**
# parser = argparse.ArgumentParser(description='preprocess.py')
#
# parser.add_argument('-config', help="Read options from this file")
#
# parser.add_argument('-train_src',
#                     default=['/home/dkb/workspace/Code/nlp_data/vocab_label/label_meaning'],
#                     help="Path to the training source data")
#
# parser.add_argument('-save_data',
#                     default='/home/dkb/workspace/Code/nlp_data/vocab_label/label_vocab',
#                     help="Output file for the prepared data")
#
# parser.add_argument('-src_vocab_size', type=int, default=3000000,
#                     help="Size of the source vocabulary")
#
# parser.add_argument('-src_vocab',
#                     default=None,
#                     help="Path to an existing source vocabulary")
#
# parser.add_argument('-src_length', type=int, default=200,
#                     help="Maximum source sequence length")
# parser.add_argument('-tgt_length', type=int, default=1000,
#                     help="Maximum target sequence length")
# parser.add_argument('-seed',       type=int, default=10,
#                     help="Random seed")
#
# parser.add_argument('-lower', default=False,
#                     action='store_true', help='lowercase data')
# parser.add_argument('-char', default=False,
#                     action='store_true', help='replace unk with char')
# parser.add_argument('-share', default=False, action='store_true',
#                     help='share the vocabulary between source and target')
#
# parser.add_argument('-report_every', type=int, default=1000000,
#                     help="Report status every this many sentences")
#
# opt = parser.parse_args()
#
#
# def makeVocabulary(filename, size, char=False):
#     vocab = dict.Dict([dict.PAD_WORD, dict.UNK_WORD,
#                        dict.BOS_WORD, dict.EOS_WORD], lower=opt.lower)
#
#     lengths = []
#     words_len=[]
#
#     if type(filename) == list:
#         for _filename in filename:
#             with open(_filename) as f:
#                 temp_dic = json.load(f)
#                 for sent in temp_dic.values():
#                     count = 0
#                     for word in sent.strip().split():
#                         lengths.append(len(word))
#                         count += 1
#                         if char:
#                             for ch in word:
#                                 vocab.add(ch)
#                         else:
#                             vocab.add(word + " ")
#                     words_len.append(count)
#     else:
#         with open(filename) as f:
#             # for sent in f:
#             for sent in f.readlines():
#                 for word in sent.strip().split():
#                     lengths.append(len(word))
#                     if char:
#                         for ch in word:
#                             vocab.add(ch)
#                     else:
#                         vocab.add(word+" ")
#
#     print('max: %d, min: %d, avg: %.2f' %
#           (max(lengths), min(lengths), sum(lengths)/len(lengths)))
#
#     print('word: max: %d, min: %d, avg: %.2f' %
#           (max(words_len), min(words_len), sum(words_len)/len(words_len)))
#
#     originalSize = vocab.size()
#     vocab = vocab.prune(size)
#     print('Created dictionary of size %d (pruned from %d)' %
#           (vocab.size(), originalSize))
#
#     return vocab
#
#
# def initVocabulary(name, dataFile, vocabFile, vocabSize, char=False):
#     vocab = None
#     if vocabFile is not None:
#         # If given, load existing word dictionary.
#         print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
#         vocab = dict.Dict()
#         vocab.loadFile(vocabFile)
#         print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')
#
#     if vocab is None:
#         # If a dictionary is still missing, generate it.
#         print('Building ' + name + ' vocabulary...')
#         vocab = makeVocabulary(dataFile, vocabSize, char=char)
#
#     return vocab
#
#
# def saveVocabulary(name, vocab, file):
#     print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
#     vocab.writeFile(file)
#
#
#
# def main():
#
#     dicts = {}
#
#     dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,# opt.src_vocab
#                                   opt.src_vocab_size)
#     # saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
#
#     if opt.src_vocab is None:
#         saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
#
#     print('Saving data to \'' + opt.save_data + '.train.pt\'...')
#
# if __name__ == "__main__":
#     main()
'''
构建临接边
'''
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
# # meaning的字典
#
# # fff1 = open("/home/dkb/workspace/Code/nlp_data/vocab_label/label_meaning", 'r+')
# # index_meaning = eval(fff1.read())  # 读取的str转换为字典
# # meaning_index = {v: k for k, v in index_meaning.items()}
# # fff1.close()
#
# fff2 = open("/home/dkb/workspace/Code/nlp_data/vocab_label/index_label", 'r+')
# index_label = eval(fff2.read())  # 读取的str转换为字典
# label_index = {v: k for k, v in index_label.items()}
# fff2.close()
#
# for k, label in index_label.items():
#         cur_child = Hie_dic[label]
#         last_child = -1
#         last_location = -1
#
#         for i, child in enumerate(cur_child):
#             if child == -1:
#                 last_location = i-2
#                 last_child = cur_child[last_location]
#                 break
#
#         with open("/home/dkb/workspace/Code/nlp_data/vocab_label/edge_3.txt", "a") as f:  # 格式化字符串还能这么用！
#             f.write(label_index[label] + " "
#                     + label_index[label] + "\n")
#
#         if last_location == 0:
#            print("label:"+label+",father:"+first_dic_index_name[last_child])
#            with open("/home/dkb/workspace/Code/nlp_data/vocab_label/edge_3.txt", "a") as f:  # 格式化字符串还能这么用！
#                f.write( label_index[first_dic_index_name[last_child]]+ " "
#                        +label_index[label]  + "\n")
#
#         elif last_location == 1:
#             print("label:" + label + ",father:" + second_dic_index_name[last_child])
#             with open("/home/dkb/workspace/Code/nlp_data/vocab_label/edge_3.txt", "a") as f:  # 格式化字符串还能这么用！
#                 f.write(label_index[second_dic_index_name[last_child]] + " "
#                         + label_index[label] + "\n")
#
#         elif last_location == 2:
#             print("label:" + label + ",father:" + third_dic_index_name[last_child])
#             with open("/home/dkb/workspace/Code/nlp_data/vocab_label/edge_3.txt", "a") as f:  # 格式化字符串还能这么用！
#                 f.write(label_index[third_dic_index_name[last_child]] + " "
#                         + label_index[label] +"\n")
#         else:
#             print("label:" + label + ",father: root")
# '''
# 构成节点文件
# '''
# fff2 = open("/home/dkb/workspace/Code/nlp_data/vocab_label/index_label", 'r+')
# index_label = eval(fff2.read())  # 读取的str转换为字典
# label_index = {v: k for k, v in index_label.items()}
# fff2.close()
#
# for k, label in index_label.items():
#     candidate = np.zeros(len(index_label))
#     candidate[int(k)] = 1
#     # print(candidate)
#     with open("/home/dkb/workspace/Code/nlp_data/vocab_label/node.txt", "a") as f:  # 格式化字符串还能这么用！
#                 f.write(k + " " +
#                         "1"
#                         + " " + label +"\n")
#


'''
构建邻接矩阵等要素，torch 尝试,同graph utils
'''
# import scipy.sparse as sp  # scipy.sparse稀疏矩阵包，且用sp名称等价
# import torch
#
#
# def encode_onehot(labels):  # 将标签转换为one-hot编码形式
#     classes = set(labels)  # set()函数就是提取输入的组成元素，且进行不重复无序的排列输出
#     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
#                     enumerate(classes)}
#     # 这一句主要功能就是进行转化成dict字典数据类型，且键为元素，值为one-hot编码
#     # enumerate()将可遍历对象组合成一个含数据下标和数据的索引序列
#     # for i,c in XXX 将XXX序列进行循环遍历赋给(i,c)，这里是i得数据下标，c得数据
#     # len()返回元素的个数，np.identity()函数创建对角矩阵，返回主对角线元素为1，其余元素为0
#     # 矩阵[i,:]是仅保留第一维度的下标i的元素和第二维度所有元素，直白来看就是提取了矩阵的第i行
#     # {}生成了字典，c:xxx 是字典的形式，c作为键，xxx作为值，在for in循环下进行组成字典
#     # c:xxx在for in前面这种结构我还是没查到所以然，只是python跑出来看到了结果明白了怎么运行
#     labels_onehot = np.array(list(map(classes_dict.get, labels)),
#                              dtype=np.int32)
#     # array()是numpy是数组格式，dtype是数组元素的数据类型，list()用于将元组转换为列表
#     # map(function, iterable)是对指定序列iterable中的每一个元素调用function函数，
#     # 根据提供的函数对指定序列做映射，返回包含每次function函数返回值的新列表
#     # 这句话的意思就是将输入一一对应one-hot编码进行输出
#     return labels_onehot
#
#
# idx_features_labels = np.genfromtxt("/home/dkb/workspace/Code/nlp_data/vocab_label/test_node.txt",
#                                     dtype=np.dtype(str))
# aaa = idx_features_labels[:, 1:-1]
# features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
# # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
# # [:, 1:-1]是指行全部选中、列选取第二列至倒数第二列，float32类型
# # 这句功能就是去除论文样本的编号和类别，留下每篇论文的词向量，并将稀疏矩阵编码压缩
# labels = encode_onehot(idx_features_labels[:, -1])
# # 提取论文样本的类别标签，并将其转换为one-hot编码形式
#
#
# # 构建图
# def normalize(mx):  # 这里是计算D^-1A，而不是计算论文中的D^-1/2AD^-1/2
#     """行规范化稀疏矩阵"""
#     # 这个函数思路就是在邻接矩阵基础上转化出度矩阵，并求D^-1A随机游走归一化拉普拉斯算子
#     # 函数实现的规范化方法是将输入左乘一个D^-1算子，就是将矩阵每行进行归一化
#
#     rowsum = np.array(mx.sum(1))
#     # .sum(1)计算输入矩阵的第1维度求和的结果，这里是将二维矩阵的每一行元素求和
#     r_inv = np.power(rowsum, -1).flatten()
#     # rowsum数组元素求-1次方，flatten()返回一个折叠成一维的数组（默认按行的方向降维）
#     # 求倒数
#
#     r_inv[np.isinf(r_inv)] = 0.
#     # isinf()测试元素是否为正无穷或负无穷,若是则返回真，否则是假，最后返回一个与输入形状相同的布尔数组
#     # 如果某一行全为0，则倒数r_inv算出来会等于无穷大，将这些行的r_inv置为0
#     # 这句就是将数组中无穷大的元素置0处理
#     r_mat_inv = sp.diags(r_inv)  # 稀疏对角矩阵
#     # 构建对角元素为r_inv的对角矩阵
#     # sp.diags()函数根据给定的对象创建对角矩阵，对角线上的元素为给定对象中的元素
#     mx = r_mat_inv.dot(mx)  # 点积
#     # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘
#     # 所谓矩阵点积就是两个矩阵正常相乘而已
#     return mx  # D^-1A
#
#
# def accuracy(output, labels):  # 准确率，此函数可参考学习借鉴复用
#     preds = output.max(1)[1].type_as(labels)
#     # max(1)返回每一行最大值组成的一维数组和索引,output.max(1)[1]表示最大值所在的索引indice
#     # type_as()将张量转化为labels类型
#     correct = preds.eq(labels).double()
#     # eq是判断preds与labels是否相等，相等的话对应元素置1，不等置0
#     correct = correct.sum()
#     # 对其求和，即求出相等(置1)的个数
#     return correct / len(labels)  # 计算准确率
#
#
# # Scipy中的sparse matrix转换为PyTorch中的sparse matrix，此函数可参考学习借鉴复用
# # 构建稀疏张量，一般需要Coo索引、值以及形状大小等信息
# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """将scipy稀疏矩阵转换为torch稀疏张量。"""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     # tocoo()是将此矩阵转换为Coo格式，astype()转换数组的数据类型
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     # vstack()将两个数组按垂直方向堆叠成一个新数组
#     # torch.from_numpy()是numpy中的ndarray转化成pytorch中的tensor
#     # Coo的索引
#     values = torch.from_numpy(sparse_mx.data)
#     # Coo的值
#     shape = torch.Size(sparse_mx.shape)
#     # Coo的形状大小
#     return torch.sparse.FloatTensor(indices, values, shape)  # sparse.FloatTensor()构造构造稀疏张量
#
# ##
# idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
# # 提取论文样本的编号id数组
# idx_map = {j: i for i, j in enumerate(idx)}
# # 由样本id到样本索引的映射字典
# # enumerate()将可遍历对象组合成一个含数据下标和数据的索引序列
# # {}生成了字典，论文编号id作为索引的键，顺序数据下标值i作为键值:0,1,2,...
# edges_unordered = np.genfromtxt("/home/dkb/workspace/Code/nlp_data/vocab_label/test_edge.txt",
#                                 dtype=np.int32)
# # 论文样本之间的引用关系的数组
# # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
# # np.genfromtxt(fname, dtype)
# # frame：文件名	../data/cora/cora.cites		dtype：数据类型	int32
# edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                  dtype=np.int32).reshape(edges_unordered.shape)
# # 将论文样本之间的引用关系用样本字典索引之间的关系表示，
# # 说白了就是将论文引用关系数组中的数据(论文id）替换成对应字典的索引键值
# # list()用于将元组转换为列表。flatten()是将关系数组降为一维，默认按一行一行排列
# # map()是对降维后的一维关系数组序列中的每一个元素调用idx_map.get进行字典索引，
# # 即将一维的论文引用关系数组中论文id转化为对应的键值数据
# # .shape是读取数组的维度，.reshape()是将一维数组复原成原来维数形式
# adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                     shape=(labels.shape[0], labels.shape[0]),
#                     dtype=np.float32)
# # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
# # edges.shape[0]表示引用关系数组的维度数（行数），np.ones全1的n维数组
# # edges[:, 0]被引用论文的索引数组做行号row，edges[:, 1]引用论文的索引数组做列号col
# # labels.shape[0]总论文样本的数量，做方阵维数
# # 前面说白了就是引用论文的索引做列，被引用论文的索引做行，然后在这个矩阵面填充1，其余填充0
#
# # 建立对称邻接矩阵
# adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
# # 将非对称邻接矩阵转变为对称邻接矩阵（有向图转无向图）
# # A.multiply(B)是A与B的Hadamard乘积，A>B是指按位将A大于B的位置进行置1其余置0（仅就这里而言可以这么理解，我没找到具体解释）
# # adj=adj+((adj转置)⊙(adj.T > adj))-((adj)⊙(adj.T > adj))
# # 基本上就是将非对称矩阵中的非0元素进行对应对称位置的填补，得到对称邻接矩阵
#
# features = normalize(features)
# # features是样本特征的压缩稀疏矩阵，行规范化稀疏矩阵，具体函数后面有定义
# adj = normalize(adj + sp.eye(adj.shape[0]))
# # 对称邻接矩阵+单位矩阵，并进行归一化
# # 这里即是A+I，添加了自连接的邻接矩阵
# # adj=D^-1(A+I)
#
# # 分割为train，val，test三个集，最终数据加载为torch的格式并且分成三个数据集
# idx_train = range(140)  # 0~139，训练集索引列表
# idx_val = range(200, 500)  # 200~499，验证集索引列表
# idx_test = range(500, 1500)  # 500~1499，测试集索引列表
# # range()创建整数列表
#
# features = torch.FloatTensor(np.array(features.todense()))  # 将特征矩阵转化为张量形式
# # .todense()与.csr_matrix()对应，将压缩的稀疏矩阵进行还原
# labels = torch.LongTensor(np.where(labels)[1])
# # np.where(condition)，输出满足条件condition(非0)的元素的坐标，np.where()[1]则表示返回列的索引、下标值
# # 说白了就是将每个标签one-hot向量中非0元素位置输出成标签
# # one-hot向量label转常规label：0,1,2,3,……
# adj = sparse_mx_to_torch_sparse_tensor(adj)
# # 将scipy稀疏矩阵转换为torch稀疏张量，具体函数下面有定义
#
# idx_train = torch.LongTensor(idx_train)  # 训练集索引列表
# idx_val = torch.LongTensor(idx_val)  # 验证集索引列表
# idx_test = torch.LongTensor(idx_test)  # 测试集索引列表
# # 转化为张量
# print("Done")
#
#
# # 返回（样本关系的对称邻接矩阵的稀疏张量，样本特征张量，样本标签，
# # 		训练集索引列表，验证集索引列表，测试集索引列表）
#

'''
meaning转为id
'''
# from formatter.Seq2Seq.dict import Dict
#
#
# label_vocab_path = '/home/dkb/workspace/Code/nlp_data/vocab_label/label_vocab.src.dict'
#
#
# label_vocab = Dict()
# label_vocab.loadFile(label_vocab_path)
#
# meaning_dic = {}
# file_name = '/home/dkb/workspace/Code/nlp_data/vocab_label/label_meaning'
# with open(file_name, 'r', encoding="utf-8") as load_f:
#     meaning_dic = json.load(load_f)
#
# washed_dic = {}
#
# for k,v in meaning_dic.items():
#     srcWords = v.split()
#     srcWords = [word for word in srcWords]
#
#     washed_dic[k] = label_vocab.convertToIdx(srcWords, unkWord=True)
#
# print("Done")
'''
转化dic
'''
# f = open("/home/dkb/workspace/Code/nlp_data/jieba/trg_level_3.dic", 'r+')
# i = -4
# dic_json={}
#
# for line in f:
#     if i >= 0:
#         (key, value) = line.strip().split()
#         dic_json[key] = i
#         i += 1
#     else:
#         i += 1
#         continue
#
# dict_sorted=sorted(dic_json.items(), key=lambda d:d[1])
# results=[key for key,value in dict_sorted]
#
# results = np.array(results)
#
#
# dic_json = sorted(dic_json.items(),key=lambda x:x[1],reverse = False)
#
# dic_out = {}
# for i in range(0,len(dic_json)):
#     dic_out[dic_json[i][0]] = i
#
# file_name = '/home/dkb/workspace/Code/nlp_data/jieba/trg_label.json'
# json_str = json.dumps(dic_out, ensure_ascii=False)
#
# with open(file_name, 'w', encoding='utf-8') as f:
#     f.write(json_str)
# print("写入json文件：", dic_out)
# #################
# f2 = open('/home/dkb/workspace/Code/nlp_data/jieba/trg_label.json', 'r+')
# ff = open('/home/dkb/workspace/Code/nlp_data/jieba/trg_label.dic', 'r+')
# onedic = eval(f2.readline())  # 读取的str转换为字典
#
# for di in onedic:
#         line = di + " " + str(int(onedic[di]))
#         ff.write(line)
#         ff.write('\n')
# '''统计标签数量'''
# data = []
# d = {}
# f = open("/home/dkb/workspace/Code/nlp_data/jieba/valid.json", "r", encoding='utf8')
# cnt = 0
# for line in f:
#     data.append(json.loads(line))
#     cnt += 1
#
# for temp in data:
#     tline = temp["label"].strip()
#     tgtWords = tline.split()
#     tgt_Words = [word for word in tgtWords[:3]]
#     for j in tgt_Words:
#         if not j in d:
#             d[j] = 1
#         else:
#             d[j] = d[j] + 1
#
#
#
# print(d)
# print('\n')
# print(len(data))
# print('\n')
# print(sum(d.values()))

# 验证集
# 156912
# {'合同、无因管理、不当得利纠纷': 35213, '合同纠纷': 35188, '借款合同纠纷': 17956, '侵权责任纠纷1': 9139, '侵权责任纠纷': 9139, '机动车交通事故责任纠纷': 8534, '买卖合同纠纷': 4676, '与公司、证券、保险、票据等有关的民事纠纷': 753, '保险纠纷': 306, '财产保险合同纠纷': 237, '租赁合同纠纷': 1886, '追偿权纠纷': 511, '承揽合同纠纷': 678, '房屋买卖合同纠纷': 2626, '婚姻家庭、继承纠纷': 6627, '婚姻家庭纠纷': 6627, '离婚纠纷': 5957, '银行卡纠纷': 1588, '建设工程合同纠纷': 882, '委托合同纠纷': 196, '服务合同纠纷': 1103, '医疗损害责任纠纷': 114, '确认合同效力纠纷': 202, '合伙协议纠纷': 96, '房屋拆迁安置补偿合同纠纷': 114, '离婚后财产纠纷': 109, '物权纠纷': 410, '物权保护纠纷': 376, '财产损害赔偿纠纷': 211, '与企业有关的纠纷': 137, '挂靠经营合同纠纷': 137, '赡养纠纷': 147, '同居关系纠纷': 108, '农村土地承包合同纠纷': 177, '保证合同纠纷': 176, '债权转让合同纠纷': 91, '劳务合同纠纷': 581, '提供劳务者受害责任纠纷': 383, '期货交易纠纷': 155, '期货交易代理合同纠纷': 155, '人格权纠纷1': 389, '人格权纠纷': 389, '生命权、健康权、身体权纠纷': 389, '典当纠纷': 23, '抚养纠纷': 243, '运输合同纠纷': 218, '融资租赁合同纠纷': 185, '婚约财产纠纷': 63, '教育机构责任纠纷': 29, '债务转移合同纠纷': 28, '人身保险合同纠纷': 26, '劳动争议、人事争议': 136, '劳动争议': 136, '劳动合同纠纷': 104, '居间合同纠纷': 97, '与公司有关的纠纷': 71, '股权转让纠纷': 71, '返还原物纠纷': 49, '物权确认纠纷': 36, '供用热力合同纠纷': 59, '知识产权与竞争纠纷': 21, '知识产权权属、侵权纠纷': 21, '商标权权属、侵权纠纷': 21, '与破产有关的纠纷': 84, '追收未缴出资纠纷': 84, '用益物权纠纷': 34, '土地承包经营权纠纷': 34, '广告合同纠纷': 34, '农业承包合同纠纷': 31, '排除妨害纠纷': 56, '不当得利纠纷2': 25, '不当得利纠纷': 25}

# 测试集
# 156908
# {'合同、无因管理、不当得利纠纷': 35210, '合同纠纷': 35184, '借款合同纠纷': 17954, '侵权责任纠纷1': 9139, '侵权责任纠纷': 9139, '机动车交通事故责任纠纷': 8533, '建设工程合同纠纷': 883, '服务合同纠纷': 1105, '买卖合同纠纷': 4675, '婚姻家庭、继承纠纷': 6627, '婚姻家庭纠纷': 6627, '离婚纠纷': 5957, '与公司、证券、保险、票据等有关的民事纠纷': 754, '与公司有关的纠纷': 71, '股权转让纠纷': 71, '与破产有关的纠纷': 84, '追收未缴出资纠纷': 84, '银行卡纠纷': 1588, '租赁合同纠纷': 1885, '运输合同纠纷': 217, '劳务合同纠纷': 581, '期货交易纠纷': 155, '期货交易代理合同纠纷': 155, '人格权纠纷1': 389, '人格权纠纷': 389, '生命权、健康权、身体权纠纷': 389, '房屋买卖合同纠纷': 2625, '承揽合同纠纷': 677, '保证合同纠纷': 176, '债权转让合同纠纷': 91, '知识产权与竞争纠纷': 22, '知识产权权属、侵权纠纷': 22, '商标权权属、侵权纠纷': 22, '与企业有关的纠纷': 136, '挂靠经营合同纠纷': 136, '农业承包合同纠纷': 31, '房屋拆迁安置补偿合同纠纷': 113, '物权纠纷': 411, '物权保护纠纷': 377, '财产损害赔偿纠纷': 211, '提供劳务者受害责任纠纷': 382, '农村土地承包合同纠纷': 179, '融资租赁合同纠纷': 184, '确认合同效力纠纷': 203, '追偿权纠纷': 511, '合伙协议纠纷': 96, '委托合同纠纷': 195, '婚约财产纠纷': 64, '返还原物纠纷': 50, '同居关系纠纷': 108, '排除妨害纠纷': 56, '劳动争议、人事争议': 135, '劳动争议': 135, '劳动合同纠纷': 103, '保险纠纷': 308, '财产保险合同纠纷': 238, '医疗损害责任纠纷': 114, '抚养纠纷': 243, '赡养纠纷': 147, '不当得利纠纷2': 26, '不当得利纠纷': 26, '广告合同纠纷': 35, '人身保险合同纠纷': 27, '物权确认纠纷': 36, '供用热力合同纠纷': 59, '离婚后财产纠纷': 108, '居间合同纠纷': 96, '用益物权纠纷': 34, '土地承包经营权纠纷': 34, '典当纠纷': 23, '教育机构责任纠纷': 30, '债务转移合同纠纷': 28}
'''
统计label含义平均长度
'''
# total_dic = {}
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/washed_total.json'
# with open(file_name, 'r', encoding="utf-8") as load_f:
#     total_dic = json.load(load_f)
# lenn = 0
# cnt = 0
# for key in total_dic:
#     cnt+=1
#     lenn+=len(total_dic[key]["meaning"])
#
# lenn = lenn/cnt
# # lenn = 453.3333....
# print("Done")

'''
合并label与src词典
'''
# src_dict = Dict("/home/dkb/workspace/Code/nlp_data/jieba/src_dic")
# label_dict = Dict("/home/dkb/workspace/Code/nlp_data/vocab_label/label_vocab.src.dict")
#
# for i in range(0,label_dict.size()):
#     word = label_dict.getLabel(i)
#     idx = src_dict.add(word)
#     print(word+""+str(idx)+"")
# src_dict.writeFile("/home/dkb/workspace/Code/nlp_data/jieba/src_label.dic")
# print("Done")
'''
构建 "其他 标签
'''
# # 涉及层次字典操作 ##############################################################################
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
# # 涉及层次字典操作 ##############################################################################
#
#
# idx1 = first_dic_name_index["人格权纠纷1"]
# idx2 = first_dic_name_index["物权纠纷"]
# idx3 = first_dic_name_index["与公司、证券、保险、票据等有关的民事纠纷"]
# idx4 = first_dic_name_index["劳动争议、人事争议"]
# list1 = ["人格权纠纷1", "物权纠纷", "与公司、证券、保险、票据等有关的民事纠纷", "劳动争议、人事争议"]
# list2 = ["人格权纠纷", "物权保护纠纷", "用益物权纠纷", "与企业有关的纠纷", "与公司有关的纠纷", "与破产有关的纠纷", "期货交易纠纷", "保险纠纷", "劳动争议"]
# list3 = ["生命权、健康权、身体权纠纷", "物权确认纠纷", "返还原物纠纷", "排除妨害纠纷", "财产损害赔偿纠纷", "土地承包经营权纠纷", "挂靠经营合同纠纷", "股权转让纠纷", "追收未缴出资纠纷",
#          "期货交易代理合同纠纷", "财产保险合同纠纷", "人身保险合同纠纷", "劳动合同纠纷"]
# for i in list2:
#
#     # for key1 in twodic_name_label.keys():# 这里是实际索引
#     vec1 = Hie_dic[i]
#     print("**********parent:" + str(i) + str(vec1))
#     listchild = []
#     restchild = []
#     for key2 in third_dic_name_index.keys():  # 这里是字典里面的索引
#         vec2 = Hie_dic[key2]
#         if (vec1[1] == vec2[1]):  # 1
#             try:
#                 listchild.append(threedic_name_label[key2])  # 字典转化为实际索引
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
#
#     print("child list:")
#     print(listchild)
#     print("rest list:")
#     print(restchild)
#     print("##")
'''
构建dummy '其他'标签
'''
# list1 = ["人格权纠纷1", "物权纠纷", "与公司、证券、保险、票据等有关的民事纠纷", "劳动争议、人事争议"]
# list2 = ["人格权纠纷", "物权保护纠纷", "用益物权纠纷", "与企业有关的纠纷", "与公司有关的纠纷", "与破产有关的纠纷", "期货交易纠纷", "保险纠纷", "劳动争议"]
# list3 = ["生命权、健康权、身体权纠纷", "物权确认纠纷", "返还原物纠纷", "排除妨害纠纷", "财产损害赔偿纠纷",
#          "土地承包经营权纠纷", "挂靠经营合同纠纷", "股权转让纠纷", "追收未缴出资纠纷",
#          "期货交易代理合同纠纷", "财产保险合同纠纷", "人身保险合同纠纷", "劳动合同纠纷"]
#
# dic_json={}
#
# trg_dic = Dict("/home/dkb/workspace/Code/nlp_data/jieba/trg_label.dic")
# dummy_dic = Dict
# for i in list1:
#     idx = trg_dic.lookup(i)
#     dic_json[idx] = i
#
# for i in list2:
#     idx = trg_dic.lookup(i)
#     dic_json[idx] = i
#
# for i in list3:
#     idx = trg_dic.lookup(i)
#     dic_json[idx] = i
#
# file_name = '/home/dkb/workspace/Code/nlp_data/dummy/dummy.json'
# json_str = json.dumps(dic_json, ensure_ascii=False)
#
# with open(file_name, 'w', encoding='utf-8') as f:
#     f.write(json_str)
# # 验证
# f = open('/home/dkb/workspace/Code/nlp_data/dummy/dummy.json', encoding='utf-8')  # 打开文件
# Hie_dic = json.load(f)
# print("Done")
'''
记录'其他'标签
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
# f = open('/home/dkb/workspace/Code/nlp_data/dummy/ori_dum.json', encoding='utf-8')  # 打开文件
# dummy = json.load(f)
#
# re_trg = Dict("/home/dkb/workspace/Code/nlp_data/dummy/re_trg_3.dic")
# re_dum = Dict()
#
# trg_dic = Dict("/home/dkb/workspace/Code/nlp_data/jieba/trg_label.dic")
# # for i in range(0,trg_dic.size()):
# #     if str(i) in dummy.keys():
# #         continue
# #     if i < 22:
# #         continue
# #     labelname = trg_dic.getLabel(i)
# #
# #     re_trg.add(label=labelname)
# # re_trg.add(label='dummy3')
#
# cnt = re_trg.size()
# for i in dummy.keys():
#     re_dum.add(dummy[i],cnt)
#     cnt += 1
#
# # re_trg.writeFile("/home/dkb/workspace/Code/nlp_data/dummy/re_trg_3.dic")
#
# with open("/home/dkb/workspace/Code/nlp_data/dummy/re_dum.dic", 'w') as file:
#     for i in range(re_trg.size(),cnt):
#         label = re_dum.idxToLabel[i]
#         file.write('%s %d\n' % (label, i))

# # re_dum = Dict("/home/dkb/workspace/Code/nlp_data/dummy/re_dum.dic")
# # new_dum = Dict()
# # for i in range(49, 75):
# #     labelname = re_dum.getLabel(i)
# #     new_dum.add(label=labelname)
# # new_dum.writeFile("/home/dkb/workspace/Code/nlp_data/dummy/dum.dic")
'''
合并 非dummy标签和一二三级的dummy标签
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
# f = open('/home/dkb/workspace/Code/nlp_data/dummy/ori_dum.json', encoding='utf-8')  # 打开文件
# dummy = json.load(f)
#
# re_trg = Dict("/home/dkb/workspace/Code/nlp_data/dummy/re_trg_3.dic")
#
# re_dum = Dict("/home/dkb/workspace/Code/nlp_data/dummy/re_dum.dic")
#
# trg_dic = Dict("/home/dkb/workspace/Code/nlp_data/jieba/trg_label.dic")
#
# merge_dic = Dict()
#
# for i in range(0,re_trg.size()):
#     if re_trg.getLabel(i) == 'dummy1':
#         for i in range(49,49 + re_dum.size()):
#             if re_dum.getLabel(i) in list(onedic_name_label.keys()):
#                 merge_dic.add(label=re_dum.getLabel(i))
#         continue
#
#     if re_trg.getLabel(i) == 'dummy2':
#         for i in range(49, 49 + re_dum.size()):
#             if re_dum.getLabel(i) in list(twodic_name_label.keys()):
#                 merge_dic.add(label=re_dum.getLabel(i))
#         continue
#     if re_trg.getLabel(i) == 'dummy3':
#         for i in range(49, 49 + re_dum.size()):
#             if re_dum.getLabel(i) in list(threedic_name_label.keys()):
#                 merge_dic.add(label=re_dum.getLabel(i))
#         continue
#     labelname = re_trg.getLabel(i)
#     merge_dic.add(label=labelname)
#
#
# merge_dic.writeFile("/home/dkb/workspace/Code/nlp_data/dummy/merge.dic")
#
# # with open("/home/dkb/workspace/Code/nlp_data/dummy/re_dum.dic", 'w') as file:
# #     for i in range(re_trg.size(),cnt):
# #         label = re_dum.idxToLabel[i]
# #         file.write('%s %d\n' % (label, i))
'''
构建dummy之间的上下相连关系
'''

'''
构建分层的字典
根据level-index文件
找出两个级别案由之间的切分关系
转化为【0，1，0，1，0，0，0】
'''
#
# # base_path = '/home/dkb/workspace/Code/analyze/'
#
#
# f = open('/home/dkb/workspace/Code/pretrain/Hie/Hie_dic.json', encoding='utf-8')  # 打开文件
# Hie_dic = json.load(f)  # 把json串变成python的数据类型：字典，传一个文件对象，它会帮你读文件，不需要再单独读文件
# f.close()
# level_list = []
# for k, v in Hie_dic.items():
#     level_list.append(v)
# # 因为Hie.dic中的索引是根据下面的来的，因此需要通过first/second/..._dic的name作为中介寻找在Hie中的索引
#
# listdic = {}
# restdic = {}
# dic_1_2 = []
#
# re_dummy_dict = Dict("/home/dkb/workspace/Code/nlp_data/dummy/dum.dic")
# dummy_list = re_dummy_dict.convertToLabels(range(re_dummy_dict.size()), None)
#
#
# i = 0
# for key1 in dummy_list[4:13]:  # 这里是实际索引
#     vec1 = Hie_dic[key1]
#     print("**********parent:" + str(re_dummy_dict.lookup(key1)) + str(key1))
#     listchild = []
#     restchild = []
#     for key2 in dummy_list[13:]:  # 这里是字典里面的索引
#         vec2 = Hie_dic[key2]
#         if vec1[1] == vec2[1]:  # 1
#             try:
#                 listchild.append(re_dummy_dict.lookup(key2))  # 字典转化为实际索引
#                 print("-child:" + str(re_dummy_dict.lookup(key2)) + str(key2))
#             except KeyError as e:
#                 continue
#
#         else:
#             try:
#                 restchild.append(re_dummy_dict.lookup(key2))
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
# aaa = np.zeros((len(dummy_list[4:13]), len(dummy_list)), dtype=int)
#
# for i, val in enumerate(dic_1_2):
#     if len(val) != 0:
#         aaa[i][np.array(val)] = 1
#
# print("hup")
# # fw = open("/home/dkb/workspace/Code/Code/pretrain/Arti/res-1-2.txt",'w+')
# # fw.write(str(resdic_1_2)) #把字典转化为str
# # fw.close()
#
# # base_path = '/home/dkb/workspace/Code/analyze/'
#
# np.savetxt("/home/dkb/workspace/Code/nlp_data/level-2-3.txt", aaa)

'''
为dummy构造标签含义的文件：meaning_dummy
顺序与dummy.dic相同
'''
#
# def seg_depart(sentence):
#     # sentence = process(sentence.strip())
#     # sentence = format_str(sentence)
#
#     # 对文档中的每一行进行中文分词
#     sentence = sentence.strip()
#
#     # sentence = sentence.translate(str.maketrans("", ""), string.punctuation)
#     # sentence = removePunctuation(sentence)
#
#     # print("正在分词")
#     sentence_depart = jieba.cut(sentence)
#     # 创建一个停用词列表
#     # stopwords = stopwordslist()
#     # 输出结果为outstr
#     outstr = ''
#     # 去停用词
#     i = 0
#     for word in sentence_depart:
#         # if word not in stopwords:
#         if word != '\t' and word != '\n':
#             i += 1
#             if i > 500:
#                 break
#             outstr += word
#             outstr += " "
#     # print(i)
#     return outstr
#
# file_name = '/home/dkb/workspace/Code/nlp_data/labe_mean/washed_total.json'
#
# with open(file_name,'r',encoding='utf8')as fp:
#     json_data = json.load(fp)
#
# re_dummy_dict = Dict("/home/dkb/workspace/Code/nlp_data/jieba/trg_label.dic")
# labels = re_dummy_dict.convertToLabels(list(range(0,re_dummy_dict.size())),None)
#
# dic_label = {}
# num = 0
#
# for label in labels:
#     for k in json_data:
#         if label == json_data[k]['label']:
#             outstr = seg_depart(json_data[k]['meaning'])
#             dic_label[num] = outstr
#             num += 1
#
# file_name = '/home/dkb/workspace/Code/nlp_data/vocab_label/label_meaning'
# json_str = json.dumps(dic_label, ensure_ascii=False,indent=4)
# with open(file_name, 'w', encoding='utf-8') as f:
#     f.write(json_str)
'''
测试 dataloader的num参数设置
'''
# import time
# import torch.utils.data as d
# import torchvision
# import torchvision.transforms as transforms
#
# if __name__ == '__main__':
#     BATCH_SIZE = 100
#     transform = transforms.Compose([transforms.ToTensor(),
#                                     transforms.Normalize((0.5,), (0.5,))])
#     train_set = torchvision.datasets.MNIST('\mnist', download=True, train=True, transform=transform)
#
#     # data loaders
#     train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
#
#     for num_workers in range(20):
#         train_loader = d.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
#         # training ...
#         start = time.time()
#         for epoch in range(1):
#             for step, (batch_x, batch_y) in enumerate(train_loader):
#                 pass
#         end = time.time()
#         print('num_workers is {} and it took {} seconds'.format(num_workers, end - start))
# print("Done")
# '''
# 单分类数据集构建(都是token id 不可用于torch)
# '''
# import os
# import pickle
#
# def load_data(cache_file_h5py, cache_file_pickle):
#     """
#     load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
#     :param cache_file_h5py:
#     :param cache_file_pickle:
#     :return:
#     """
#     print(cache_file_h5py)
#     print(cache_file_pickle)
#     if not os.path.exists(cache_file_h5py) or not os.path.exists(cache_file_pickle):
#         raise RuntimeError("############################ERROR##############################\n. "
#                            "please download cache file, it include training data and vocabulary & labels. "
#                            "link can be found in README.md\n download zip file, unzip it, then put cache files as FLAGS."
#                            "cache_file_h5py and FLAGS.cache_file_pickle suggested location.")
#     print("INFO. cache file exists. going to load cache file")
#     f_data = h5py.File(cache_file_h5py, 'r')
#     print("f_data.keys:", list(f_data.keys()))
#     train_X = f_data['train_X']  # np.array(
#     print("train_X.shape:", train_X.shape)
#     train_Y = f_data['train_Y']  # np.array(
#     print("train_Y.shape:", train_Y.shape, ";")
#     vaild_X = f_data['vaild_X']  # np.array(
#     valid_Y = f_data['valid_Y']  # np.array(
#     test_X = f_data['test_X']  # np.array(
#     test_Y = f_data['test_Y']  # np.array(
#     # print(train_X)
#     # f_data.close()
#
#     # train_Z = f_data['train_Z']
#     # test_Z = f_data['test_Z']
#     # valid_Z = f_data['valid_Z']
#     # train_Z_ansc = []
#     # for j in train_Z:
#     #     train_Z_ansc.append(j.encode())
#     # test_Z_ansc = []
#     # for j in test_Z:
#     #     test_Z_ansc.append(j.encode())
#     # vaild_Z_ansc = []
#     # for j in valid_Z:
#     #     vaild_Z_ansc.append(j.encode())
#
#     word2index, label2index = None, None
#     with open(cache_file_pickle, 'rb') as data_f_pickle:
#         word2index, label2index = pickle.load(data_f_pickle)
#     print("INFO. cache file load successful...")
#     return word2index, label2index, train_X, train_Y, vaild_X, valid_Y, test_X, test_Y
#
#
# cache_file_h5py = "/home/dkb/workspace/Code/pretrain/Hie/dataHie-146.h5"
# cache_file_pickle = "/home/dkb/workspace/Code/pretrain/Hie/vocab_labelHie-146.pik"
#
# word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY=load_data(cache_file_h5py, cache_file_pickle)
# print("train_y_short:", trainY[0])
# print("Done")
#

# """
# 为slot任务，重新清洗all.csv，只取合同领域的数据
# """
# # 获取合同领域的三级标签
# print("Done")

# '''
# 合并二三级案由
# '''
# dic1 = {'合同纠纷': 0, '侵权责任纠纷': 1, '婚姻家庭纠纷': 2, '人格权纠纷': 3, '物权保护纠纷': 4, '期货交易纠纷': 5, '保险纠纷': 6, '与企业有关的纠纷': 7,
#           '劳动争议': 8, '与破产有关的纠纷': 9, '与公司有关的纠纷': 10, '用益物权纠纷': 11, '不当得利纠纷2': 12, '知识产权权属、侵权纠纷': 13}
#
# dic2 = {'借款合同纠纷': 0, '机动车交通事故责任纠纷': 1, '离婚纠纷': 2, '买卖合同纠纷': 3, '银行卡纠纷': 4, '房屋买卖合同纠纷': 5, '服务合同纠纷': 6, '租赁合同纠纷': 7,
#      '劳务合同纠纷': 8, '建设工程合同纠纷': 9, '追偿权纠纷': 10, '承揽合同纠纷': 11, '生命权、健康权、身体权纠纷': 12, '提供劳务者受害责任纠纷': 13, '财产损害赔偿纠纷': 14,
#      '委托合同纠纷': 15, '融资租赁合同纠纷': 16, '保证合同纠纷': 17, '抚养纠纷': 18, '期货交易代理合同纠纷': 19, '财产保险合同纠纷': 20, '挂靠经营合同纠纷': 21,
#      '运输合同纠纷': 22, '医疗损害责任纠纷': 23, '房屋拆迁安置补偿合同纠纷': 24, '离婚后财产纠纷': 25, '确认合同效力纠纷': 26, '劳动合同纠纷': 27, '赡养纠纷': 28,
#      '居间合同纠纷': 29, '合伙协议纠纷': 30, '债权转让合同纠纷': 31, '农村土地承包合同纠纷': 32, '同居关系纠纷': 33, '追收未缴出资纠纷': 34, '股权转让纠纷': 35,
#      '婚约财产纠纷': 36, '供用热力合同纠纷': 37, '排除妨害纠纷': 38, '返还原物纠纷': 39, '物权确认纠纷': 40, '广告合同纠纷': 41, '土地承包经营权纠纷': 42,
#      '农业承包合同纠纷': 43, '教育机构责任纠纷': 44, '债务转移合同纠纷': 45, '人身保险合同纠纷': 46, '不当得利纠纷': 47, '典当纠纷': 48, '商标权权属、侵权纠纷': 49}
#
#
# trg_dic = Dict("/home/dkb/workspace/Code/nlp_data/jieba/trg_label.dic")
#
# merge_dic = Dict()
#
# for i in range(0,trg_dic.size()):
#     if i>7:
#         merge_dic.add(label=trg_dic.getLabel(i))
#     continue
#
# merge_dic.writeFile("/home/dkb/workspace/Code/nlp_data/Bert/merge.dic")
# '''
# 二三级案由的字
# '''
# trg_dic = Dict("/home/dkb/workspace/Code/nlp_data/jieba/trg_label.dic")
#
# merge_dic = Dict()
#
# for i in range(0,trg_dic.size()):
#     if i>7:
#         ll = ' '.join(trg_dic.getLabel(i)).split(' ')
#         for j in ll:
#             merge_dic.add(label=j)
#     continue
#
# merge_dic.writeFile("/home/dkb/workspace/Code/nlp_data/Bert/char.dic")
'''
http 测试
'''
# import socket
#
# ip_port = ('162.105.87.37', 33155)
# sk = socket.socket()
# sk.bind(ip_port)
# sk.listen(5)
# while True:
#     print('server waiting...')
#     conn, addr = sk.accept()
#     client_data = conn.recv(1024)
#     print(str(client_data, 'utf8'))
#     print(client_data)
#     #返回信息
#     conn.sendall(bytes('接收成功', 'utf8'))
#     conn.close()

# '''
# 剔除二级没有的案由
# '''
# trg_dic = Dict("/home/dkb/workspace/Code/nlp_data/Bert/merge.dic")
#
# ll = ['婚姻家庭纠纷','人格权纠纷','期货交易纠纷','与企业有关的纠纷','与破产有关的纠纷','与公司有关的纠纷','用益物权纠纷','不当得利纠纷2','知识产权权属、侵权纠纷']
#
# merge_dic = Dict()
#
# for i in range(0,trg_dic.size()):
#     if trg_dic.getLabel(i) in ll:
#         continue
#     else:
#         merge_dic.add(label=trg_dic.getLabel(i))
#
# merge_dic.writeFile("/home/dkb/workspace/Code/nlp_data/Bert/clean.dic")
#
# '''
# 三级案由
# '''
# trg_dic = Dict("/home/dkb/workspace/Code/nlp_data/jieba/trg_label.dic")
#
# merge_dic = Dict()
#
# for i in range(0,trg_dic.size()):
#     if i>21:
#         merge_dic.add(label=trg_dic.getLabel(i))
#     continue
#
# merge_dic.writeFile("/home/dkb/workspace/Code/nlp_data/Bert/clean_3.dic")
'''
'''

dir = '/home/dkb/workspace/Code/Code/Test_eval/origin_clean_3_som/result1.txt'

f = open(dir)
lines = f.readlines()
count = 0
target = []
output = []
origin = []
for line in lines:
    if count % 3 == 0:
        target.append(line.strip().split('| ')[1])
        # print(target)
    if count % 3 == 1:
        output.append(line.strip().replace('[','').replace(']','').replace('\'','').split('| ')[1])
        # print(target)
    if count % 3 == 2:
        origin.append(line.strip().split('| ')[1])
    count += 1
f.close()


for i in range(0, len(target)):
    if target[i] != output[i]:
        print("============================")
        print(" |target| " + "".join(str(target[i]).strip()) + '\n')
        print(" |output| " + "".join(str(output[i]).strip()) + '\n')
        print(" |origin| " + "".join(str(origin[i]).strip()) + '\n')
        print("============================")
print("Done")

