import pandas as pd
import json
import numpy as np
import os
import re
import string
import jieba

path = '/home/dkb/workspace/law_data/law_0529'  # 案由数据聚集路径

i = 0

common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
common_used_numerals = {}
for key in common_used_numerals_tmp:
    common_used_numerals[key] = common_used_numerals_tmp[key]


def chinese2digits(uchars_chinese):
    total = 0
    r = 1  # 表示单位：个十百千...
    for i in range(len(uchars_chinese) - 1, -1, -1):
        val = common_used_numerals.get(uchars_chinese[i])
        if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
            if val > r:
                r = val
                total = total + val
            else:
                r = r * val
                # total =total + r * x
        elif val >= 10:
            if val > r:
                r = val
            else:
                r = r * val
        else:
            total = total + r * val
    return total


num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九',
                        '十']
more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']


def changeChineseNumToArab(oriStr):
    lenStr = len(oriStr)
    aProStr = ''
    if lenStr == 0:
        return aProStr

    hasNumStart = False
    numberStr = ''
    for idx in range(lenStr):
        if oriStr[idx] in num_str_start_symbol:
            if not hasNumStart:
                hasNumStart = True

            numberStr += oriStr[idx]
        else:
            if hasNumStart:
                if oriStr[idx] in more_num_str_symbol:
                    numberStr += oriStr[idx]
                    continue
                else:
                    numResult = str(chinese2digits(numberStr))
                    numberStr = ''
                    hasNumStart = False
                    aProStr += numResult

            aProStr += oriStr[idx]
            pass

    if len(numberStr) > 0:
        resultNum = chinese2digits(numberStr)
        aProStr += str(resultNum)

    return aProStr


def format_string(s):
    return s.replace("\t", " ")


# def detect_nowalk(dir_path,i):
#     files = os.listdir(dir_path)
#     for filename in files:
#         if '0' in filename:
#             print("get")
#             i+=1
#             print(i)
#         print ("file:%s\n" % filename)
#         next = os.path.join(dir_path, filename)
#         print(next)
#         if os.path.isdir(next):
#             print ("file folds:%s\n" % filename)
#             detect_nowalk(next,i)
#
# def process(our_data):
#     m1 = map(lambda s: s.replace(' ', ''), our_data)
#     return list(m1)
# def is_chinese(uchar):
#     if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
#         return True
#     else:
#         return False
# def format_str(content):
#     content_str = ''
#     for ii in content:
#         if is_chinese(ii):
#             content_str = content_str + ii
#     return content_str

def remove_punctuation(line):
    rule = re.compile(r"([^a-zA-Z0-9\u4e00-\u9fa5])")
    line = rule.sub('', line)
    return line


def seg_depart(sentence):
    # sentence = process(sentence.strip())
    # sentence = format_str(sentence)

    # 对文档中的每一行进行中文分词
    sentence = sentence.strip()
    sentence = remove_punctuation(sentence)
    # sentence = sentence.translate(str.maketrans("", ""), string.punctuation)
    # sentence = removePunctuation(sentence)

    # print("正在分词")
    sentence_depart = jieba.cut(sentence)
    # 创建一个停用词列表
    # stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    i = 0
    for word in sentence_depart:
        # if word not in stopwords:
        if word != '\t' and word != '\n':
            i += 1
            if i > 500:
                break
            outstr += word
            outstr += " "
    # print(i)
    return outstr


# detect_nowalk(path,i)

'''
开始，循环文件列表：listfile
 '''
listfile = []
flods = os.listdir(path)
for flodname in flods:
    floddir = os.path.join(path, flodname)
    if os.path.isdir(floddir):
        files = os.listdir(floddir)
        for filename in files:
            if flodname in filename:  # flodname
                listfile.append(os.path.join(floddir, filename))
                i += 1
                # print(i)
'''
遍历得到文件路径 currentdir
 '''
store_data = pd.DataFrame(columns = ["label", "des", "tort","contract","marriage"]) #创建一个空的dataframe
n = 0
for currentdir in listfile:
    print(currentdir)
    n += 1
    print(n)

    try:
        gbkdata = pd.read_csv(currentdir, encoding="GBK")
    except UnicodeDecodeError:
        gbkdata = pd.read_csv(currentdir, encoding="gb18030")
        pass
    else:
        pass
    # gbkdata = pd.read_csv(currentdir, encoding="gb18030")
    gbkdata = gbkdata.fillna('')
    '''
    读取文件后，对一个文件操作
     '''
    for index, row in gbkdata.iterrows():
        '''
        去掉描述有问题的原告内容（过短，无描述）
        '''
        if '......' in row[1] or '......' in row[4] or len(row[1]) == 0 or len(row[4]) == 0:
            gbkdata.drop(index)
            continue

        if len(row[1]) < 200:
            gbkdata.drop(index)
            continue

        '''
        分词，输出分词结果
        '''
        outstr = seg_depart(row[1])
        # print(outstr)
        # print(len(outstr))
        '''
        正则表达式提取
         '''
        dic1_law = {}
        rex1 = re.compile(r"(《中华人民共和国侵权责任法》(第[\u4E00-\u9FA5\\s]+条.)+)")

        dic2_law = {}
        rex2 = re.compile(r"(《中华人民共和国合同法》(第[\u4E00-\u9FA5\\s]+条.)+)")

        dic3_law = {}
        rex3 = re.compile(r"(《中华人民共和国婚姻法》(第[\u4E00-\u9FA5\\s]+条.)+)")

        s = format_string(row[4])

        # print(s + '\n')
        # debug = rex1.search(s)
        pre_list_chinese1 = rex1.findall(s)
        pre_list_chinese2 = rex2.findall(s)
        pre_list_chinese3 = rex3.findall(s)

        for preindex, preval in enumerate(pre_list_chinese1):
            rexagain = re.compile(r"(第[\u4E00-\u9FA5\\s]+条)")
            # print(type(rexagain))
            law_list_chinese = re.findall(rexagain, preval[0])
            for lawindex, law_chinese in enumerate(law_list_chinese):
                law_chinese.strip()
                # re.sub('[\r\n\t]', '', law_chinese)
                law_chinese = law_chinese.replace('第', '')
                law_chinese = law_chinese.replace('条', '')
                law_number_str = changeChineseNumToArab(law_chinese)
                # 数据中有law_number = '114和248'情况，再次提取数字
                number_list = re.findall(r'(\d+)', law_number_str)
                for law_number in number_list:
                    if 100 < int(law_number):
                        continue
                    dic1_law[law_number] = True

        for preindex, preval in enumerate(pre_list_chinese2):
            rexagain = re.compile(r"(第[\u4E00-\u9FA5\\s]+条)")
            law_list_chinese = re.findall(rexagain, preval[0])
            for lawindex, law_chinese in enumerate(law_list_chinese):
                law_chinese.strip()
                # re.sub('[\r\n\t]', '', law_chinese)
                law_chinese = law_chinese.replace('第', '')
                law_chinese = law_chinese.replace('条', '')
                law_number_str = changeChineseNumToArab(law_chinese)
                # 数据中有law_number = '114和248'情况，再次提取数字
                number_list = re.findall(r'\d+', law_number_str)
                for law_number in number_list:
                    if 500 < int(law_number):
                        continue
                    dic2_law[law_number] = True

        for preindex, preval in enumerate(pre_list_chinese3):
            rexagain = re.compile(r"(第[\u4E00-\u9FA5\\s]+条)")
            law_list_chinese = re.findall(rexagain, preval[0])
            for lawindex, law_chinese in enumerate(law_list_chinese):
                law_chinese.strip()
                # re.sub('[\r\n\t]', '', law_chinese)
                law_chinese = law_chinese.replace('第', '')
                law_chinese = law_chinese.replace('条', '')
                law_number_str = changeChineseNumToArab(law_chinese)
                # 数据中有law_number = '114和248'情况，再次提取数字
                number_list = re.findall(r'\d+', law_number_str)
                for law_number in number_list:
                    if 60 < int(law_number):
                        continue
                    dic3_law[law_number] = True

        # if len(dic1_law) == 0:
        #     gbkdata.drop(index)
        #     continue
        # print("first")
        row1_law_list = []
        for key in dic1_law.keys():
            row1_law_list.append(key)
        # print(row1_law_list)
        # print("second")
        row2_law_list = []
        for key in dic2_law.keys():
            row2_law_list.append(key)
        # print(row2_law_list)
        # print("third")
        row3_law_list = []
        for key in dic3_law.keys():
            row3_law_list.append(key)
        # print(row3_law_list)
        '''
        输出法条列表
        '''
        # ["label", "des", "tort","contract","marriage"]

        strlist = currentdir.split('/')
        label = strlist[-2]
        # print(label)

        new = pd.DataFrame({'label': label,
                            'des': outstr,
                            'tort': "|".join(row1_law_list),
                            'contract': "|".join(row2_law_list),
                            'marriage': "|".join(row3_law_list)},
                           index=[1])
        # 自定义索引为：1 ，这里也可以不设置index
        # print(new)

        store_data = store_data.append(new, ignore_index=True)  # ignore_index=True,表示不按原来的索引，从0开始自动递增
        # print(store_data)

    #防止内存不够，在这里接连append进目标文件：/home/dkb/workspace/Code/pretrain/Article_des.csv
    store_data.to_csv('./Article_des.csv',mode='a',header=None, encoding = "utf-8")
    store_data.drop(store_data.index,inplace=True)
#内存足够大的话，在这里一次性存即可
# store_data.to_csv('/home/dkb/workspace/Code/pretrain/all_with_law.csv')

print("Done")
