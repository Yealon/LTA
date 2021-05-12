import time
import re
import pandas as pd
import json
import numpy as np

# path = '/home/dkb/workspace/Code/pretrain/Hie.xlsx' # 处理最初版本 Hie.xlsx
#
# data = pd.DataFrame(pd.read_excel(path,sheet_name='Sheet2')) # 处理最初版本 Hie.xlsx

# path = '/home/dkb/workspace/Code/pretrain/washed.xlsx'  # 处理清洗后版本 washed.xlsx
#
# data = pd.DataFrame(pd.read_excel(path, sheet_name='Sheet1'))  # 处理清洗后版本 washed.xlsx
#
# data = data.fillna('')

'''
一级案由
'''
# result = data['一级案由']
#
# first2index={}
# i = 1
# for index, row in result.items():
#
#     if row == '':
#         continue
#     row = row.strip()
#     # print(row[4:])
#     first2index[row[4:]] = i
#     i += 1
#
# f = open("/home/dkb/workspace/Code/pretrain/first2index.txt", 'w')
# f.write(str(first2index))
# f.close()
# print("save dict successfully.")

'''
二级案由
'''

# result = data['二级案由']
# second2index = {}
# n = 1
# for index, row in result.items():
#
#     if row == '':
#         continue
#     row = row.strip()
#     if n <= 10:
#         second2index[row[2:]] = n
#         # print(row[2:]+"||"+str(n))
#     elif n <= 20:
#         second2index[row[3:]] = n
#         # print(row[3:]+"||"+str(n))
#     elif n %10 ==0:
#         second2index[row[3:]] = n
#         # print(row[3:] + "||" + str(n))
#     elif n < 100:
#         second2index[row[4:]] = n
#         # print(row[4:] + "||" + str(n))
#
#     n = n+1
#
# f = open("/home/dkb/workspace/Code/pretrain/second2index.txt", 'w')
# f.write(str(second2index))
# f.close()
# print("save dict successfully.")


'''
三级案由
'''
# result = data['三级案由']
# third2index = {}
# n = 1
# dicindex = 1
#
# for index, row in result.items():
#
#     if row == '':
#         continue
#     row = row.strip()
#     if '96、借用合同纠纷' == row:
#         # print(row + "||" + str(96))
#         #  直接把'借用合同'纠纷写入文件，但是这个纠纷文件中重复了两次，此处直接省略，下一次记录
#         # third2index['借用合同纠纷'] = dicindex
#         # dicindex += 1
#         continue
#     if '348、性骚扰损害责任纠纷' == row:
#         n = n-1
#     if n < 10:
#         # print(row[2:]+"||"+str(n))
#         third2index[row[2:]] = dicindex
#     elif n < 100:
#         # print(row[3:]+"||"+str(n))
#         third2index[row[3:]] = dicindex
#
#     elif n < 1000:
#         # print(row[4:]+"||"+str(n))
#         third2index[row[4:]] = dicindex
#
#     n = n+1
#     dicindex += 1
#
# f = open("/home/dkb/workspace/Code/pretrain/third2index.txt", 'w')
# f.write(str(third2index))
# f.close()
# print("save dict successfully.")
#

'''
处理四级案由
'''

# result = data['四级案由']
# forth2index = {}
# n = 1
# for index, row in result.items():
#
#     if row == '':
#         continue
#     row = row.strip()
#     # print(re.split('[)）]', row))
#     forth_content = re.split('[)）]', row,1)
#     forth2index[forth_content[1].replace("\n", "")] = n
#     n = n+1
#
# f = open("/home/dkb/workspace/Code/pretrain/forth2index.txt", 'w')
# f.write(str(forth2index))
# f.close()
# print("save dict successfully.")


'''
处理excel文件格式
这里处理后，wash.excel会出现两个"借用合同纠纷"，需要手动删除第一个
'''
# n = 1
# k = 1
#
# for index, row in data.iterrows():
#     #  按行遍历，一级案由处理
#         if row[0] == '':
#             data.iloc[index, 0] = ''
#         else:
#             row[0] = row[0].strip()
#             data.iloc[index, 0] = row[0][4:]
#
#     #  按行遍历，二级案由处理
#         if row[1] == '':
#             data.iloc[index, 1] = ''
#         else:
#             row[1] = row[1].strip()
#             if n <= 10:
#                 data.iloc[index, 1] = row[1][2:]
#                 n = n + 1
#             elif n <= 20:
#                 data.iloc[index, 1] = row[1][3:]
#                 n = n + 1
#             elif n %10 ==0:
#                 data.iloc[index, 1] = row[1][3:]
#                 n = n + 1
#             elif n < 100:
#                 data.iloc[index, 1] = row[1][4:]
#                 n = n + 1
#
#     #  按行遍历，三级案由处理
#         if row[2] == '':
#             data.iloc[index, 2] = ''
#         else:
#             row[2] = row[2].strip()
#             if('96、借用合同纠纷' == row[2]):
#                 data.iloc[index, 2] = '借用合同纠纷'
#                 continue
#             if('348、性骚扰损害责任纠纷' == row[2]):
#                 k = k-1
#             if k < 10:
#                 # print(row[2:]+"||"+str(n))
#                 data.iloc[index, 2] = row[2][2:]
#                 k = k + 1
#             elif k < 100:
#                 # print(row[3:]+"||"+str(n))
#                 data.iloc[index, 2] = row[2][3:]
#                 k = k + 1
#             elif k < 1000:
#                 # print(row[4:]+"||"+str(n))
#                 data.iloc[index, 2] = row[2][4:]
#                 k = k + 1
#     #  按行遍历，四级案由处理
#         if row[3] == '':
#             data.iloc[index, 3] = ''
#         else:
#             row[3] = row[3].strip()
#             tmp_content = re.split('[)）]', row[3], 1)
#             forth_content = tmp_content[1].replace("\n", "")
#             data.iloc[index, 3] = forth_content
#
#
# #  存入Excel
# data.to_excel('washed.xlsx', sheet_name='Sheet1', index=False, header=True)
# # print("Done")


'''
为excel每个案由打标记
这里注意改变开头的地址
'''
#
# fr1 = open("/home/dkb/workspace/Code/pretrain/first2index.txt",'r+')
# firstdic = eval(fr1.read()) #读取的str转换为字典
# fr1.close()
#
# fr2 = open("/home/dkb/workspace/Code/pretrain/second2index.txt",'r+')
# seconddic = eval(fr2.read()) #读取的str转换为字典
# fr2.close()
#
# fr3 = open("/home/dkb/workspace/Code/pretrain/third2index.txt",'r+')
# thirddic = eval(fr3.read()) #读取的str转换为字典
# fr3.close()
#
# fr4 = open("/home/dkb/workspace/Code/pretrain/forth2index.txt",'r+')
# forthdic = eval(fr4.read()) #读取的str转换为字典
# fr4.close()
#
#
# # label_object = open('/home/dkb/workspace/Code/pretrain/label_Hie.txt', 'w')
# Hie_dic={}
#
# firstline = ''
# secondline = ''
# thirdline = ''
#
# for index, row in data.iterrows():
#     if len(data.iloc[index, 0]) != 0:
#         # label_object.write(str(data.iloc[index, 0]) + "||" + str(firstdic[data.iloc[index, 0]]) + "/" + str(-1) + "/" + str(-1) + "\n")
#         Hie_dic[str(data.iloc[index, 0])]  = [firstdic[data.iloc[index, 0]],-1,-1,-1]
#         #print(str(data.iloc[index, 0]) + "||" + str(firstdic[data.iloc[index, 0]]) + "/" + str(-1) + "/" + str(-1) + "\n")
#         firstline = data.iloc[index, 0]
#
#     if len(data.iloc[index, 1]) != 0:
#         Hie_dic[str(data.iloc[index, 1])] = [firstdic[firstline], seconddic[data.iloc[index, 1]],-1,-1]
#         # label_object.write(str(data.iloc[index, 1]) + "||" + str(firstdic[firstline]) + "/" + str(seconddic[data.iloc[index, 1]]) + "/" + str(-1) + "\n")
#         #print(str(data.iloc[index, 1]) + "||" + str(firstdic[firstline]) + "/" + str(seconddic[data.iloc[index, 1]]) + "/" + str(-1) + "\n")
#         secondline = data.iloc[index, 1]
#
#     if len(data.iloc[index, 2]) != 0:
#         Hie_dic[str(data.iloc[index, 2])] = [firstdic[firstline], seconddic[secondline],thirddic[data.iloc[index, 2]],-1]
#         # label_object.write(str(data.iloc[index, 2]) + "||" + str(firstdic[firstline] )+ "/" + str(seconddic[secondline]) + "/" + str(thirddic[ data.iloc[index, 2] ]) + "\n")
#         #print(str(data.iloc[index, 2]) + "||" + str(firstdic[firstline] )+ "/" + str(seconddic[secondline]) + "/" + str(thirddic[ data.iloc[index, 2] ]) + "\n")
#         thirdline = data.iloc[index, 2]
#     if len(data.iloc[index, 3]) != 0:
#         Hie_dic[str(data.iloc[index, 3])] = [firstdic[firstline], seconddic[secondline], thirddic[thirdline],forthdic[data.iloc[index, 3]]]
#
#
# # label_object.close()
# res2 = json.dumps(Hie_dic,indent=8,ensure_ascii=False)
# print(res2)
# print(len(Hie_dic))
#
# with open("./Hie_dic.json",'w',encoding='utf-8') as f:
#     f.write(res2)
#
# print("generate label-Hie successful...")

'''
构建_hierarchical_violation的list
根据every—name文件
找出两个案由之间的切分关系
例如 9 -> 128  [(0, 15), (15, 37), (52, 20), (72, 9), (81, 7), (88, 17), (106, 14), (120, 5), (125, 3)]
'''

# dic1 = {
#         "劳动争议": 0,
#         "合同纠纷": 1,
#         "继承纠纷": 2,
#         "侵权责任纠纷": 3,
#         "物权保护纠纷": 4,
#         "保险纠纷": 5,
#         "所有权纠纷": 6
# }
dic1 = {
        "机动车交通事故责任纠纷": 0,
        "离婚纠纷": 1,
        "买卖合同纠纷": 2,
        "劳务合同纠纷": 3,
        "房屋买卖合同纠纷": 4,
        "追偿权纠纷": 5,
        "租赁合同纠纷": 6,
        "借款合同纠纷": 7,
        "生命权、健康权、身体权纠纷": 8,
        "承揽合同纠纷": 9,
        "财产损害赔偿纠纷": 10,
        "提供劳务者受害责任纠纷": 11,
        "财产保险合同纠纷": 12,
        "不当得利纠纷": 13,
        "保证合同纠纷": 14,
        "劳动合同纠纷": 15,
        "排除妨害纠纷": 16,
        "婚约财产纠纷": 17,
        "返还原物纠纷": 18,
        "离婚后财产纠纷": 19,
        "合伙协议纠纷": 20,
        "委托合同纠纷": 21,
        "运输合同纠纷": 22,
        "赡养纠纷": 23,
        "建设工程合同纠纷": 24,
        "服务合同纠纷": 25,
        "房屋拆迁安置补偿合同纠纷": 26,
        "侵害集体经济组织成员权益纠纷": 27,
        "农村土地承包合同纠纷": 28,
        "挂靠经营合同纠纷": 29,
        "土地承包经营权纠纷": 30,
        "法定继承纠纷": 31,
        "居间合同纠纷": 32,
        "相邻关系纠纷": 33,
        "供用热力合同纠纷": 34,
        "医疗损害责任纠纷": 35,
        "融资租赁合同纠纷": 36,
        "债权转让合同纠纷": 37,
        "产品责任纠纷": 38,
        "股权转让纠纷": 39,
        "人身保险合同纠纷": 40,
        "确认合同效力纠纷": 41,
        "共有纠纷": 42,
        "分家析产纠纷": 43,
        "名誉权纠纷": 44,
        "同居关系纠纷": 45,
        "遗嘱继承纠纷": 46,
        "恢复原状纠纷": 47,
        "被继承人债务清偿纠纷": 48,
        "农业承包合同纠纷": 49,
        "抚养纠纷": 50,
        "储蓄存款合同纠纷": 51,
        "广告合同纠纷": 52,
        "教育机构责任纠纷": 53,
        "债务转移合同纠纷": 54,
        "探望权纠纷": 55,
        "物权确认纠纷": 56,
        "定金合同纠纷": 57,
        "抵押合同纠纷": 58,
        "赠与合同纠纷": 59,
        "保管合同纠纷": 60,
        "饲养动物损害责任纠纷": 61,
        "占有物返还纠纷": 62,
        "股东资格确认纠纷": 63,
        "船员劳务合同纠纷": 64,
        "违反安全保障义务责任纠纷": 65,
        "一般人格权纠纷": 66,
        "婚姻无效纠纷": 67,
        "种植、养殖回收合同纠纷": 68,
        "供用水合同纠纷": 69,
        "票据追索权纠纷": 70,
        "著作权权属、侵权纠纷": 71,
        "义务帮工人受害责任纠纷": 72,
        "银行卡纠纷": 73,
        "股东知情权纠纷": 74,
        "供用电合同纠纷": 75,
        "宅基地使用权纠纷": 76,
        "林业承包合同纠纷": 77,
        "典当纠纷": 78,
        "提供劳务者致害责任纠纷": 79,
        "债权人撤销权纠纷": 80
}

dic2 = {        "民间借贷纠纷": 0,
        "金融借款合同纠纷": 1,
        "信用卡纠纷": 2,
        "物业服务合同纠纷": 3,
        "房屋租赁合同纠纷": 4,
        "追索劳动报酬纠纷": 5,
        "商品房预售合同纠纷": 6,
        "商品房销售合同纠纷": 7,
        "建设工程施工合同纠纷": 8,
        "抚养费纠纷": 9,
        "加工合同纠纷": 10,
        "小额借款合同纠纷": 11,
        "装饰装修合同纠纷": 12,
        "同居关系子女抚养纠纷": 13,
        "确认合同无效纠纷": 14,
        "所有权确认纠纷": 15,
        "变更抚养关系纠纷": 16,
        "确认劳动关系纠纷": 17,
        "定作合同纠纷": 18,
        "分期付款买卖合同纠纷": 19,
        "建筑设备租赁合同纠纷": 20,
        "财产损失保险合同纠纷": 21,
        "工伤保险待遇纠纷": 22,
        "承包地征收补偿费用分配纠纷": 23,
        "车辆租赁合同纠纷": 24,
        "赡养费纠纷": 25,
        "建设工程分包合同纠纷": 26,
        "土地租赁合同纠纷": 27,
        "公路货物运输合同纠纷": 28,
        "侵害商标权纠纷": 29,
        "经济补偿金纠纷": 30,
        "确认合同有效纠纷": 31,
        "土地承包经营权出租合同纠纷": 32,
        "修理合同纠纷": 33,
        "保险人代位求偿权纠纷": 34,
        "产品销售者责任纠纷": 35,
        "责任保险合同纠纷": 36,
        "农村建房施工合同纠纷": 37,
        "共有物分割纠纷": 38,
        "商品房预约合同纠纷": 39,
        "餐饮服务合同纠纷": 40,
        "土地承包经营权转包合同纠纷": 41,
        "侵害作品信息网络传播权纠纷": 42,
        "电信服务合同纠纷": 43,
        "同居关系析产纠纷": 44,
        "公路旅客运输合同纠纷": 45,
        "企业借贷纠纷": 46,
        "医疗服务合同纠纷": 47,
        "侵害作品放映权纠纷": 48,
        "侵害外观设计专利权纠纷": 49,
        "农村房屋买卖合同纠纷": 50,
        "意外伤害保险合同纠纷": 51,
        "相邻通行纠纷": 52,
        "旅游合同纠纷": 53,
        "网络购物合同纠纷": 54,
        "城市公交运输合同纠纷": 55,
        "凭样品买卖合同纠纷": 56}


f = open('/home/dkb/workspace/Code/pretrain/Hie_dic.json', encoding="utf-8")
Hie_vec = json.load(f)
listdic={}
restdic={}
i=0
for key1 in dic1.keys():
        vec1 = Hie_vec[key1]
        # print("**********parent:" + str(dic1[key1]))
        listchild = []
        restchild = []
        for key2 in dic2.keys():
                vec2 = Hie_vec[key2]
                if(vec1[2] == vec2[2]):# 1
                        listchild.append(dic2[key2])
                        # print("-child:"+str(dic2[key2]))
                else:
                        restchild.append(dic2[key2])
                        # print("+++rest:" + str(dic2[key2]))
        listdic[i]=listchild
        restdic[i]=restchild
        # print("child list:")
        print(listchild)
        # print("rest list:")
        # print(restchild)
        i+=1

print("##")
print(listdic)
print("Done")
