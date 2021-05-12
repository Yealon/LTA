import json
import jieba
from torch.utils.data import Dataset


class JsonFromTerminalDataset(Dataset):
    def __init__(self, config, mode,data, encoding="utf8"):

        self.config = config
        self.mode = mode

        self.encoding = encoding
        # 如果为line代表一行一个json数据；如果为single代表整个文件为一个json数据。

        self.data = []

        # {\"text\": \"\原 告诉 称 与 被告 在 2010 年初 通过 网络 相识 于 当年 7 月 26 日 在 安徽省 宿松县 民政局 登记 结婚 2011 年 2 月 20 日 生育 一子 唐 经济
        # 付出 均 不够 被告 有时 更换 电话号码 不 告知 原告 有时 联系 不上 被告 2014 年 5 月 双方 因 买房 问题 产生 重大 分歧 后 感情 越来越 差 为底 破裂 其次 认为 自己 有 稳定 的
        # 工作 和 收入 被告 系 再婚 且 在 第一次 婚姻 中育 有 孩子 平时 对 儿子 照顾 较少 故 认为 离婚 后 儿子 判由 原联系 甚 少 夫妻感情 无 和 好 可能 原告 为 维护 原告 合法权益 故
        # 再次 诉讼 来院 请求 判令 1 准许 原被告 离婚 2 儿子 唐某 乙随 原告 共同 生活 由担",
        # \"law\": \"\",,32\"\", \"label\": \"婚姻家庭、继承纠纷 婚姻家庭纠纷 离婚纠纷\"}

        sp_data = jieba.cut(data)
        outstr = ''
        for word in sp_data:
            i = 0
            if word != '\t' and word != '\n':
                i += 1
                if i > 500:
                    break
                outstr += word
                outstr += " "
        # in_p = '{\"text\": \"' + outstr + "\", \"law\": \"\",,32\"\", \"label\": \"婚姻家庭、继承纠纷 婚姻家庭纠纷 离婚纠纷\"}"
        in_p = {
            "text":outstr,
            "label": "dummy1 dummy2 dummy3"
        }
        jstr = json.dumps(in_p, ensure_ascii=False)
        self.data.append(json.loads(jstr))


    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)