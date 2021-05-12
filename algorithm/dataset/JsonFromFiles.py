import json
from torch.utils.data import Dataset

class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8"):

        self.config = config
        self.mode = mode
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding
        # 如果为line代表一行一个json数据；如果为single代表整个文件为一个json数据。
        self.json_format = config.get("data", "json_format")
        self.data = []

        if self.json_format == "single":
            self.data = self.data + json.load(open(self.data_path, "r", encoding=encoding))
        else:
            f = open(self.data_path, "r", encoding=encoding)
            for line in f:
                self.data.append(json.loads(line))

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)