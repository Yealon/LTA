from init.init_formatter import init_formatter
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

formatter = {}


def route_formatter(config, task_list, *args, **params):
    for task in task_list:
        formatter[task] = init_formatter(config, task, *args, **params)


def get_formatter(config, task, *args, **params):
    def train_collate_fn(data):
        return formatter["train"].process(data, config, "train")

    def valid_collate_fn(data):
        return formatter["valid"].process(data, config, "valid")

    def test_collate_fn(data):
        return formatter["test"].process(data, config, "test")

    if task == "train":
        return train_collate_fn
    elif task == "valid":
        return valid_collate_fn
    else:
        return test_collate_fn