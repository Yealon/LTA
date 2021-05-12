import torch as torch
import numpy as np
from utils.dict_utils import Dict


def threshold_onehot(fun):

    def wrapper(*args, **kwargs):
        out_data = fun(*args, **kwargs)
        mode = out_data['mode']
        y_pre = out_data['y_pre']
        config = out_data['config']
        y = out_data['y']


        file_path = config.get("model", "eval_vocab_path")
        label_dict = Dict(file_path)

        scores = y_pre.data
        threshold = config.getfloat("output", "threshold")
        one_hot_predicted_labels = torch.gt(scores, threshold).data.int()
        one_hot_predicted_labels = one_hot_predicted_labels.cuda().data.cpu().numpy()

        one_hot_predicted_labels = [sa for sa in one_hot_predicted_labels]
        one_hot_label = [s.cuda().data.cpu().numpy() for s in y]

        if 'test' == mode:
            raw_tgt = out_data['raw_tgt']
            test_predict = [label_dict.convertToLabels(np.argwhere(sa > 0), '</s>') for sa in one_hot_predicted_labels]
            test_target = [s for s in raw_tgt]

            return {'y': one_hot_label,
                    'y_pre': one_hot_predicted_labels,
                    'test_predict': test_predict,
                    'test_target': test_target}
        else:
            return {'y': one_hot_label,
                    'y_pre': one_hot_predicted_labels}

    return wrapper