import numpy as np
from transformers import BertTokenizer
import Levenshtein

from utils.dict_utils import Dict


def transfer_one_hot_label(l, size):

    result = np.zeros(size)
    # try:
    if not None in l:
        result[l] = 1
    # except Exception as e:
    #     logger.warning("[mismatch] %s" %e)

    return result


def SOM_output(fun):

    def wrapper(*args, **kwargs):
        out_data = fun(*args, **kwargs)

        mode = out_data['mode']
        y_pre = out_data['y_pre']
        config = out_data['config']
        y = out_data['y']

        char_dic = Dict('/home/dkb/workspace/Code/nlp_data/Bert/char.dic')
        file_path = config.get("model", "eval_vocab_path")
        label_dict = Dict(file_path)


        indices = []
        orign_out = []
        for gen_list in y_pre:
            g = BertTokenizer.from_pretrained(config.get("model", "bert_path")).convert_ids_to_tokens(gen_list)
            gen = ""
            for gg in g:
                if gg == '[EOS]':
                    break
                elif gg not in char_dic.convertToLabels(range(char_dic.size()), None):
                    gg = 'N'
                gen += gg
            # indices += [gen.replace(' ##', '')]
            min_index = -1
            temp_min = 100000
            gen = gen.replace(' ##', '')
            if 'test' == mode:
                orign_out += [gen]

            for i in range(0, label_dict.size()):
                temp = Levenshtein.distance(gen, label_dict.getLabel(i))
                if temp < temp_min:
                    temp_min = temp
                    min_index = i
            # indices += [label_dict.getLabel(min_index)]
            indices += [min_index]
        # todo
        indices = np.array(indices)
        one_hot_predicted_labels = np.array([transfer_one_hot_label([y], label_dict.size()) for y in indices])
        one_hot_label = [s.cuda().data.cpu().numpy() for s in y]

        if 'test' == mode:
            raw_tgt = out_data['raw_tgt']
            test_predict = [label_dict.convertToLabels(np.argwhere(sa > 0), '</s>') for sa in one_hot_predicted_labels]
            test_target = [s for s in raw_tgt]

            return {'y': one_hot_label,
                    'y_pre': one_hot_predicted_labels,
                    'test_predict': test_predict,
                    'test_target': test_target,
                    'orign_out': orign_out}
        else:
            return {'y': one_hot_label,
                    'y_pre': one_hot_predicted_labels}

    return wrapper


