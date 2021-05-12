import numpy as np
import utils.dict_utils as Dict_F

def transfer_one_hot_label(l, size):

    result = np.zeros(size)
    # try:
    if not None in l:
        result[l] = 1
    # except Exception as e:
    #     logger.warning("[mismatch] %s" %e)

    return result


def seq_index(fun):

    def wrapper(*args, **kwargs):
        out_data = fun(*args, **kwargs)

        mode = out_data['mode']
        y_pre = out_data['y_pre']
        config = out_data['config']
        y = out_data['y']

        file_path = config.get("model", "eval_vocab_path")
        label_dict = Dict_F.Dict(file_path)

        y_pre_indexs = y_pre

        pre_list = []
        for k in range(len(y)):
            pres = []

            for i in y_pre_indexs[k]:
                if Dict_F.EOS == i:
                    break
                else:
                    if i - 4 >= 0:
                        pres += [i - 4]
                    else:
                        continue
                    # pres += [i - 4 if i - 4 >= 0 else continue]
            # pre_list.append(pres)
            pre_list.append(transfer_one_hot_label(pres, label_dict.size()))

        pre_list = np.array(pre_list)

        pre_list = [sa for sa in pre_list]
        one_hot_label = [s.cuda().data.cpu().numpy() for s in y]

        if 'test' == mode:
            raw_tgt = out_data['raw_tgt']
            test_predict = [label_dict.convertToLabels(np.argwhere(sa > 0), '</s>') for sa in pre_list]
            test_target = [s for s in raw_tgt]

            return {'y': one_hot_label,
                    'y_pre': pre_list,
                    'test_predict': test_predict,
                    'test_target': test_target}
        else:
            return {'y': one_hot_label,
                    'y_pre': pre_list}


    return wrapper
