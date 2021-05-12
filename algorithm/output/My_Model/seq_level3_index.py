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


def seq_level3_index(fun):

    def wrapper(*args, **kwargs):
        out_data = fun(*args, **kwargs)

        y_pre = out_data['y_pre']
        config = out_data['config']
        label_dict = out_data['label_dict']
        y = out_data['y']

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

            if len(pres) < 3:
                pres.append(None)
            else:
                pres[2] = pres[2] - 22

            pre_list.append(transfer_one_hot_label([pres[2]], label_dict.size()))

        pre_list = np.array(pre_list)

        one_hot_label = [s.cuda().data.cpu().numpy() for s in y]
        return {'y': one_hot_label,
                'y_pre': pre_list}

    return wrapper

