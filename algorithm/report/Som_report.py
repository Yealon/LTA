import numpy as np
import os
from utils.dict_utils import Dict
from runx.logx import logx
import warnings
import sklearn

from utils.message_utils import report_msg, infor_msg, warning_msg, correct_msg

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def report_Som(report_prepare, epoch, config, mode):
    y_pre = report_prepare['y_pre']
    y = report_prepare['y']


    y_pre = np.array(y_pre)
    y = np.array(y)

    macro_f1 = sklearn.metrics.f1_score(y, y_pre, average='macro')
    macro_precision = sklearn.metrics.precision_score(y, y_pre, average='macro')
    macro_recall = sklearn.metrics.recall_score(y, y_pre, average='macro')
    micro_f1 = sklearn.metrics.f1_score(y, y_pre, average='micro')
    micro_precision = sklearn.metrics.precision_score(y, y_pre, average='micro')
    micro_recall = sklearn.metrics.recall_score(y, y_pre, average='micro')
    Accuracy = sklearn.metrics.accuracy_score(y, y_pre)


    label_dict = Dict(config.get("model", "eval_vocab_path"))
    dict_sorted = label_dict.convertToLabels(range(label_dict.size()), '</s>')
    results = np.array(dict_sorted)
    detail = sklearn.metrics.classification_report(y, y_pre, target_names=results, digits=4)

    # return {
    #     "macro_f1": macro_f1,
    #     "macro_precision": macro_precision,
    #     "macro_recall": macro_recall,
    #     "micro_f1": micro_f1,
    #     "micro_precision": micro_precision,
    #     "micro_recall": micro_recall,
    #     "acc": acc,
    #     "detail": detail
    # }

    logx.msg(report_msg(f'Accuracy: {Accuracy}'))
    logx.msg(report_msg(f'macro_precision: {macro_precision}, macro_recall: {macro_recall}, macro_f1: {macro_f1}'))
    logx.msg(report_msg(f'micro_precision: {micro_precision}, micro_recall: {micro_recall}, micro_f1: {micro_f1}'))
    logx.msg(report_msg(f'detail: \n{detail}'))

    metrics = {'macro_f1': float(macro_f1), 'Accuracy': float(Accuracy), 'micro_f1': float(micro_f1)}
    logx.metric(mode, metrics, epoch)

    # print(f'Accuracy: {acc} \n')
    # print(mode + " on %d epoch \n" % epoch)
    # print(
    #     'macro_precision : %f      macro_recall : %f      macro_f1 : %f \n' % (macro_precision, macro_recall, macro_f1))
    # print(
    #     'micro_precision : %f      micro_recall : %f      micro_f1 : %f \n' % (micro_precision, micro_recall, micro_f1))
    # print(mode + " report: \n")
    # print(sklearn.metrics.classification_report(y, y_pre, target_names=results, digits=4))

    if 'test' == mode:
        test_predict = report_prepare['test_predict']
        test_target = report_prepare['test_target']
        orign_out = report_prepare['orign_out']

        ref_dir = os.path.join(config.get("output", "eval_output"), config.get("output", "model_name"))
        if os.path.exists(ref_dir):
            logx.msg(warning_msg(f'Dir {ref_dir} path exists, Please change the name of model'))
        else:
            logx.msg(correct_msg(f'Dir path is available'))

        os.makedirs(ref_dir, exist_ok=True)

        ref_file = os.path.join(ref_dir, "result%d.txt" % epoch)
        if os.path.exists(ref_file):
            logx.msg(warning_msg(f'File {ref_dir} path exists, Please check the epoch'))
        else:
            logx.msg(correct_msg(f'File path is available'))

        logx.msg(infor_msg(f'Start Saving Test results to {ref_file}...'))
        with open(ref_file, mode='a') as filename:
            length = min(len(test_target), len(test_predict))
            for i in range(length):
                filename.write(" |target| " + "".join(str(test_target[i]).strip()) + '\n')
                filename.write(" |output| " + "".join(str(test_predict[i]).strip()) + '\n')
                filename.write(" |orign_out| " + "".join(str(orign_out[i]).strip()) + '\n')
        logx.msg(infor_msg(f'Test Done, Test results File dir is {ref_file}, Good Luck! :)'))

    return metrics