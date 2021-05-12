import torch
from torch.autograd import Variable
from runx.logx import logx

from utils.message_utils import infor_msg, warning_msg, erro_msg


def test_process(model, dataset, epoch, config, gpu_mode, report_function, mode):
    model.eval()
    logx.msg(infor_msg(f"=============={mode} Start....==============="))

    step = -1

    report_prepare = {}

    for step, data in enumerate(dataset):
        for key in data.keys():
            # 将经过Formatter处理后的tensor变量封装为Variable
            if isinstance(data[key], torch.Tensor):
                if gpu_mode > 0:
                    data[key] = Variable(data[key].cuda(non_blocking=True))
                else:
                    data[key] = Variable(data[key])

        model_test = model.module if hasattr(model, 'module') else model
        results = model_test.valid(data, mode)

        for k in results.keys():
            try:
                report_prepare[k] += results[k]
            except KeyError:
                logx.msg(warning_msg(f"new key:{k}"))
                report_prepare[k] = results[k]

    if step == -1:
        logx.msg(erro_msg("There is no data given to the model in this epoch, check your data."))
        raise NotImplementedError

    test_metric = report_function(report_prepare, epoch, config, mode)
    logx.msg(infor_msg(f"=============={mode} End==============="))
    return test_metric

