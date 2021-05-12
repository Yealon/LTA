import argparse
import os

from dataset.JsonFromTerminalDataset import JsonFromTerminalDataset
from formatter.Router import get_formatter
from utils.dict_utils import Dict
from utils.message_utils import infor_msg, erro_msg, warning_msg, correct_msg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id")
    parser.add_argument('--checkpoint', help="checkpoint file path", required=True)
    args = parser.parse_args()

    configFilePath = args.config

    gpu_mode = 0
    if args.gpu is None:
        print(erro_msg("Do not support cpu version, please use gpu"))
        raise NotImplementedError
    else:
        gpu_mode = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.system("clear")
    from torch.utils.data import DataLoader
    from config.parser import create_config
    from process.init_process import init_all

    import torch
    from torch.autograd import Variable
    import numpy as np
    from runx.logx import logx
    import datetime
    import pytz

    config = create_config(configFilePath)
    cur_time = datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y_%m_%d__%H_%M_%S")
    # cur_time = strftime("%Y_%m_%d__%H_%M_%S", gmtime())

    identifier = config.get('logging', 'identifier')
    logdir = os.path.join('logs_predict/', identifier, cur_time)

    logx.initialize(logdir=logdir, coolname=True, tensorboard=True)


    config = create_config(configFilePath)

    cuda = torch.cuda.is_available()
    logx.msg(infor_msg("CUDA available: %s" % str(cuda)))
    if not cuda and gpu_mode > 0:
        logx.msg(erro_msg("CUDA is not available but specific gpu id"))
        raise NotImplementedError

    parameters = init_all(config, gpu_mode, args.checkpoint, "test")

    batch_size = 1

    logx.msg(infor_msg("please input here!"))
    dataaa = input()

    predict_dataset = DataLoader(dataset=JsonFromTerminalDataset(config, "test", dataaa),
                                 batch_size=batch_size,
                                 pin_memory=True,
                                 collate_fn=get_formatter(config, "test"))


    model = parameters["model"]
    dataset = predict_dataset
    epoch =  parameters["trained_epoch"]
    # output_function = parameters["output_function"]


    model.eval()
    logx.msg(infor_msg("Predict Start\n"))

    try:
        file_path = config.get("model", "eval_vocab_path")
        label_dict = Dict(file_path)
    except Exception as e:
        label_dict = None
        logx.msg(erro_msg("[eval] eval_vocab_path has not been defined in config file, use None as default."))

    # eval_label_dict = Dict(config.get("model", "eval_vocab_path"))

    step = -1

    target, predict = [], []
    test_target, test_predict = [], []
    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if gpu_mode > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])

        model_test = model.module if hasattr(model, 'module') else model
        results = model_test.valid(data, "test")

        # eval_prepare = output_function(data['tgt'], results, label_dict, config)

        test_predict += [label_dict.convertToLabels(np.argwhere(sa > 0), '</s>') for sa in results['y_pre']]
        test_target += [s for s in data['raw_tgt']]
        logx.msg(correct_msg(test_predict))

    if step == -1:
        logx.msg(erro_msg("There is no data given to the model in this epoch, check your data."))
        raise NotImplementedError
