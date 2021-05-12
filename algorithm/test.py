import argparse
import os
import torch


from utils.message_utils import infor_msg, erro_msg, warning_msg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="single gpu id", required=True)
    parser.add_argument('--checkpoint', help="checkpoint file path", required=True)
    parser.add_argument('--logdir', type=str, help="logdir file path", default='logs_test/')

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
    from init.init_dataset import dataset_list
    from process.init_process import init_all
    from config.parser import create_config
    from formatter.Router import get_formatter
    from process.test_process import test_process
    from runx.logx import logx
    import datetime
    import pytz

    config = create_config(configFilePath)
    cur_time = datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y_%m_%d__%H_%M_%S")
    # cur_time = strftime("%Y_%m_%d__%H_%M_%S", gmtime())

    identifier = config.get('logging', 'identifier')
    logdir = os.path.join(args.logdir, identifier, cur_time)

    logx.initialize(logdir=logdir, coolname=True, tensorboard=True)

    # 检查 CUDA
    cuda = torch.cuda.is_available()
    logx.msg(infor_msg("CUDA available: %s" % str(cuda)))
    if not cuda and gpu_mode > 0:
        logx.msg(erro_msg("CUDA is not available but specific gpu id"))
        raise NotImplementedError

    # 初始化
    parameters = init_all(config, gpu_mode, args.checkpoint, "test")

    try:
        batch_size = config.getint("eval", "batch_size")
    except Exception as e:
        batch_size = config.getint("train", "batch_size")
        logx.msg(warning_msg(f"[eval] batch size has not been defined in config file, use [train] batch_size instead:{batch_size}"))

    try:
        shuffle = config.getboolean("eval", "shuffle")
    except Exception as e:
        shuffle = False
        logx.msg(warning_msg("[eval] shuffle has not been defined in config file, use false as default."))
    try:
        reader_num = config.getint("eval", "reader_num")
    except Exception as e:
        reader_num = config.getint("train", "reader_num")
        logx.msg(warning_msg(f"[eval] reader num has not been defined in config file, use [train] reader num instead:{reader_num}"))

    test_dataset = DataLoader(dataset=dataset_list[config.get("data", "test_dataset_type")](config, "test"),
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=reader_num,
                              pin_memory=True,
                              collate_fn=get_formatter(config, "test"))

    test_process(parameters["model"], test_dataset, parameters["trained_epoch"],
                 config, gpu_mode, parameters["report_function"], 'test')
