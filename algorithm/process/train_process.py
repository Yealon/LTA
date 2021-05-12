"""''''''''''''''''''''''''''''''''''''''
'           # 模型训练与记录 #
'
'  训练模型，每个epoch储存checkpoint并在验证集进行验证
'  每'config.getint("output", "output_time")'个batch，输出loss信息
'
''''''''''''''''''''''''''''''''''''''"""

from runx.logx import logx
import torch
from torch.autograd import Variable

from timeit import default_timer as timer
from torch.utils.data import DataLoader

from init.init_lr_scheduler import init_lr_scheduler
from utils.message_utils import gen_time_str, warning_msg, infor_msg, erro_msg, epoch_msg, correct_msg
from init.init_dataset import dataset_list
from formatter.Router import route_formatter
from formatter.Router import get_formatter
from process.test_process import test_process



def checkpoint(model, optimizer, trained_epoch, config, global_step, metric):
    """
    储存checkpoint文件
    :param model:           model
    :param optimizer:       optimizer
    :param trained_epoch:   本次训练的epoch数，从1开始计数
    :param config:          config
    :param global_step:     全局训练轮次， 从0开始计数

    """
    if logx.rank0:
        model_to_save = model.module if hasattr(model, 'module') else model
        save_params = {
            "model": model_to_save.state_dict(),
            "optimizer_name": config.get("optim", "optimizer"),
            "optimizer": optimizer.state_dict(),
            "trained_epoch": trained_epoch,
            "global_step": global_step
        }

        try:
            logx.save_model(
                    save_params,
                    metric = metric,
                    epoch=trained_epoch,
                    higher_better=False,
                    delete_old = False)
            logx.msg(correct_msg(f"epoch{trained_epoch} save done, file path:{str(logx.save_ckpt_fn)}"))

        except Exception as e:
            logx.msg(warning_msg("Cannot save models with error:[ %s], continue anyway" % str(e)))


def train_process(parameters, config, gpu_mode, do_test):
    """
    模型的详细训练过程
    """
    # # dataset
    which_train = config.get("data", "train_dataset_type")
    which_test = config.get("data", "test_dataset_type")
    which_valid = config.get("data", "valid_dataset_type")
    # if which in dataset_list:
    train_data = dataset_list[which_train](config, "train")
    test_data = dataset_list[which_test](config, "test")
    valid_data = dataset_list[which_valid](config, "valid")

    batch_size = config.getint("train", "batch_size")
    shuffle = config.getboolean("train", "shuffle")
    reader_num = config.getint("train", "reader_num")

    if 2 == gpu_mode:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        dataset = DataLoader(dataset=train_data,
                             batch_size=batch_size,
                             num_workers=reader_num,
                             collate_fn=get_formatter(config, "train"),
                             pin_memory=True,
                             sampler=train_sampler)
    else:
        dataset = DataLoader(dataset=train_data,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=reader_num,
                             collate_fn=get_formatter(config, "train"),
                             pin_memory=True)
    # eval
    try:
        batch_size = config.getint("eval", "batch_size")
    except Exception as e:
        logx.msg(warning_msg(f"[eval] batch size has not been defined in config file, use [train] batch_size instead:{batch_size}"))
    try:
        shuffle = config.getboolean("eval", "shuffle")
    except Exception as e:
        shuffle = False
        logx.msg(warning_msg("[eval] shuffle has not been defined in config file, use false as default."))
    try:
        reader_num = config.getint("eval", "reader_num")
    except Exception as e:
        logx.msg(warning_msg(f"[eval] reader num has not been defined in config file, use [train] reader num instead:{reader_num}"))

    valid_dataset = DataLoader(dataset=valid_data,
                               batch_size=batch_size,
                               num_workers=reader_num,
                               shuffle=shuffle,
                               collate_fn=get_formatter(config, "valid"),
                               pin_memory=True)

    # # dataset
    epoch = config.getint("train", "epoch") + 1

    output_time = config.getint("output", "output_time")

    # 训练参数
    trained_epoch = parameters["trained_epoch"] + 1  # 初始化为 1
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    global_step = parameters["global_step"]
    report_function = parameters["report_function"]

    # # optimizer 参数
    # if not config.getboolean("train", "use_my_scheduler"):
    #     step_size = config.getint("train", "step_size")
    #     gamma = config.getfloat("train", "lr_multiplier")
    #     exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # elif config.get("train", "optimizer") == "Som":
    #     optimizer.add_sch(config, len(dataset))

    exp_lr_scheduler = init_lr_scheduler(optimizer, config, len(dataset), model=model)

    logx.msg(infor_msg("==============Training start....=============="))

    total_len = len(dataset)

    # 开始训练，epoch范围：[trained_epoch, config.train.epoch + 1)
    for epoch_num in range(trained_epoch, epoch):
        # #############一个epoch开始#############
        start_time = timer()
        current_epoch = epoch_num
        if 2 == gpu_mode:
            train_sampler.set_epoch(epoch_num)

        total_loss = 0

        step = -1

        for step, data in enumerate(dataset):
            # ############# 一个step开始 #############
            for key in data.keys():
                # 将经过Formatter处理后的tensor变量封装为Variable
                if isinstance(data[key], torch.Tensor):
                    if gpu_mode > 0:
                        data[key] = Variable(data[key].cuda(non_blocking=True))
                    else:
                        data[key] = Variable(data[key])

            model.zero_grad()

            loss = model(data)

            loss.backward()

            loss = loss.item()

            total_loss += float(loss)

            optimizer.step()
            if 'step' == config.get("optim", "update_scheduler") and exp_lr_scheduler is not None:
                exp_lr_scheduler.step()

            if step % output_time == 0:
                delta_t = timer() - start_time

                logx.msg(epoch_msg(Epoch=current_epoch,
                               Stage='train',
                               Iterations="%d/%d" % (step + 1, total_len),
                               Time_Usage="%s/%s" % (gen_time_str(delta_t),gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                               Loss="%.3lf" % (total_loss / (step + 1)),
                               ))

            global_step += 1

            # ############# 一个step结束 #############
        if 'epoch' == config.get("optim", "update_scheduler")and exp_lr_scheduler is not None:
            exp_lr_scheduler.step()

        # 最后一个step的输出
        delta_t = timer() - start_time
        logx.msg(epoch_msg(Epoch=current_epoch,
                       Stage='train',
                       Iterations="%d/%d" % (step + 1, total_len),
                       Time_Usage="%s/%s" % (
                       gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                       Loss="%.3lf" % (total_loss / (step + 1)),
                       ))

        if step == -1:
            logx.msg(erro_msg("There is no data given to the model in this epoch, check your data."))
            raise NotImplementedError

        # 每个epoch进行一次验证
        if logx.rank0:
            with torch.no_grad():
                test_metric = test_process(model, valid_dataset, current_epoch, config, gpu_mode, report_function, 'validate')
                model.train()
        # 每个epoch存一次 tensorboard 和 checkpoint, 这里的logx.metric只存tensorboard，不存入metrics.csv，csv文件只存validate/test阶段的指标结果
            checkpoint(model, optimizer, current_epoch, config, global_step,test_metric[config.get('report', 'metric')])

        logx.metric('train', {'loss': float(total_loss) / (step + 1)}, current_epoch)

        # #############一个epoch结束#############

# 所有epoch结束
    with torch.no_grad():
        if do_test and logx.rank0:
            # 若训练中进行测试，初始化测试集
            route_formatter(config, ["test"])
            test_dataset = DataLoader(dataset=test_data,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=reader_num,
                                      pin_memory=True,
                                      collate_fn=get_formatter(config, "test"))

            test_process(model, test_dataset, epoch - 1, config, gpu_mode, report_function, 'test')
