from formatter.BasicFormatter import BasicFormatter
from formatter.My_Model.LSTM_Match.DummyFormatter import LSTMDummyFormatter
from formatter.My_Model.LSTM_Match.Match_Formatter import Match_Formatter
from formatter.Seq2Seq.SeqFormatter import SeqFormatter
from formatter.TextCNN.CNNFormatter import CNNFormatter
from formatter.TextLSTM.LSTMFormatter import LSTMFormatter
from formatter.Transformer.TransFormatter import TransFormatter
from formatter.Bert.BertFormatter import BasicBertFormatter
from formatter.My_Model.som.SomFormatter import SomFormatter
from runx.logx import logx
from utils.message_utils import gen_time_str, warning_msg, infor_msg, erro_msg, epoch_msg, correct_msg

formatter_list = {
    "Basic": BasicFormatter,
    "CNNFormatter": CNNFormatter,
    "LSTMFormatter":LSTMFormatter,
    "SeqFormatter": SeqFormatter,
    "TransFormatter": TransFormatter,
    "BertFormatter": BasicBertFormatter,
    "Match_Formatter": Match_Formatter,
    "LSTMDummyFormatter": LSTMDummyFormatter,
    "SomFormatter": SomFormatter
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logx.msg(warning_msg(f'[data] {temp_mode}_formatter_type has not been defined in config file, use [data] train_formatter_type instead."'))
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logx.msg(erro_msg(f"There is no formatter called {which}, check your config."))
        raise NotImplementedError
