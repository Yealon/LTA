
from model.My_Model.Match_LSTM.Par_Match_LSTM import ParMatchLSTM
from model.My_Model.somdst.model import SomDST
from model.Seq.Seq2SeqATT import Seq2Seq_Att
from model.My_Model.Focal_loss.TextCNN_Focal import TextCNN_Focal
from model.TextCNN.TextCNN import TextCNN
from model.TextLSTM.TextLSTM import TextLSTM
from model.My_Model.Focal_loss.TextLSTM_Focal import TextLSTM_Focal
from model.Bert.Bert import BasicBert
from model.Transformer.TransModel import TransModel

model_list = {

    "TextCNN": TextCNN,
    "TextLSTM": TextLSTM,
    "Seq2Seq": Seq2Seq_Att,
    "Transformer": TransModel,
    "ParMatchLSTM": ParMatchLSTM,
    "TextCNN_Focal": TextCNN_Focal,
    "TextLSTM_Focal": TextLSTM_Focal,
    "Bert": BasicBert,
    "somdst": SomDST
}

def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError