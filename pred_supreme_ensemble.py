# %%
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
from utils import *
from tqdm import tqdm
from trainDataloader import SimDataset, EvalSimDataset, EvalSimWithLabelDataset, SupremeClsDataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
LABEL_ID = '1'
tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')

eval_list = load_sim_dev('./dataset/101/c_dev_with_label')
for i in range(len(eval_list) - 1, -1, -1):
    if int(eval_list[i][2]) != int(LABEL_ID):
        eval_list.pop(i)

# 主要用于统计类别std_id数量 #
myDataset = SupremeClsDataset(tokenizer, './dataset/computed/test_with_label'.format(LABEL_ID), './dataset/supreme/l{}/std_dict'.format(LABEL_ID), 100)

# %%
EVAL_EPOCH = [12, 22]

config = BertConfig.from_json_file('./dataset/bert_config.json')
config.num_labels = len(myDataset.cls_label_2_id)
_config = BertConfig.from_json_file('./dataset/bert_config.json')
_config.num_labels = len(myDataset.cls_label_2_id)
model = BertForSequenceClassification(config=config)
_model = BertForSequenceClassification(config=_config)

model.cuda()
_model.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
_model = torch.nn.DataParallel(_model, device_ids=[0, 1, 2, 3]).cuda()
model.to(device)
_model.to(device)

model_dict = torch.load("./model/supreme/l{}/bert_supreme_{}.pth".format(LABEL_ID, EVAL_EPOCH[0])).module.state_dict()
model.module.load_state_dict(model_dict)

_model_dict = torch.load("./model/supreme/l{}/bert_supreme_{}.pth".format(LABEL_ID, EVAL_EPOCH[1])).module.state_dict()
_model.module.load_state_dict(_model_dict)

# %%
with torch.no_grad():
    model.eval()
    _model.eval()
    eval_correct_num = 0
    eval_list_iter = tqdm(eval_list)
    for idx, item in enumerate(eval_list_iter):
        eval_list_iter.set_description('{}/{}'.format(idx + 1, len(eval_list)))
        eval_list_iter.set_postfix(correct_num=eval_correct_num, eval_acc=eval_correct_num / (idx + 1))

        T = tokenizer(item[3], add_special_tokens=True, max_length=100, padding='max_length', truncation=True)
        outputs = model(torch.tensor(T['input_ids']).unsqueeze(0).cuda(), attention_mask=torch.tensor(T['attention_mask']).unsqueeze(0).cuda())
        _outputs = _model(torch.tensor(T['input_ids']).unsqueeze(0).cuda(), attention_mask=torch.tensor(T['attention_mask']).unsqueeze(0).cuda())
        logits = outputs[0]
        _logits = _outputs[0]
        pred_scores = F.softmax(logits, -1)
        _pred_scores = F.softmax(logits, -1)

        pred = ((pred_scores + _pred_scores) / 2).max(-1)[1]
        max_item_id = myDataset.cls_id_2_label[pred[0].data.item()]
        if int(max_item_id) == int(item[0]):
            eval_correct_num += 1
    print('Eval_acc: {}\n'.format(eval_correct_num / len(eval_list)))

# %%
