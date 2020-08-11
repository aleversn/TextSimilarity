# %%
import sys
import json
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
GLOBAL_DICT = {
    '15': {
        'dict': [15, 16],
        'path': './model/judger/15x16/bert_judger_15x16__4.pth'
    }
}
LABEL_ID = '1'
EVAL_EPOCH = 12
CUR_PATH = ''
tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')

eval_list = load_sim_dev('./dataset/101/c_dev_with_label')
for i in range(len(eval_list) - 1, -1, -1):
    if int(eval_list[i][2]) != int(LABEL_ID):
        eval_list.pop(i)

# 主要用于统计类别std_id数量 #
myDataset = SupremeClsDataset(tokenizer, './dataset/computed/test_with_label'.format(LABEL_ID), './dataset/supreme/l{}/std_dict'.format(LABEL_ID), 100)


# %%
config = BertConfig.from_json_file('./dataset/bert_config.json')
config.num_labels = len(myDataset.cls_label_2_id)
judger_config = BertConfig.from_json_file('./dataset/bert_config.json')
judger_config.num_labels = 2
model = BertForSequenceClassification(config=config)
model_1 = BertForSequenceClassification(config=judger_config)

model.cuda()
model_1.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
model_1 = torch.nn.DataParallel(model_1, device_ids=[0, 1, 2, 3]).cuda()
model.to(device)
model_1.to(device)

model_dict = torch.load("./model/supreme/l{}/bert_supreme_{}.pth".format(LABEL_ID, EVAL_EPOCH)).module.state_dict()
model.module.load_state_dict(model_dict)

# %%
with torch.no_grad():
    model.eval()
    model_1.eval()
    correct_dict = []
    err_dict = []
    eval_correct_num = 0
    eval_list_iter = tqdm(eval_list)
    for idx, item in enumerate(eval_list_iter):
        cur_eval_result_scores = torch.tensor([])
        eval_list_iter.set_description('{}/{}'.format(idx + 1, len(eval_list)))
        eval_list_iter.set_postfix(correct_num=eval_correct_num, eval_acc=eval_correct_num / (idx + 1))

        T = tokenizer(item[3], add_special_tokens=True, max_length=100, padding='max_length', truncation=True)
        outputs = model(torch.tensor(T['input_ids']).unsqueeze(0).cuda(), attention_mask=torch.tensor(T['attention_mask']).unsqueeze(0).cuda())
        logits = outputs[0]
        pred = logits.max(-1)[1]
        pred_scores = F.softmax(logits, -1)

        max_item_id = myDataset.cls_id_2_label[pred[0].data.item()]
        if(str(max_item_id) in GLOBAL_DICT):
            judger = GLOBAL_DICT[str(max_item_id)]
            if CUR_PATH != judger['path']:
                md = torch.load(judger['path']).module.state_dict()
                CUR_PATH = judger['path']
                model_1.module.load_state_dict(md)
            outputs_ = model_1(torch.tensor(T['input_ids']).unsqueeze(0).cuda(), attention_mask=torch.tensor(T['attention_mask']).unsqueeze(0).cuda())
            logits_ = outputs_[0]
            pred_scores_ = F.softmax(logits_, -1)
            pred_ = pred_scores_.max(-1)[1]
            max_item_id = judger['dict'][pred_[0]]
        if int(max_item_id) == int(item[0]):
            eval_correct_num += 1
            correct_dict.append({
                'correct_id': item[0],
                'pred_ids': [myDataset.cls_id_2_label[i] for i in range(len(myDataset.cls_id_2_label))],
                'scores': pred_scores[0].tolist()
            })
        else:
            err_dict.append({
                'correct_id': item[0],
                'pred_ids': [myDataset.cls_id_2_label[i] for i in range(len(myDataset.cls_id_2_label))],
                'scores': pred_scores[0].tolist()
            })
    # with open('correct.json', encoding='utf-8', mode='w+') as f:
    #     f.write(json.dumps(correct_dict))
    # with open('err.json', encoding='utf-8', mode='w+') as f:
    #     f.write(json.dumps(err_dict))
    print('Eval_acc: {}\n'.format(eval_correct_num / len(eval_list)))

# %%
