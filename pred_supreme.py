# %%
import sys
import random
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from utils import *
from tqdm import tqdm
from trainDataloader import SupremeClsDataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
class SupremePredClsDataset(SupremeClsDataset):
    def __init__(self, tokenizer, file_name, stdid_vocab, label_id, padding_length=128):
        self.tokenizer = tokenizer
        self.label_id = label_id
        self.padding_length = padding_length
        self.data_init(file_name)
        self.cls_vocab_init(stdid_vocab)
        
    def data_init(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            self.ori_list = f.read().split('\n')
        for i in range(len(self.ori_list) - 1, -1, -1):
            std_id, exit_id, label_id, sentence = self.ori_list[i].split('\t')
            if int(label_id) != int(self.label_id):
                self.ori_list.pop(i)

    def __getitem__(self, idx):
        std_id, exit_id, label, sentence = self.ori_list[idx].strip().split('\t')
        T = self.tokenizer(sentence, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        sentence = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        std_id = 0
        return sentence, attn_mask, std_id, exit_id, label

LABEL_ID = '1'
    
tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')

myDataset = SupremePredClsDataset(tokenizer, './dataset/computed/test_with_label'.format(LABEL_ID), './dataset/supreme/l{}/std_dict'.format(LABEL_ID), LABEL_ID, 100)
dataiter = DataLoader(myDataset, batch_size=512)

# %%
config = BertConfig.from_json_file('./dataset/bert_config.json')
config.num_labels = len(myDataset.cls_label_2_id)
model = BertForSequenceClassification.from_pretrained('./model/bert_pre58_3/pytorch_model.bin', config=config)

model.cuda()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[2, 3]).cuda()
model.to(device)

model_dict = torch.load("./model/supreme/l{}/bert_supreme_12.pth".format(LABEL_ID)).module.state_dict()
model.module.load_state_dict(model_dict)

# %%
with torch.no_grad():
    eval_iter = tqdm(dataiter)
    for sentences, attn_masks, _, ext_ids, _ in eval_iter:
        model.eval()
        if torch.cuda.is_available():
            sentences = Variable(sentences.cuda(2))
            attn_masks = Variable(attn_masks.cuda(2))
        else:
            sentences = Variable(sentences)
            attn_masks = Variable(attn_masks)

        outputs = model(sentences, attention_mask=attn_masks)
        logits = outputs[0]

        pred_scores = F.softmax(logits, -1)
        pred = pred_scores.max(-1)[1]

        result = ''
        for idx, item in enumerate(ext_ids):
            result += '{},{}\n'.format(item, myDataset.cls_id_2_label[pred[idx].data.item()])
        with open('./supreme/l{}'.format(LABEL_ID), encoding='utf-8', mode='a+') as f:
            f.write(result)
            

# %%
