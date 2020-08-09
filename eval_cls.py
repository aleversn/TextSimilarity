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
from trainDataloader import ClsDataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')

myDataset = ClsDataset(tokenizer, './dataset/101/cls_train', './dataset/cls_dict', 100)
dataiter = DataLoader(myDataset, batch_size=120)

myData_eval = ClsDataset(tokenizer, './dataset/101/cls_test', './dataset/cls_dict', 100)
dataiter_eval = DataLoader(myData_eval, batch_size=120)

# %%
config = BertConfig.from_json_file('./dataset/bert_config.json')
config.num_labels = len(myDataset.cls_label_2_id)
model = BertForSequenceClassification(config=config)
model_1 = BertForSequenceClassification(config=config)

model.cuda()
model_1.cuda()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[2, 3]).cuda()
model_1 = torch.nn.DataParallel(model, device_ids=[2, 3]).cuda()
model.to(device)
model_1.to(device)

model_dict = torch.load("./model/bert_cls/bert_58_cls8.pth").module.state_dict()
model.module.load_state_dict(model_dict)
model_dict_2 = torch.load("./model/bert_cls/bert_58_cls13.pth").module.state_dict()
model_1.module.load_state_dict(model_dict_2)

# %%

with torch.no_grad():
    eval_count = 0
    eval_loss = 0
    eval_acc = []
    eval_iter = tqdm(dataiter_eval)
    for sentences, attn_masks, labels in eval_iter:
        model.eval()
        model_1.eval()
        if torch.cuda.is_available():
            sentences = Variable(sentences.cuda(2))
            attn_masks = Variable(attn_masks.cuda(2))
            labels = Variable(labels.cuda(2))
        else:
            sentences = Variable(sentences)
            attn_masks = Variable(attn_masks)
            labels = Variable(labels)

        outputs = model(sentences, attention_mask=attn_masks, labels=labels)
        loss, logits = outputs[:2]

        outputs_2 = model_1(sentences, attention_mask=attn_masks, labels=labels)
        loss_2, logits_2 = outputs_2[:2]
        
        loss = loss.mean()
        
        logits = logits + logits_2
        pred_scores = F.softmax(logits, -1)
        pred = pred_scores.max(-1)[1]

        eval_loss += loss.data.item()
        eval_count += 1

        eval_acc.append((pred == labels).sum().data.item() / pred.shape[0])
        
        eval_iter.set_postfix(eval_loss=eval_loss / eval_count, eval_acc=np.mean(eval_acc))
WriteSDC('log_eval_bert_cls.log', 'epoch: {} eval_acc: {} eval_loss: {}\n'.format(0, np.mean(eval_acc), eval_loss / eval_count))

# %%
