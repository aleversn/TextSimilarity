# %%
import os
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
LABEL_ID = '1'

tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')

myDataset = SupremeClsDataset(tokenizer, './dataset/supreme/l{}/s_train'.format(LABEL_ID), './dataset/supreme/l{}/std_dict'.format(LABEL_ID), 100)
dataiter = DataLoader(myDataset, batch_size=128)

myData_eval = SupremeClsDataset(tokenizer, './dataset/supreme/l{}/s_dev'.format(LABEL_ID), './dataset/supreme/l{}/std_dict'.format(LABEL_ID), 100)
dataiter_eval = DataLoader(myData_eval, batch_size=128)

# %%
config = BertConfig.from_json_file('./dataset/bert_config.json')
config.num_labels = len(myDataset.cls_label_2_id)
model = BertForSequenceClassification.from_pretrained('./model/bert_pre58_4/pytorch_model.bin', config=config)

model.cuda()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[2, 3]).cuda()
model.to(device)

save_offset = 900
# model_dict = torch.load("./model/supreme/l{}/bert_supreme_{}.pth".format(LABEL_ID, save_offset)).module.state_dict()
# model.module.load_state_dict(model_dict)

# %%
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.01)

losses = []

num_epochs = 30
for epoch in range(num_epochs):
    train_count = 0
    train_loss = 0
    train_acc = []
    train_iter = tqdm(dataiter)
    for sentences, attn_masks, std_ids, _, _ in train_iter:
        model.train()
        if torch.cuda.is_available():
            sentences = Variable(sentences.cuda(2))
            attn_masks = Variable(attn_masks.cuda(2))
            std_ids = Variable(std_ids.cuda(2))
        else:
            sentences = Variable(sentences)
            attn_masks = Variable(attn_masks)
            std_ids = Variable(std_ids)
        model.zero_grad()

        outputs = model(sentences, attention_mask=attn_masks, labels=std_ids)
        loss, logits = outputs[:2]

        loss = loss.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_scores = F.softmax(logits, -1)
        pred = pred_scores.max(-1)[1]

        train_loss += loss.data.item()
        train_count += 1

        train_acc.append((pred == std_ids).sum().data.item() / pred.shape[0])
        
        train_iter.set_description('Epoch: {}/{} Train'.format(epoch + 1, num_epochs))
        train_iter.set_postfix(train_loss=train_loss / train_count, train_acc=np.mean(train_acc))
    if not os.path.isdir('./model/supreme/l{}'.format(LABEL_ID)):
        os.makedirs('./model/supreme/l{}'.format(LABEL_ID))
    torch.save(model, './model/supreme/l{}/bert_supreme_{}.pth'.format(LABEL_ID, epoch + 1 + save_offset))

    with torch.no_grad():
        eval_count = 0
        eval_loss = 0
        eval_acc = []
        eval_iter = tqdm(dataiter_eval)
        for sentences, attn_masks, std_ids, ext_id, label in eval_iter:
            model.eval()
            if torch.cuda.is_available():
                sentences = Variable(sentences.cuda(2))
                attn_masks = Variable(attn_masks.cuda(2))
                std_ids = Variable(std_ids.cuda(2))
            else:
                sentences = Variable(sentences)
                attn_masks = Variable(attn_masks)
                std_ids = Variable(std_ids)

            outputs = model(sentences, attention_mask=attn_masks, labels=std_ids)
            loss, logits = outputs[:2]

            loss = loss.mean()

            pred_scores = F.softmax(logits, -1)
            pred = pred_scores.max(-1)[1]

            eval_loss += loss.data.item()
            eval_count += 1

            eval_acc.append((pred == std_ids).sum().data.item() / pred.shape[0])
            
            eval_iter.set_description('Epoch: {}/{} Eval'.format(epoch + 1, num_epochs))
            eval_iter.set_postfix(eval_loss=eval_loss / eval_count, eval_acc=np.mean(eval_acc))
    WriteSDC('supreme/log_l{}.log'.format(LABEL_ID), 'epoch: {} train_acc: {} train_loss: {} eval_acc: {} eval_loss: {}\n'.format(epoch + 1 + save_offset, np.mean(train_acc), train_loss / train_count, np.mean(eval_acc), eval_loss / eval_count))

# %%
