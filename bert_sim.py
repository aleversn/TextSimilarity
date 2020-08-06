# %%
import sys
import random
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
from utils import *
from tqdm import tqdm
from trainDataloader import BertSimDataset, BertEvalSimDataset, BertEvalSimWithLabelDataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')

myDataset = BertSimDataset(tokenizer, './dataset/computed/final_train', 100)
dataiter = DataLoader(myDataset, batch_size=200)

eval_list = load_sim_dev('./dataset/computed/c_dev_with_label')
myData_eval = BertEvalSimWithLabelDataset(tokenizer, './dataset/std_data', 100)

# %%
config = BertConfig.from_json_file('./dataset/bert_config.json')
config.num_labels = 2
model = BertForSequenceClassification.from_pretrained('./model/bert_pre58_3/pytorch_model.bin', config=config)

model.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
model.to(device)

model_dict = torch.load("./model/bert_sim/bert_sim_2.pth").module.state_dict()
model.module.load_state_dict(model_dict)

# %%
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)

save_offset = 2
num_epochs = 120
for epoch in range(num_epochs):
    
    train_loss = 0
    train_acc = 0
    train_count = 0
    model.train()
    train_iter = tqdm(dataiter)
    for sentences, attn_masks, types, labels in train_iter:
        if torch.cuda.is_available():
            sentences = Variable(sentences.cuda())
            attn_masks = Variable(attn_masks.cuda())
            types = Variable(types.cuda())
            labels = Variable(labels.cuda())
        else:
            sentences = Variable(sentences)
            attn_masks = Variable(attn_masks)
            types = Variable(types)
            labels = Variable(labels)
        
        outputs = model(sentences, attention_mask=attn_masks, token_type_ids=types, labels=labels)
        loss, logits = outputs[:2]

        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.data.item()
        # 计算分类的准确率
        pred_scores = F.softmax(logits, -1)
        pred = pred_scores.max(-1)[1]
        train_acc += (pred == labels).sum().data.item() / sentences.shape[0]
        
        train_count += 1
        train_iter.set_description('Epoch: {}/{} Train'.format(epoch + 1, num_epochs))
        train_iter.set_postfix(train_loss=train_loss / train_count, train_acc=train_acc / train_count)
    torch.save(model, './model/bert_sim/bert_sim_{}.pth'.format(epoch + 1 + save_offset))
    WriteSDC('log_bert_sim.log', 'epoch: {} train_acc: {} loss: {}\n'.format(epoch + 1 + save_offset, train_acc / train_count, train_loss / train_count))
    
    if epoch == 0 or epoch % 9 != 0:
        continue
    
    eval_bert(model, eval_list, myData_eval, epoch, save_offset, 'log_bert_sim_eval.log')

# %%
