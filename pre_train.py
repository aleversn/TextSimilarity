# %%
import os
import sys
import random
from tqdm import tqdm
from utils import *
from dataloader import PreTrainDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertForMaskedLM, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')
word_idx = tokenizer("205 19 290 289 58 54", add_special_tokens=True, max_length=20, padding="max_length")
print(word_idx)

class RandomPreTrainDataset(PreTrainDataset):
    def __init__(self, tokenizer, file_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.max_masked_num = int(padding_length * 0.1)
        self.whole_ori_list = self.load_pre_train(file_name)
        self.ori_list = random.sample(self.whole_ori_list, int(len(self.whole_ori_list) / 5))
        self.masked_idx = self.tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]

    def sample(self):
        self.ori_list = random.sample(self.whole_ori_list, int(len(self.whole_ori_list) / 5))

# %%
config = BertConfig.from_json_file('./dataset/bert_config.json')
model = BertForMaskedLM.from_pretrained('./model/bert_pre58_3/pytorch_model.bin', config=config)

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# model = torch.nn.DataParallel(model, device_ids=[2, 3]).cuda()
# model.to(device)
# model_dict = torch.load("./model/bert_58_pretrain2.pth").module.state_dict()
# model.module.load_state_dict(model_dict)

# %%
model.cuda()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[2, 3])
    model.to(device)

# %%
myData = RandomPreTrainDataset(tokenizer, './dataset/pre_train_data', 50)
dataiter = DataLoader(myData, batch_size=120)

# %%
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)

losses = []

save_offset = 0
num_epochs = 80
for epoch in range(num_epochs):
    train_count = 0
    train_loss = 0
    train_iter = tqdm(dataiter)
    for sentence, attn_masks, tags in train_iter:
        model.train()
        if torch.cuda.is_available():
            sentence = Variable(sentence.cuda())
            attn_masks = Variable(attn_masks.cuda())
            tags = Variable(tags.cuda())
        else:
            sentence = Variable(sentence)
            attn_masks = Variable(attn_masks)
            tags = Variable(tags)
        model.zero_grad()

        outputs = model(sentence, attention_mask=attn_masks, labels=tags)
        loss, prediction_scores = outputs[:2]

        loss = loss.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
        train_count += 1
        
        train_iter.set_description('Epoch: {}/{} Train'.format(epoch + 1, num_epochs))
        train_iter.set_postfix(train_loss=train_loss / train_count, cur_pred_scores=prediction_scores.mean().data.item())
    os.makedirs('./model/bert_pre58_{}'.format(epoch + 1 + save_offset))
    model.module.save_pretrained('./model/bert_pre58_{}'.format(epoch + 1 + save_offset))
    WriteSDC('log_20200803.log', 'epoch: {} loss: {}\n'.format(epoch + 1 + save_offset, train_loss / train_count))
    torch.cuda.empty_cache()
    myData.sample()

# %%
