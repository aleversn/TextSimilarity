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
from trainDataloader import SimDataset, EvalSimDataset, EvalSimWithLabelDataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
class BiSimDataset(SimDataset):
    def __getitem__(self, idx):
        reversed = idx % 2 == 0
        if idx % 2 == 0:
            idx = int(idx / 2)
        else:
            idx = int((idx - 1) / 2)
        sent_1, sent_2, label = self.ori_list[idx].strip().split('\t')
        sent_1 = self.tokenizer(sent_1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']
        sent_2 = self.tokenizer(sent_2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']
        
        if reversed:
            sentence = torch.tensor([sent_2, sent_1])
        else:
            sentence = torch.tensor([sent_1, sent_2])
        label = torch.tensor(float(label))

        return sentence, label
    
    def __len__ (self):
        return 2 * len(self.ori_list)

tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')

myDataset = SimDataset(tokenizer, './dataset/101/final_train_x', 100)
dataiter = DataLoader(myDataset, batch_size=1024)

eval_list = load_sim_dev('./dataset/101/c_dev_with_label')
myData_eval = EvalSimWithLabelDataset(tokenizer, './dataset/std_data', 80)

# %%
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        
#         outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        outputs = (encoder_outputs * weights)
        
#         return outputs, weights
        return outputs

class ESIM(nn.Module):
    def __init__(self):
        super(ESIM, self).__init__()
#         self.args = args
        self.num_word = 49807
        self.dropout = 0.5
        self.hidden_size = 300
        self.embeds_dim = 300
        self.linear_size = 200
        
        self.embeds = nn.Embedding(self.num_word, self.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        
        self.lstm2 = nn.LSTM(self.hidden_size*8, self.hidden_size, batch_first=True, bidirectional=True)
        
        self.attention = SelfAttention(self.hidden_size*2)
        
        self.fc = nn.Sequential (
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear_size, 2),
            nn.Softmax(dim=-1)
        )
    
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, input):
        # batch_size * seq_len
        sent1, sent2 = input[:, 0, :], input[:, 1, :]
        mask1, mask2 = sent1.eq(0), sent2.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)
        
        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)
            
        q1_compose = self.attention(q1_compose)
        q2_compose = self.attention(q2_compose)
        
        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        
        similarity = self.fc(x)
        
        return similarity

# %%
esim = ESIM()
if torch.cuda.is_available():
    esim.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    esim = nn.DataParallel(esim, device_ids=[0, 1, 2, 3])
    esim.to(device)
model_dict = torch.load("./model/esim/esim_sim_1019.pth").module.state_dict()
esim.module.load_state_dict(model_dict)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(esim.parameters(), lr=1e-3)

save_offset = 1019
num_epochs = 120
for epoch in range(num_epochs):
    
    train_loss = 0
    train_acc = 0
    train_count = 0
    esim.train()
    train_iter = tqdm(dataiter)
    for sentences, label in train_iter:
        if torch.cuda.is_available():
            sentences = Variable(sentences.cuda())
            label = Variable(label.cuda())
        else:
            sentences = Variable(sentences)
            label = Variable(label)
        # forward
        out = esim(sentences)
        loss = criterion(out[:, 1], label.float())
        loss = loss.mean()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.data.item()
        # 计算分类的准确率
        pred_scores = out[:,1]
        _, pred = out.max(1)
        train_acc += (pred == label).sum().data.item() / sentences.shape[0]
        
        train_count += 1
        train_iter.set_description('Epoch: {}/{} Train'.format(epoch + 1, num_epochs))
        train_iter.set_postfix(train_loss=train_loss / train_count, train_acc=train_acc / train_count)
    torch.save(esim, './model/esim/esim_sim_{}.pth'.format(epoch + 1 + save_offset))
    WriteSDC('log_esim.log', 'epoch: {} train_acc: {} loss: {}\n'.format(epoch + 1 + save_offset, train_acc / train_count, train_loss / train_count))
    
    if epoch == 0 or epoch % 3 != 0:
        continue
    
    eval(esim, eval_list, myData_eval, epoch, save_offset)

# %%
