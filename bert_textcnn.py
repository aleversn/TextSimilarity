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
from trainDataloader import JudgerClsDataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
# %%
STD_ID_1 = '15'
STD_ID_2 = '16'

tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')

myDataset = JudgerClsDataset(tokenizer, './dataset/101/c_train_with_label', STD_ID_1, STD_ID_2, 100)
dataiter = DataLoader(myDataset, batch_size=120)

myData_eval = JudgerClsDataset(tokenizer, './dataset/101/c_dev_with_label', STD_ID_1, STD_ID_2, 100)
dataiter_eval = DataLoader(myData_eval, batch_size=120)

# %%


class CNNClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_size, kernel_dim=100, kernel_sizes=(2, 3, 4), dropout=0.5):
        super(CNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # print(self.embedding)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])
        '''
        上面是个for循环，不好理解写成下面也是没问题的。
        self.conv13 = nn.Conv2d(Ci, Co, (2, D))
        self.conv14 = nn.Conv2d(Ci, Co, (3, D))
        self.conv15 = nn.Conv2d(Ci, Co, (4, D))
        '''

        # kernal_size = (K,D)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)


#     def init_weights(self, pretrained_word_vectors, is_static=False):
#         self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
#         if is_static:#这里不使用预训练的词向量
#             self.embedding.weight.requires_grad = False


    def forward(self, inputs, is_training=False):
        inputs = inputs.unsqueeze(1)
        # inputs = self.embedding(inputs).unsqueeze(1) # (B,1,T,D)
        # print(inputs.shape)
        inputs = [F.relu(conv(inputs)).squeeze(3)
                  for conv in self.convs]  # [(N,Co,W), ...]*len(Ks)
        # print(inputs[0].shape)

        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                  for i in inputs]  # [(N,Co), ...]*len(Ks)
        '''
        最大池化也可以拆分理解
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        # print(len(inputs))
        concated = torch.cat(inputs, 1)
        # print(concated.shape)
        if is_training:
            concated = self.dropout(concated)  # (N,len(Ks)*Co)
        out = self.fc(concated)
        # print(out.shape)
        return F.log_softmax(out, 1)


# %%
textCNN = CNNClassifier(vocab_size=49807, embedding_dim=768,
                        output_size=len(myDataset.cls_label_2_id))
textCNN.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
textCNN = torch.nn.DataParallel(textCNN, device_ids=[0, 1, 2, 3]).cuda()
textCNN.to(device)

config = BertConfig.from_json_file('./dataset/bert_config.json')
config.output_hidden_states = True
model = BertModel.from_pretrained(
    './model/bert_pre58_4/pytorch_model.bin', config=config)

model.cuda()
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
model.to(device)

save_offset = 12

supreme_config = BertConfig.from_json_file('./dataset/bert_config.json')
supreme_config.num_labels = len(myDataset.cls_label_2_id)
model_ = BertForSequenceClassification(config=supreme_config)

model_.cuda()
model_ = torch.nn.DataParallel(model_, device_ids=[0, 1, 2, 3]).cuda()
model_.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([{'params': model.parameters(), 'lr': 5e-5},
                        {'params': textCNN.parameters(), 'lr': 1e-3}], lr=1e-3, weight_decay=0.)

# %%
losses = []

num_epochs = 30
for epoch in range(num_epochs):
    train_count = 0
    train_loss = 0
    train_acc = []
    train_iter = tqdm(dataiter)
    for sentences, attn_masks, std_ids, _, _ in train_iter:
        model.eval()
        textCNN.train()
        if torch.cuda.is_available():
            sentences = Variable(sentences.cuda())
            attn_masks = Variable(attn_masks.cuda())
            std_ids = Variable(std_ids.cuda())
        else:
            sentences = Variable(sentences)
            attn_masks = Variable(attn_masks)
            std_ids = Variable(std_ids)
        textCNN.zero_grad()

        em = model(input_ids=sentences, attention_mask=attn_masks)[0]
        outputs = textCNN(em, True)
        loss = criterion(outputs.view(-1, outputs.shape[1]), std_ids.view(-1))

        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = outputs.max(-1)[1]

        train_loss += loss.data.item()
        train_count += 1

        train_acc.append((pred == std_ids).sum().data.item() / pred.shape[0])

        train_iter.set_description(
            'Epoch: {}/{} Train'.format(epoch + 1, num_epochs))
        train_iter.set_postfix(train_loss=train_loss /
                               train_count, train_acc=np.mean(train_acc))
    if not os.path.isdir('./model/judger_textcnn/{}x{}'.format(STD_ID_1, STD_ID_2)):
        os.makedirs('./model/judger_textcnn/{}x{}'.format(STD_ID_1, STD_ID_2))
    torch.save(model, "./model/judger_textcnn/{}x{}/textcnn_judger_{}x{}__{}.pth".format(STD_ID_1, STD_ID_2, STD_ID_1, STD_ID_2, epoch + 1 + save_offset))

    with torch.no_grad():
        eval_count = 0
        eval_loss = 0
        eval_acc = []
        eval_iter = tqdm(dataiter_eval)
        for sentences, attn_masks, std_ids, ext_id, label in eval_iter:
            model.eval()
            textCNN.eval()
            model_.eval()
            if torch.cuda.is_available():
                sentences = Variable(sentences.cuda())
                attn_masks = Variable(attn_masks.cuda())
                std_ids = Variable(std_ids.cuda())
            else:
                sentences = Variable(sentences)
                attn_masks = Variable(attn_masks)
                std_ids = Variable(std_ids)

            em = model(input_ids=sentences, attention_mask=attn_masks)[0]
            outputs = textCNN(em)
            loss = criterion(
                outputs.view(-1, outputs.shape[1]), std_ids.view(-1))

            loss = loss.mean()

            outputs_ = model_(input_ids=sentences, attention_mask=attn_masks)
            logits = outputs_[0]

            pred_scores = F.softmax(logits, -1)

            pred = ((outputs + pred_scores) / 2).max(-1)[1]

            eval_loss += loss.data.item()
            eval_count += 1

            eval_acc.append(
                (pred == std_ids).sum().data.item() / pred.shape[0])

            eval_iter.set_description(
                'Epoch: {}/{} Eval'.format(epoch + 1, num_epochs))
            eval_iter.set_postfix(eval_loss=eval_loss /
                                  eval_count, eval_acc=np.mean(eval_acc))
    WriteSDC('judger_textcnn/log_l{}x{}.log'.format(STD_ID_1, STD_ID_2), 'epoch: {} train_acc: {} train_loss: {} eval_acc: {} eval_loss: {}\n'.format(
        epoch + 1 + save_offset, np.mean(train_acc), train_loss / train_count, np.mean(eval_acc), eval_loss / eval_count))

# %%
