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
from trainDataloader import ClsDataset, ClsPredDataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')

myDataset = ClsPredDataset(tokenizer, './dataset/test_data', './dataset/cls_dict', 100)
dataiter = DataLoader(myDataset, batch_size=512)

# %%
config = BertConfig.from_json_file('./dataset/bert_config.json')
config.num_labels = len(myDataset.cls_label_2_id)
model = BertForSequenceClassification.from_pretrained('./model/bert_pre58_3/pytorch_model.bin', config=config)

model.cuda()
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[2, 3]).cuda()
model.to(device)

model_dict = torch.load("./model/bert_cls/bert_58_cls8.pth").module.state_dict()
model.module.load_state_dict(model_dict)

# %%
with torch.no_grad():
    eval_iter = tqdm(dataiter)
    for sentences, attn_masks, sents_, ext_ids in eval_iter:
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
        for idx, item in enumerate(sents_):
            result += '{}\t{}\t{}\t{}\n'.format(0, ext_ids[idx], myDataset.cls_id_2_label[pred[idx].data.item()], item)
        with open('./dataset/computed/test_with_label', encoding='utf-8', mode='a+') as f:
            f.write(result)
            

# %%
