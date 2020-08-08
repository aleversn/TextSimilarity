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

eval_list = load_sim_dev('./dataset/computed/test_with_label')
myData_eval = BertEvalSimWithLabelDataset(tokenizer, './dataset/std_data', 200)

# %%
config = BertConfig.from_json_file('./dataset/bert_config.json')
config.num_labels = 2
model = BertForSequenceClassification.from_pretrained('./model/bert_pre58_3/pytorch_model.bin', config=config)

model.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
model.to(device)

EVAL_EPOCH = 928
model_dict = torch.load("./model/bert_sim/bert_sim_{}.pth".format(EVAL_EPOCH)).module.state_dict()
model.module.load_state_dict(model_dict)

# %%
pred_bert(model, eval_list, myData_eval)

# %%
