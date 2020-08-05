# %%
import torch
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
class SimDataset(Dataset):
    def __init__(self, tokenizer, file_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(file_name)
    
    def __getitem__(self, idx):
        sent_1, sent_2, label = self.ori_list[idx].strip().split('\t')
        sent_1 = self.tokenizer(sent_1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']
        sent_2 = self.tokenizer(sent_2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']

        sentence = torch.tensor([sent_1, sent_2])
        label = torch.tensor(float(label))

        return sentence, label
    
    def __len__ (self):
        return len(self.ori_list)

    @staticmethod
    def load_train(file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        ori_list = ori_list[:len(ori_list) - 1]
        return ori_list

class EvalSimDataset(Dataset):
    def __init__(self, tokenizer, std_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        std_ori_list = self.load_data(std_name)
        self.std_list = []
        for line in std_ori_list:
            line = line.strip().split('\t')
            label = line[0].split('__')[2]
            id = line[1]
            sentence = line[2]
            self.std_list.append([label, id, sentence])
    
    def make_data(self, eval_data):
        self.std_id = eval_data[0]
        self.eval_sentence = eval_data[2]
        self.eval_list = []
        for item in self.std_list:
            self.eval_list.append([self.eval_sentence, item[2], item[1]])
    
    def __getitem__(self, idx):
        sent_1, sent_2, cur_std_id = self.eval_list[idx]
        sent_1 = self.tokenizer(sent_1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']
        sent_2 = self.tokenizer(sent_2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']

        sentence = torch.tensor([sent_1, sent_2])

        return sentence, torch.tensor(int(cur_std_id))
    
    def __len__ (self):
        return len(self.eval_list)
    
    @staticmethod
    def load_data(file_name):
        with open(file_name, encoding='utf-8') as f:
            data_list = f.read().split('\n')
        data_list = data_list[:len(data_list) - 1]
        return data_list

class ClsDataset(Dataset):

    def __init__(self, tokenizer, file_name, cls_vocab, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.data_init(file_name)
        self.cls_vocab_init(cls_vocab)
    
    def cls_vocab_init(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        self.cls_label_2_id = {}
        self.cls_id_2_label = {}
        for idx, item in enumerate(ori_list):
            item = item.strip()
            self.cls_label_2_id[item] = idx
            self.cls_id_2_label[idx] = item
    
    def data_init(self, file_name):
        with open(file_name, encoding='utf-8') as f:
            self.ori_list = f.read().split('\n')
    
    def __getitem__(self, idx):
        label, sentence = self.ori_list[idx].strip().split('\t')
        T = self.tokenizer(sentence, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        sentence = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        label = torch.tensor(self.cls_label_2_id[label])
        return sentence, attn_mask, label
    
    def __len__ (self):
        return len(self.ori_list)
        