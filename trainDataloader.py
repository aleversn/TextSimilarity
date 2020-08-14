# %%
import torch
import random
from preprocess import Preprocess
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
class SimDataset(Dataset):
    def __init__(self, tokenizer, file_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.ori_list = self.load_train(file_name)
        random.shuffle(self.ori_list)
    
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

class BertSimDataset(SimDataset):

    def __getitem__(self, idx):
        sent_1, sent_2, label = self.ori_list[idx].strip().split('\t')
        sentence = '{} [SEP] {}'.format(sent_1, sent_2)
        T = self.tokenizer(sentence, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        
        sentence = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_id = torch.tensor(T['token_type_ids'])
    
        label = torch.tensor(int(label))
        return sentence, attn_mask, token_type_id, label

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

class EvalSimWithLabelDataset(EvalSimDataset):

    def __init__(self, tokenizer, std_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.label_dict, self.std_dict, self.std_list = Preprocess.load_std(std_name)

    def make_data(self, eval_data):
        self.std_id = eval_data[0]
        self.ext_id = eval_data[1]
        self.label_id = eval_data[2]
        self.eval_sentence = eval_data[3]
        self.eval_list = []

        for item in self.label_dict[self.label_id]:
            self.eval_list.append([self.eval_sentence, item[1], item[0]])

class BertEvalSimDataset(EvalSimDataset):

    def __getitem__(self, idx):
        sent_1, sent_2, cur_std_id = self.eval_list[idx]
        sentence = '{} [SEP] {}'.format(sent_1, sent_2)
        T = self.tokenizer(sentence, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        
        sentence = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_id = torch.tensor(T['token_type_ids'])

        return sentence, attn_mask, token_type_id, torch.tensor(int(cur_std_id))

class BertEvalSimWithLabelDataset(BertEvalSimDataset):

    def __init__(self, tokenizer, std_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.label_dict, self.std_dict, self.std_list = Preprocess.load_std(std_name)

    def make_data(self, eval_data):
        self.std_id = eval_data[0]
        self.ext_id = eval_data[1]
        self.label_id = eval_data[2]
        self.eval_sentence = eval_data[3]
        self.eval_list = []

        for item in self.label_dict[self.label_id]:
            self.eval_list.append([self.eval_sentence, item[1], item[0]])

class EnsembleEvalSimWithLabelDataset(BertEvalSimWithLabelDataset):

    def __getitem__(self, idx):
        sent_1, sent_2, cur_std_id = self.eval_list[idx]
        sentence = '{} [SEP] {}'.format(sent_1, sent_2)
        T = self.tokenizer(sentence, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        
        sentence = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        token_type_id = torch.tensor(T['token_type_ids'])

        sent_1 = self.tokenizer(sent_1, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']
        sent_2 = self.tokenizer(sent_2, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']
        sentence_esim = torch.tensor([sent_1, sent_2])

        return sentence, attn_mask, token_type_id, sentence_esim, torch.tensor(int(cur_std_id))

class ClsDataset(Dataset):

    def __init__(self, tokenizer, file_name, cls_vocab, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.data_init(file_name)
        self.cls_vocab_init(cls_vocab)
    
    def cls_vocab_init(self, file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
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

class ClsPredDataset(ClsDataset):

    def __getitem__(self, idx):
        _, ext_id, sent_ = self.ori_list[idx].strip().split('\t')
        T = self.tokenizer(sent_, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        sentence = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        return sentence, attn_mask, sent_, ext_id

class SupremeClsDataset(Dataset):

    def __init__(self, tokenizer, file_name, stdid_vocab, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.data_init(file_name)
        self.cls_vocab_init(stdid_vocab)
    
    def cls_vocab_init(self, file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
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
        std_id, exit_id, label, sentence = self.ori_list[idx].strip().split('\t')
        T = self.tokenizer(sentence, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        sentence = torch.tensor(T['input_ids'])
        attn_mask = torch.tensor(T['attention_mask'])
        std_id = torch.tensor(self.cls_label_2_id[std_id])
        return sentence, attn_mask, std_id, exit_id, label
    
    def __len__ (self):
        return len(self.ori_list)

class JudgerClsDataset(SupremeClsDataset):

    def __init__(self, tokenizer, file_name, std_1, std_2, padding_length=128, one_std=False):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.data_init(file_name, std_1, std_2, one_std)
        self.cls_vocab_init(std_1, std_2)

    def cls_vocab_init(self, std_1, std_2):
        self.cls_label_2_id = {}
        self.cls_id_2_label = {}
        self.cls_label_2_id[std_1] = 0
        self.cls_id_2_label[0] = std_1
        self.cls_label_2_id[std_2] = 1
        self.cls_id_2_label[1] = std_2

    def data_init(self, file_name, std_1, std_2, one_std=False):
        with open(file_name, encoding='utf-8') as f:
            self.ori_list = f.read().split('\n')
        for i in range(len(self.ori_list) - 1, -1, -1):
            std_id, ext_id, label_id, sentence = self.ori_list[i
            ].strip().split('\t')
            if one_std != False:
                if int(std_id) != int(one_std):
                    self.ori_list.pop(i)
            elif int(std_id) != int(std_1) and int(std_id) != int(std_2):
                self.ori_list.pop(i)

class SupremeSimDataset(Dataset):
    def __init__(self, tokenizer, file_name, std_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.data_init(file_name)
        self.label_dict, self.std_dict, self.std_list = Preprocess.load_std(std_name)
    
    def data_init(self, file_name):
        with open(file_name, encoding='utf-8', mode='r') as f:
            self.ori_list = f.read().split('\n')
        self.train_dict = {}
        for line in self.ori_list:
            std_id, ext_id, label_id, sentence = line.strip().split('\t')
            if std_id not in self.train_dict:
                self.train_dict[std_id] = []
            self.train_dict[std_id].append([std_id, ext_id, label_id, sentence])
    
    def make_data(self):
        self.train_list = []
        for line in self.ori_list:
            std_id, ext_id, label_id, sentence = line.strip().split('\t')
            std_answer = self.std_dict[std_id][2]
            self.train_list.append([sentence, std_answer, 1])
            self.train_list.append([sentence, random.sample(self.train_dict[std_id], 1)[0][3], 1])
            for oth in random.sample(self.label_dict[label_id], 5):
                if std_id != oth[0]:
                    self.train_list.append([sentence, oth[1], 0])
                    break
            for oth in random.sample(self.ori_list, 5):
                oth_std_id, _, _, oth_sentence = oth.strip().split('\t')
                if std_id != oth_std_id:
                    self.train_list.append([sentence, oth_sentence, 0])
                    break
    
    def encode(self, a, b, return_type='esim'):
        if return_type == 'esim':
            a = self.tokenizer(a, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']
            b = self.tokenizer(b, add_special_tokens=False, max_length=self.padding_length, padding='max_length', truncation=True)['input_ids']
            return torch.tensor([a, b])
        
        sentence = '{}[SEP]{}'.format(a, b)
        T = self.tokenizer(sentence, add_special_tokens=True, max_length=self.padding_length, padding='max_length', truncation=True)
        sentence = T['input_ids']
        attn_mask = T['attention_mask']
        token_type_id = T['token_type_ids']
        return torch.tensor(sentence), torch.tensor(attn_mask), torch.tensor(token_type_id)
    
    def make_jury(self, target, return_type='esim'):
        train = torch.tensor([])
        max_len = 100 if len(self.train_dict[target[0]]) >= 100 else len(self.train_dict[target[0]])
        for item in random.sample(self.train_dict[target[0]], max_len):
            train_item = self.encode(target[3], item[3], return_type)
            train = torch.cat((train, train_item.unsqueeze(0)))
        return train
    
    def __getitem__(self, idx):
        s1, s2, label = self.train_list[idx]

        sentence = self.encode(s1, s2)
        label = torch.tensor(label)

        return sentence, label
    
    def __len__(self):
        return len(self.train_list)
