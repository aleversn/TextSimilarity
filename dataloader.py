# %%
import torch
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import BertModel, BertConfig, BertTokenizer, BertForSequenceClassification

# %%
# tokenizer = BertTokenizer.from_pretrained('./dataset/vocab')
# word_idx = tokenizer("205 19 290 289 58 54", add_special_tokens=True, max_length=20, padding="max_length", truncation=True)['input_ids']
# print(word_idx)

# %%
class PreTrainDataset(Dataset):
    def __init__(self, tokenizer, file_name, padding_length=128):
        self.tokenizer = tokenizer
        self.padding_length = padding_length
        self.max_masked_num = int(padding_length * 0.1)
        self.ori_list = self.load_pre_train(file_name)
        self.masked_idx = self.tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]
    
    def __getitem__(self, idx):
        T = self.tokenizer(self.ori_list[idx], add_special_tokens=True, max_length=self.padding_length, padding="max_length", truncation=True)
        sentence = T['input_ids']
        attn_mask = T['attention_mask']

        index_arr = [i for i in range(len(sentence))]
        index_arr = index_arr[1:]
        random.shuffle(index_arr)
        index_arr = index_arr[:int(len(index_arr) * 0.15)]
        masked_arr = index_arr[:int(len(index_arr) * 0.8)]
        err_arr = index_arr[int(len(index_arr) * 0.8):int(len(index_arr) * 0.9)]

        tags = torch.tensor(sentence)

        for idx in masked_arr:
            sentence[idx] = self.masked_idx
            tags[idx] = -100
        for idx in err_arr:
            sentence[idx] = int(random.random() * 49800)
        attn_mask = torch.tensor(attn_mask)
        sentence = torch.tensor(sentence)

        return sentence, attn_mask, tags
    
    def __len__ (self):
        return len(self.ori_list)

    @staticmethod
    def load_pre_train(file_name):
        with open(file_name, encoding='utf-8') as f:
            ori_list = f.read().split('\n')
        # for i in reversed(range(len(ori_list))):
        #     if len(ori_list[i]) < 1:
        #         ori_list.pop(i)
        length = len(ori_list)
        if len(ori_list[length - 1]) < 1:
            ori_list.pop(length - 1)
        return ori_list

# d = PreTrainDataset(tokenizer, './dataset/pre_train_data')
# print(d.__getitem__(0))

# %%
