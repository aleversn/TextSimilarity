# %%
import torch
import datetime
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset

current_time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# %%
def WriteSDC(name, info):
    with open("./log/{}.txt".format(name), mode="a+", encoding="utf-8") as f:
        f.write(info)

def load_sim_dev(filename):
    with open(filename, encoding='utf-8') as f:
        dev_list = f.read().split('\n')
    result = []
    for line in dev_list:
        line = line.strip().split('\t')
        result.append(line)
    return result

def max_length_of_train(filename):
    count = []
    max = 0
    with open(filename, encoding='utf-8') as f:
        ori_list = f.read().split('\n')
    ori_list = ori_list[:len(ori_list) - 1]
    for item in ori_list:
        item = item.strip().split('\t')
        if max < len(item[0]):
            max = len(item[0])
        if max < len(item[1]):
            max = len(item[1])
        count.append(len(item[0]))
        count.append(len(item[1]))
    return max, np.mean(count), np.median(count), np.argmax(np.bincount(count))

def eval(esim, eval_list, myData_eval, epoch, save_offset):
    with torch.no_grad():
        esim.eval()
        eval_correct_num = 0
        eval_list_iter = tqdm(eval_list)
        for idx, item in enumerate(eval_list_iter):
            cur_eval_result_scores = torch.tensor([])
            cur_eval_result_stdid = torch.LongTensor([])
            myData_eval.make_data(item)
            dataiter_eval = DataLoader(myData_eval, batch_size=1000)
            for sentences, cur_std_id in dataiter_eval:
                if torch.cuda.is_available():
                    sentences = Variable(sentences.cuda())
                    cur_std_id = Variable(cur_std_id.cuda())
                    cur_eval_result_scores = Variable(cur_eval_result_scores.cuda())
                    cur_eval_result_stdid = Variable(cur_eval_result_stdid.cuda())
                else:
                    sentences = Variable(sentences)
                    cur_std_id = Variable(cur_std_id)
                out = esim(sentences)
                pred_scores = out[:,1]
                cur_eval_result_scores = torch.cat((cur_eval_result_scores, pred_scores))
                cur_eval_result_stdid = torch.cat((cur_eval_result_stdid,cur_std_id))
            eval_list_iter.set_description('{}/{}'.format(idx + 1, len(eval_list)))
            eval_list_iter.set_postfix(correct_num=eval_correct_num, eval_acc=eval_correct_num / (idx + 1))
            max_item_index = cur_eval_result_scores.sort(descending=True)[1][0].data.item()
            max_item_id = cur_std_id[max_item_index].data.item()
            if max_item_id == int(item[0]):
                eval_correct_num += 1
        print('Eval_acc: {}\n'.format(eval_correct_num / len(eval_list)))
        WriteSDC('log_eval_20200803.log', 'epoch: {} eval_num: {} eval_acc: {}\n'.format(epoch + 1 + save_offset, eval_correct_num, eval_correct_num / len(eval_list)))

def pred(test_file_name, std_name, esim, eval_list, myData_eval):
    result_std_id = []
    with torch.no_grad():
        esim.eval()
        eval_correct_num = 0
        eval_list_iter = tqdm(eval_list)
        for idx, item in enumerate(eval_list_iter):
            cur_eval_result_scores = torch.tensor([])
            cur_eval_result_stdid = torch.LongTensor([])
            myData_eval.make_data(item)
            dataiter_eval = DataLoader(myData_eval, batch_size=1000)
            for sentences, cur_std_id in dataiter_eval:
                if torch.cuda.is_available():
                    sentences = Variable(sentences.cuda())
                    cur_std_id = Variable(cur_std_id.cuda())
                    cur_eval_result_scores = Variable(cur_eval_result_scores.cuda())
                    cur_eval_result_stdid = Variable(cur_eval_result_stdid.cuda())
                else:
                    sentences = Variable(sentences)
                    cur_std_id = Variable(cur_std_id)
                out = esim(sentences)
                pred_scores = out[:,1]
                cur_eval_result_scores = torch.cat((cur_eval_result_scores, pred_scores))
                cur_eval_result_stdid = torch.cat((cur_eval_result_stdid,cur_std_id))
            eval_list_iter.set_description('{}/{}'.format(idx + 1, len(eval_list)))
            eval_list_iter.set_postfix(correct_num=eval_correct_num, eval_acc=eval_correct_num / (idx + 1))
            max_item_index = cur_eval_result_scores.sort(descending=True)[1][0].data.item()
            max_item_id = cur_std_id[max_item_index].data.item()
            result_std_id.append(max_item_id)
    with open('result.csv', encoding='utf-8', mode='a+') as f:
        f.write('ext_id,std_id')
        for idx, item in enumerate(eval_list):
            f.write('\n{},{}'.format(item[1], result_std_id[idx]))
            


# %%
max_length_of_train('./dataset/computed/final_train')

# %%
