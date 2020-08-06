# %%
import random

# %%
class Preprocess():
    '''
    加载标准问题
    return label_dict, std_dict, std_list
    label_dict: 由label_id获取当前属于该label_id的所有标准问题
    std_dict: 由sta_id获取当前标准问题
    std_list: 标准问题数组
    '''
    @staticmethod
    def load_std(file_name):
        with open(file_name, encoding='utf-8') as f:
            std_ori_list = f.read().split('\n')
        std_ori_list = std_ori_list[:len(std_ori_list) - 1]
        label_dict = {}
        std_dict = {}
        std_list = []
        for line in std_ori_list:
            line = line.strip().split('\t')
            label = line[0].split('__')[2]
            std_id = line[1]
            sentence = line[2]
            if label not in label_dict:
                label_dict[label] = []
            label_dict[label].append([std_id, sentence])
            std_dict[std_id] = [label, std_id, sentence]
            std_list.append([label, std_id, sentence])
        return label_dict, std_dict, std_list
    
    '''
    把数据集拆分为c_train和c_dev
    train_data格式: std_id ext_id sentence
    '''
    @staticmethod
    def split_train(file_name, save_train, save_dev, cut_length=250000):
        with open(file_name, encoding='utf-8') as f:
            ori_train_list = f.read().split('\n')
        ori_train_list = ori_train_list[:len(ori_train_list) - 1]
        random.shuffle(ori_train_list)
        train, dev = ori_train_list[:cut_length], ori_train_list[cut_length:]
        with open('./dataset/computed/{}'.format(save_train), encoding='utf-8', mode='a+') as f:
            for idx, line in enumerate(train):
                if idx != 0:
                    f.write('\n')
                f.write('{}'.format(line))
        with open('./dataset/computed/{}'.format(save_dev), encoding='utf-8', mode='a+') as f:
            for idx, line in enumerate(dev):
                if idx != 0:
                    f.write('\n')
                f.write('{}'.format(line))
    
    '''
    把c_train处理为配对数据集final_train
    '''
    @staticmethod
    def process_sim_data(std_name, file_name, save_name):
        label_dict, std_dict, _ = Preprocess.load_std(std_name)
        result = ''
        with open(file_name, encoding='utf-8') as f:
            train_list = f.read().split('\n')
        for line in train_list:
            std_id, _, sentence = line.strip().split('\t')
            label_id = std_dict[std_id][0]
            result += '{}\t{}\t{}\n'.format(sentence, std_dict[std_id][2], 1)
            sample_arr = random.sample(label_dict[label_id], len(label_dict[label_id]))
            for item in sample_arr:
                if std_id != item[0]:
                    result += '{}\t{}\t{}\n'.format(sentence, item[1], 0)
                    break
        with open(save_name, encoding='utf-8', mode='w+') as f:
            f.write(result)
        return len(label_dict)
    
    '''
    处理验证集为带label的验证集
    保存格式 std_id ext_id label_id sentence
    '''
    @staticmethod
    def process_dev_data_with_label(std_name, dev_name, save_name):
        label_dict, std_dict, _ = Preprocess.load_std(std_name)
        result = ''
        with open(dev_name, encoding='utf-8') as f:
            dev_list = f.read().split('\n')
        for line in dev_list:
            std_id, ext_id, sentence = line.strip().split('\t')
            label_id = std_dict[std_id][0]
            if result != '':
                result += '\n'
            result += '{}\t{}\t{}\t{}'.format(std_id, ext_id, label_id, sentence)
        with open(save_name, encoding='utf-8', mode='w+') as f:
            f.write(result)
        return len(label_dict)
    
    @staticmethod
    def process_cls_data(std_name, file_name, save_name):
        _, std_dict, std_list = Preprocess.load_std(std_name)
        result = ''
        with open(file_name, encoding='utf-8') as f:
            train_list = f.read().split('\n')
        for line in train_list:
            std_id, _, sentence = line.strip().split('\t')
            label_id = std_dict[std_id][0]
            result += '{}\t{}\n'.format(label_id, sentence)
        for item in std_list:
            result += '{}\t{}\n'.format(item[0], item[2])
        with open(save_name, encoding='utf-8', mode='w+') as f:
            f.write(result)
    
    @staticmethod
    def generate_cls_dict(std_name, save_name):
        _, std_dict, std_list = Preprocess.load_std(std_name)
        cls_dict = []
        result = ''
        for item in std_list:
            if item[0] not in cls_dict:
                cls_dict.append(item[0])
        for item in cls_dict:
            result += '{}\n'.format(item)
        with open(save_name, encoding='utf-8', mode='w+') as f:
            f.write(result)

# %%
# Preprocess.generate_cls_dict('./dataset/std_data', './dataset/cls_dict')

# # %%
# Preprocess.split_train('./dataset/computed/cls_data', 'cls_train', 'cls_test', 248000)

# # %%
# Preprocess.process_dev_data_with_label('./dataset/std_data', './dataset/computed/c_dev', './dataset/computed/c_dev_with_label')

# # %%
# label_dict, std_dict, std_list = Preprocess.load_std('./dataset/std_data')
# %%
