import torch
import torch.utils.data as pt_data
from torch.utils.data import IterableDataset, DataLoader, ConcatDataset

import os
import numpy as np
import pickle
import random
import math
from collections import defaultdict
import glob

class Dataset(pt_data.Dataset):
    def __init__(self):
        self.rsm = None
        self.frag_info = None
        self.feat = None
        self.label = None
        self.file = None
        self.precursor_id = None

    def __getitem__(self, idx):
        return_dict = {"rsm": self.rsm[idx],
                       "frag_info": self.frag_info[idx],
                       "feat": self.feat[idx],
                       "label": self.label[idx],
                       "file": self.file[idx],
                       "precursor_id": self.precursor_id[idx]}
        return return_dict

    def __len__(self):
        return len(self.precursor_id)

    def fit_scale(self):
        pass

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

def collate_batch(batch_data):
    """Collate batch of samples."""
    one_batch_rsm = torch.tensor(np.array([batch["rsm"] for batch in batch_data]), dtype=torch.float)
    one_batch_frag_info = torch.tensor(np.array([batch["frag_info"] for batch in batch_data]), dtype=torch.float)
    one_batch_feat = torch.tensor(np.array([batch["feat"] for batch in batch_data]), dtype=torch.float)
    one_batch_label = torch.tensor(np.array([batch["label"] for batch in batch_data]), dtype=torch.float)

    one_batch_rsm = torch.nan_to_num(one_batch_rsm)
    one_batch_frag_info = torch.nan_to_num(one_batch_frag_info)
    one_batch_feat = torch.nan_to_num(one_batch_feat)
    one_batch_label = torch.nan_to_num(one_batch_label)

    one_batch_file_name = [batch["file"] for batch in batch_data]
    one_batch_precursor_id = [batch["precursor_id"] for batch in batch_data]
    
    return one_batch_rsm, one_batch_frag_info, one_batch_feat, one_batch_label, one_batch_file_name, one_batch_precursor_id


def shuffle_file_list(file_list, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(file_list), generator=generator).numpy()
    file_list = (np.array(file_list)[idx]).tolist()
    return file_list


# https://blog.csdn.net/zhang19990111/article/details/131636456
def create_iterable_dataset(data_path,
                            logging,
                            config,
                            parse='train',
                            multi_node=False):
    """
    Note: If you want to load all data in the memory, please set "read_part" to False.
    Args:
        :param data_path: A string. dataset's path.
        :param logging: out logging.
        :param config: data from the yaml file.
        :param parse: train or val
        :param multi_node: is multi_nod or not. 
    :return:
    """    
    if parse == 'train':
        # 训练阶段
        total_train_path = data_path.split(';')
    
        train_file_list = []
        for train_path in total_train_path:
            train_part_file_list = glob.glob(f'{train_path}/*.pkl')
            
            # logging.info(f"******************{train_path} origin  loaded: {len(train_part_file_list)};**********")
            train_part_file_list_clean = []
            for file_path in train_part_file_list:
                train_part_file_list_clean.append(file_path)
            
            # logging.info(f"******************{train_path} filter loaded: {len(train_part_file_list_clean)};**********")
            if len(train_part_file_list_clean) > 0:
                train_file_list.extend(train_part_file_list_clean)

        train_file_list = [f for f in train_file_list if not is_file_empty(f)]

        random.shuffle(train_file_list)
        train_file_list = shuffle_file_list(train_file_list, config['seed'])
        
        logging.info(f"******************train loaded: {len(train_file_list)};**********")
        
        # update gpu_num
        if multi_node:
            gpu_num = int(config['gpu_num'])
        else:
            gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # 按照3*gpu_num方式，对数据集截断
        file_bin_num = len(train_file_list) // (3 * gpu_num)
        file_truncation_num = file_bin_num * (3 * gpu_num)
        train_file_list = train_file_list[:file_truncation_num]
        
        logging.info(f"******************after truncation,  train loaded: {len(train_file_list)};**********")

        # 文件列表分桶
        file_bin_dict = defaultdict(list)

        # 每3个pkl文件，放在一个桶内
        for i in range(len(train_file_list)):
            file_bin_dict[i // 3].append(train_file_list[i])
        file_bin_list = list(file_bin_dict.keys())

        train_dl = IterableDiartDataset(file_bin_list,
                                        file_bin_dict=file_bin_dict,
                                        batch_size=config["train_batch_size"],
                                        buffer_size=config["buffer_size"], #len(file_bin_list), # 蓄水池深度
                                        gpu_num=gpu_num,
                                        shuffle=True,
                                        seed=config['seed'],
                                        multi_node=multi_node)
        logging.info(
            f"Data loaded: {len(train_dl) * config['train_batch_size']:,} training samples, batch_size: {config['train_batch_size']}, multi_node: {multi_node}"
        )
        return train_dl
        
    elif parse == 'val':
        # 验证阶段
        if ';' in data_path:
            total_val_path = data_path.split(';')
    
            val_file_list = []
            for val_path in total_val_path:
                val_part_file_list = glob.glob(f'{val_path}/*.pkl')

                # logging.info(f"******************{train_path} origin  loaded: {len(train_part_file_list)};**********")
                val_part_file_list_clean = []
                for file_path in val_part_file_list:
                    val_part_file_list_clean.append(file_path)

                # logging.info(f"******************{train_path} filter loaded: {len(train_part_file_list_clean)};**********")
                if len(val_part_file_list_clean) > 0:
                    val_file_list.extend(val_part_file_list_clean)
        else:
            val_file_list = glob.glob(f'{data_path}/*.pkl')
        valid_file_list = [f for f in val_file_list if not is_file_empty(f)]

        # update gpu_num
        if multi_node:
            gpu_num = int(config['gpu_num'])
        else:
            gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1

        # 文件列表分桶
        file_bin_dict = defaultdict(list)

        # 每个pkl文件，放在一个桶内
        for i in range(len(valid_file_list)):
            file_bin_dict[i // 1].append(valid_file_list[i])
        file_bin_list = list(file_bin_dict.keys())

        val_dl = IterableDiartDataset(file_bin_list,
                                      file_bin_dict=file_bin_dict,
                                      batch_size=config["predict_batch_size"],
                                      gpu_num=gpu_num,
                                      shuffle=False,
                                      multi_node=multi_node)

        logging.info(
            f"{len(val_dl) * config['predict_batch_size']:,} validation samples, batch_size: {config['predict_batch_size']}, multi_node: {multi_node}"
        )
        return val_dl

    else:
        # eval阶段
        valid_file_list = [f for f in data_path if not is_file_empty(f)]
        gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1

        # 文件列表分桶
        file_bin_dict = defaultdict(list)

        # 每个pkl文件，放在一个桶内
        for i in range(len(valid_file_list)):
            file_bin_dict[i // 1].append(valid_file_list[i])
        file_bin_list = list(file_bin_dict.keys())

        val_dl = IterableDiartDataset(file_bin_list,
                                      file_bin_dict=file_bin_dict,
                                      batch_size=config["predict_batch_size"],
                                      gpu_num=gpu_num,
                                      shuffle=False,
                                      multi_node=multi_node)

        logging.info(
            f"{len(val_dl) * config['predict_batch_size']:,} validation samples, batch_size: {config['predict_batch_size']}, multi_node: {multi_node}"
        )
        return val_dl


class IterableDiartDataset(IterableDataset):
    """
    Custom dataset class for dataset in order to use efficient
    dataloader tool provided by PyTorch.
    """

    def __init__(self,
                 file_list: list,
                 file_bin_dict=None,
                 batch_size=1024,
                 bath_file_size=1,
                 buffer_size=2,
                 epoch=0,
                 gpu_num=1,
                 shuffle=True,
                 seed=0,
                 multi_node=False):
        super(IterableDiartDataset).__init__()
        # 文件列表
        self.epoch = epoch
        self.file_list = file_list
        self.file_bin_dict = file_bin_dict
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.seed = seed

        self.gpu_num = gpu_num
        self.multi_node = multi_node
        
        # 单次抽样的文件大小
        self.bath_file_size = bath_file_size
        self.buffer_size = buffer_size
        
    def parse_file(self, file_name):
        if self.file_bin_dict is not None:
            data = []
            for bin_file in file_name:
                try:
                    f = open(bin_file, "rb")
                    data.append(pickle.loads(f.read()))
                    f.close()
                except:
                    print(f'load {bin_file} error!!')
                    continue
            data = ConcatDataset(data)
        else:
            f = open(file_name, "rb")
            data = pickle.loads(f.read())
            f.close()
        # print('parse_file: ', file_name, flush=True)
        return DataLoader(data,
                          shuffle=False,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          num_workers=0,
                          collate_fn=collate_batch)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def file_mapper(self, file_list):
        idx = 0
        file_num = len(file_list)
        while idx < file_num:
            if self.file_bin_dict is not None:
                yield self.parse_file(self.file_bin_dict[file_list[idx]])
            else:
                yield self.parse_file(file_list[idx])
            idx += 1

    def __iter__(self):
        if self.gpu_num > 1:
            if self.multi_node:# 多机多卡
                if 'RANK' in os.environ:
                    rank = int(os.environ['RANK'])
                else:
                    rank = 0
                file_itr = self.file_list[rank::self.gpu_num]

            else:# 单机多卡
                if 'LOCAL_RANK' in os.environ:
                    local_rank = int(os.environ['LOCAL_RANK'])
                else:
                    local_rank = 0
                
                file_itr = self.file_list[local_rank::self.gpu_num]
        else:
            # 单卡
            file_itr = self.file_list

        file_mapped_itr = self.file_mapper(file_itr)

        if self.shuffle:
            return self._shuffle(file_mapped_itr)
        else:
            return file_mapped_itr

    def __len__(self):
        if self.gpu_num > 1:
            return math.ceil(len(self.file_list) / self.gpu_num)
        else:
            return len(self.file_list)

    def generate_random_num(self):
        while True:
            random_nums = random.sample(range(self.buffer_size), self.bath_file_size)
            yield from random_nums

    # 蓄水池抽样（每个数据都是以m/N的概率获得的）
    def _shuffle(self, mapped_itr):
        buffer = []
        for dt in mapped_itr:
            # 如果接收的数据量小于m，则依次放入蓄水池。
            if len(buffer) < self.buffer_size:
                buffer.append(dt)
            # 当接收到第i个数据时，i >= m，在[0, i]范围内取以随机数d。
            # 若d的落在[0, m-1]范围内，则用接收到的第i个数据替换蓄水池中的第d个数据。
            else:
                i = next(self.generate_random_num())
                yield buffer[i]
                buffer[i] = dt
        random.shuffle(buffer)
        yield from buffer
