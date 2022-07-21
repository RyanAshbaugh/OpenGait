import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json


class DataSetBRIAR(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
        seqs_info: the list with each element indicating
        a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg)
        self.training = training
        self.cache = data_cfg['cache']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError('Each input data({}) should have the same '
                                 'length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError('Each input data({}) should have at least '
                                 'one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config):
        self.dataset_root = data_config['dataset_root']
        try:
            self.data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            self.data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(self.dataset_root)
        self.train_set = [label for label in train_set if label in label_list]
        self.test_set = [label for label in test_set if label in label_list]

    def get_seqs_info_list(self, label_set):
        seqs_info_list = []
        for label in label_set:
            for sequence_type in sorted(os.listdir(osp.join(self.dataset_root,
                                                            label))):
                for sequence_view in sorted(os.listdir(osp.join(self.dataset_root,
                                                                label,
                                                                sequence_type))):
                    seq_info = [label, sequence_type, sequence_view]
                    seq_path = osp.join(self.dataset_root, *seq_info)
                    seq_dirs = sorted(os.listdir(seq_path))
                    if seq_dirs != []:
                        seq_dirs = [osp.join(seq_path, data_dir)
                                    for data_dir in seq_dirs]
                        if self.data_in_use is not None:
                            seq_dirs = [data_dir for data_dir, use_bl in
                                        zip(seq_dirs,
                                            self.data_in_use) if use_bl]
                        seqs_info_list.append([*seq_info, seq_dirs])
        return seqs_info_list

    def __assemble_sequence_data_and_info(self):
        self.seqs_info = self.get_seqs_info_list(self.train_set) \
            if self.training else self.get_seqs_info_list(self.test_set)

        # loop over data files to begin breaking into smaller sequences
        full_sequence_files = [seq_info[-1][0] for seq_info in self.seqs_info]
        for full_sequence_file in full_sequence_files:
            print(full_sequence_file)
