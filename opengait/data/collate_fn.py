import math
import random
import numpy as np
from utils import get_msg_mgr


class CollateFn(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        sample_type = sample_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        self.ordered = sample_type[1]
        self.use_full_video = sample_config['use_full_video']
        self.random_sample_full_video = sample_config['random_sample_full_video']
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError
        if self.ordered not in ['ordered', 'unordered']:
            raise ValueError
        self.ordered = sample_type[1] == 'ordered'

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed']

        # unfixed cases
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max']
            self.frames_num_min = sample_config['frames_num_min']

        if self.sampler != 'all' and self.ordered:
            self.frames_skip_num = sample_config['frames_skip_num']

        self.frames_all_limit = -1
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']

        self.frames_num_fixed = sample_config['frames_num_fixed']

    def __call__(self, batch):
        batch_size = len(batch)
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        feature_num = len(batch[0][0])
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        for bt in batch:
            seqs_batch.append(bt[0])
            labs_batch.append(self.label_set.index(bt[1][0]))
            typs_batch.append(bt[1][1])
            vies_batch.append(bt[1][2])

        global count
        count = 0

        def sample_frames(seqs):
            global count
            sampled_fras = [[] for i in range(feature_num)]
            seq_len = len(seqs[0])
            indices = list(range(seq_len))

            if self.sampler in ['fixed', 'unfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))

                if self.ordered:
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                else:
                    replace = seq_len < frames_num

                    if seq_len == 0:
                        get_msg_mgr().log_debug('Find no frames in the sequence %s-%s-%s.'
                                                % (str(labs_batch[count]), str(typs_batch[count]), str(vies_batch[count])))

                    count += 1
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)

            for i in range(feature_num):
                for j in indices[:self.frames_all_limit] if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit else indices:
                    sampled_fras[i].append(seqs[i][j])

            temp_sampled_frames = [[] for i in range(feature_num)]
            if self.use_full_video:
                for ii, sampled_sequence in enumerate(sampled_fras):
                    num_seqs_from_video = int(len(sampled_sequence) / \
                        self.frames_num_fixed)

                    if self.random_sample_full_video:
                        random.shuffle(sampled_sequence)

                    for jj in range(num_seqs_from_video):
                        start = jj * self.frames_num_fixed
                        end = (jj+1) * self.frames_num_fixed
                        temp_sampled_frames[0].append(sampled_sequence[start:end])

            if len(temp_sampled_frames[0]) == 0:
                print('\ntemp: ', temp_sampled_frames)
                print('num_seqs_from_video: ', num_seqs_from_video)
                print('sampled: ', sampled_fras)
            sampled_fras = temp_sampled_frames
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        fras_batch = [sample_frames(seqs) for seqs in seqs_batch]  # [b, f]

        new_fras_batch = [[]]
        new_labs_batch = []
        new_typs_batch = []
        new_vies_batch = []

        for ii in range(len(fras_batch[0])):
            for jj in range(len(fras_batch[0][ii])):
                new_fras_batch[0].append(fras_batch[0][ii][jj])
                new_labs_batch.append(labs_batch[ii])
                new_typs_batch.append(typs_batch[ii])
                new_vies_batch.append(vies_batch[ii])

        if len(new_fras_batch[0]) == 0:
            print(10*'\n')
            print('fras_batch: ', fras_batch)
            print('batch_size: ', batch_size)
            print('len(fras_batch): ', len(fras_batch))
            print(10*'\n')

        fras_batch = new_fras_batch

        # batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]
        batch = [new_fras_batch,
                 new_labs_batch,
                 new_typs_batch,
                 new_vies_batch,
                 None]

        #print('len(new_fras_batch): ', len(new_fras_batch[0]))
        if self.sampler == "fixed":
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [f, b]
        else:
            if len(fras_batch[0]) == 0:
                print(10*'\n')
                print('len(seqs_batch): ', len(seqs_batch))
                print('fras_batch: ', fras_batch)
                print('batch_size: ', batch_size)
                print('len(fras_batch): ', len(fras_batch))
                print(10*'\n')
            seqL_batch = [[len(fras_batch[i][0])
                           for i in range(batch_size)]]  # [1, p]

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0)
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

            batch[-1] = np.asarray(seqL_batch)

        batch[0] = fras_batch
        return batch
