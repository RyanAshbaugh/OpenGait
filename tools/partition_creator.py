import json
import pandas as pd
from os import path as osp

output_partition_path = "BGC_full_partition.json"

bgc_base_path = '/research/iprobe-ashbau12/hpcc_briar/ORNL/'
train_set = []
test_set = []

bgc1_split_file = 'BGC1/CollectInfo/BGC1_Participant_Split.csv'
bgc1_1_split_file = 'BGC1.1/CollectInfo/BGC1.1_Participant_Split.csv'
bgc2_split_file = 'BGC2/CollectInfo/BGC2_Participant_Split.csv'

bgc1_df = pd.read_csv(osp.join(bgc_base_path, bgc1_split_file))
bgc1_1_df = pd.read_csv(osp.join(bgc_base_path, bgc1_1_split_file))
bgc2_df = pd.read_csv(osp.join(bgc_base_path, bgc2_split_file))

bgc1_train_subjects = bgc1_df.loc[bgc1_df['group'] == 'R&D',
                                  'Participant ID']
bgc1_test_subjects = bgc1_df.loc[bgc1_df['group'] == 'T&E',
                                 'Participant ID']

bgc1_1_train_subjects = bgc1_1_df.loc[bgc1_1_df['Group'] == 'BRS',
                                      'Participant ID']
bgc1_1_test_subjects = bgc1_1_df.loc[bgc1_1_df['Group'] == 'BTS/full',
                                     'Participant ID']

bgc2_train_subjects = bgc2_df.loc[bgc2_df['eval_set'] == 'BRS2',
                                  'subj_id']
bgc2_test_subjects = bgc2_df.loc[bgc2_df['eval_set'] == 'BTS2/full',
                                 'subj_id']

train_set.extend(bgc1_train_subjects)
test_set.extend(bgc1_test_subjects)

train_set.extend(bgc1_1_train_subjects)
test_set.extend(bgc1_1_test_subjects)

train_set.extend(bgc2_train_subjects)
test_set.extend(bgc2_test_subjects)

partition_dictionary = {}
partition_dictionary['TRAIN_SET'] = train_set
partition_dictionary['TEST_SET'] = test_set

with open(output_partition_path, 'w') as f:
    json.dump(partition_dictionary, f)
