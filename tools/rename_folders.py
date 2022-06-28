import os
import os.path as osp
from pathlib import Path


a = '/research/iprobe-ashbau12/datasets/opengait-briar/BTS1'

subjects = [ii for ii in os.listdir(a) if '.' not in ii]
subjects.sort()

subject_id_strings = []
for ii, subject in enumerate(subjects):
    subject_id = '{:03}'.format(int(subject[1:]))
    subject_id_strings.append(subject_id)
    old_folder = osp.join(a, subject)
    new_folder = osp.join(a, subject_id)
    os.rename(old_folder, new_folder)

print(subject_id_strings)
