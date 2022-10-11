import sys
import os
from time import strftime, localtime
import torch
import numpy as np
import torch.nn.functional as F
from utils import get_msg_mgr, mkdir
import pickle


def remove_no_gallery(gallery):
    return {k: v for k, v in gallery.items() if v}


def remove_no_gallery_from_probe(gallery_subjects, probe):
    return {k: v for k, v in probe.items() if k in gallery_subjects}


def isProbeInTopN(probe_y, gallery_y, sorted_indices, rank):
    '''
    columns are N ranks
    rows are probes
    '''
    probe_in_top_N = np.cumsum(np.reshape(probe_y, [-1, 1]) == \
                               gallery_y[sorted_indices[:, 0:rank]], 1) > 0
    return probe_in_top_N


def findCorrectMatches(probe_in_top_N):
    correct_matches = probe_in_top_N.cumsum(axis=1) == 1
    return correct_matches


def isCorrectMatchWithinRankN(correct_matches, rank):
    # be sure to pass only mated probes, since if a row that is all false will
    # have argmax return 0
    probe_match_within_rank_N = correct_matches.argmax(axis=1) < rank
    return probe_match_within_rank_N


def doesCorrectMatchScorePassThreshold(correct_match_scores, threshold):
    score_pass_threshold = correct_match_scores < threshold
    return score_pass_threshold


def sortDistances(dist, num_rank):
    ranked_distances = dist.sort(1)[0][:, 0:num_rank]
    return ranked_distances


def findProbesWithAFalsePositive(ranked_distances, rank, threshold):
    # only pass in non-mated probes
    false_positives = (ranked_distances[:, :rank].cpu().numpy() < \
                       threshold).cumsum(axis=1).cumsum(axis=1) == 1
    return false_positives


def get_failure_rank_indices(false_in_rows_until_correct_match, failure_ranks):
    probe_x_failure_rank_indices = {}
    correct_matches = findCorrectMatches(false_in_rows_until_correct_match)
    for ii in failure_ranks:
        lower_rank_sequencs = isCorrectMatchWithinRankN(correct_matches,
                                                        ii - 1)
        upper_rank_sequencs = isCorrectMatchWithinRankN(correct_matches,
                                                        ii)
        failure_sequence_indices = list(set(np.where((lower_rank_sequencs == False) &
                                                     (upper_rank_sequencs == True))[0]))
        probe_x_failure_rank_indices[ii] = np.asarray(failure_sequence_indices)
    return probe_x_failure_rank_indices


def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=2)  # n p c
        y = F.normalize(y, p=2, dim=2)  # n p c
    num_bin = x.size(1)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, i, ...]
        _y = y[:, i, ...]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                1).transpose(0, 1) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin

# Exclude identical-view cases


def de_diag(acc, each_angle=False):
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result

# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def identification(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    # sample_num = len(feature)

    probe_seq_dict = {'CASIA-B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}

    gallery_seq_dict = {'CASIA-B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                    view_num, view_num, num_rank]) - 1.
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                        view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                        view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x, metric)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)
    result_dict = {}
    np.set_printoptions(precision=3, suppress=True)
    if 'OUMVLP' not in dataset:
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Include identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, i])
        result_dict["scalar/test_accuracy/BG"] = de_diag(acc[1, :, :, i])
        result_dict["scalar/test_accuracy/CL"] = de_diag(acc[2, :, :, i])
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, i], True)))
            msg_mgr.log_info('BG: {}'.format(de_diag(acc[1, :, :, i], True)))
            msg_mgr.log_info('CL: {}'.format(de_diag(acc[2, :, :, i], True)))
    else:
        msg_mgr.log_info('===Rank-1 (Include identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (np.mean(acc[0, :, :, 0])))
        msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (de_diag(acc[0, :, :, 0])))
        msg_mgr.log_info(
            '===Rank-1 of each angle (Exclude identical-view cases)===')
        msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, 0], True)))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, 0])
    return result_dict


def identification_briar(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type, fnames = (data['embeddings'],
                                        data['labels'],
                                        data['types'],
                                        data['views'])
    print('feature type: ', type(feature))
    print('feature shape: ', feature.shape)
    sys.exit()

    ranks = [0, 2, 4, 9, 14]
    failure_ranks = [1, 5, 10]
    num_rank = int(np.max(np.array(ranks)) + 1)
    label = np.array(label)

    gallery_mask = np.array([True if 'controlled' in seq else False for seq in seq_type])
    probe_mask = np.array([False if 'controlled' in seq else True for seq in seq_type])

    fnames_list = list(set(fnames))
    fnames_list.sort()
    fnames_num = len(fnames_list)
    # sample_num = len(feature)

    to_save = {'gallery': {}, 'probe': {}}
    # for subject in data['labels']:
    for f, l, s, v in zip(list(feature), label, seq_type, fnames):
        if l not in to_save['gallery'].keys():
            to_save['gallery'][l] = []
            to_save['probe'][l] = {}
        if s.startswith('controlled'):
            to_save['gallery'][l].append(f)
        else:
            to_save['probe'][l][f'{s}_{v}'] = f

    to_save['gallery'] = remove_no_gallery(to_save['gallery'])
    to_save['probe'] = remove_no_gallery_from_probe(to_save['gallery'].keys(),
                                                    to_save['probe'])

    gallery_collapsed = []
    label_collapsed = []
    for l, g in to_save['gallery'].items():
        label_collapsed.append(l)
        gallery_collapsed.append(np.stack(g).mean(0))
    label_collapsed = np.array(label_collapsed)
    gallery_collapsed = np.stack(gallery_collapsed)

    gallery_x = gallery_collapsed
    gallery_y = label_collapsed

    probe_seq_dict = {'CASIA-B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']],
                      'BRIAR': [[seq for seq in set(seq_type) if 'controlled' not in seq]]}

    gallery_seq_dict = {'CASIA-B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']],
                        'BRIAR': [[seq for seq in set(list(seq_type)) if 'controlled' in seq]]}

    acc = np.zeros([len(probe_seq_dict[dataset]),
                    fnames_num, fnames_num, num_rank]) - 1.

    probe_sequence_label_mask = np.zeros(len(seq_type), dtype=bool)
    gallery_labels = to_save['gallery'].keys()
    for probe_label in to_save['probe'].keys():
        if probe_label in gallery_labels:
            probe_sequence_label_mask[np.isin(label, probe_label)] = True

    pseq_mask = np.isin(seq_type,
                        probe_seq_dict[dataset]) & probe_sequence_label_mask
    print("Number of gallery videos: {}".format(np.sum(gallery_mask)))
    print("Number of probe videos:   {}".format(np.sum(probe_mask)))
    print("Number of probe w/gallery videos: {}".format(np.sum(pseq_mask)))
    print("Number of probe w/gallery videos + gallery: {}".format(np.sum(probe_sequence_label_mask)))

    eval_metric_pickle_fname = "./all_probe_test_metrics_probe_with_gallery_all_frames.pkl"
    acc_pickle_fname = "./accuracy_and_seqs.pkl"

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            # for (v1, probe_view) in enumerate(view_list):
            #     for (v2, gallery_view) in enumerate(view_list):
            gseq_mask = np.isin(seq_type, gallery_seq) # & np.isin(
            #     view, [gallery_view])
            '''
            gallery_x = feature[gseq_mask, :]
            gallery_y = label[gseq_mask]
            '''

            # open set
            # pseq_mask = np.isin(seq_type, probe_seq)

            # closed set
            pseq_mask = np.isin(seq_type,
                                probe_seq) & probe_sequence_label_mask

            probe_x = feature[pseq_mask, :]

            # tuple of probes len num_samples converted to vector
            probe_y = label[pseq_mask]
            probe_y_vector = np.reshape(probe_y, [-1, 1])

            # calculate distance between probe features and gallery features
            dist = cuda_dist(probe_x, gallery_x, 'cos')
            sorted_match_indices = dist.sort(1)[1].cpu().numpy()

            # sort the gallery based on the sorted match scores up to max rank
            gallery_match_score_sorted = gallery_y[sorted_match_indices[:, 0:num_rank]]

            # then, identify locations where the correct match appears
            # (num_probes x num_rank)
            correct_match_locations = (probe_y_vector ==
                                       gallery_y[sorted_match_indices[:, 0:num_rank]])

            # find the cumulative sum along the rows (add columns). Once
            # greater than zero correct match has been found, still
            # (num_probes x num_rank)
            false_in_rows_until_correct_match = np.cumsum(np.reshape(probe_y, [-1, 1]) ==
                                                          gallery_y[sorted_match_indices[:, 0:num_rank]],
                                                          1) > 0

            probe_x_failure_rank_indices = get_failure_rank_indices(false_in_rows_until_correct_match,
                                                                    failure_ranks)
            # probe_x_failure_rank_sequences = get_failure_rank_sequencs(probe_x_failure_rank_indices,
            #                                                           probe_x)

            # save_failure_sequences(probe_x_failure_rank_indices, output_path)

            # then sum this along the columns to see how many 1's in rank-1 column,
            # rank-5 column, etc.
            num_matches_in_given_column = np.sum(false_in_rows_until_correct_match, 0)

            # percentage rank-1 through rank-num_rank
            num_probes = dist.shape[0]
            rank_percentages = np.round(num_matches_in_given_column * 100 / num_probes, 2)

            acc[p, :, :, :] = rank_percentages

    '''
    with open(acc_pickle_fname, "wb") as f:
        pickle.dump(false_in_rows_until_correct_match, f)
        pickle.dump(acc, f)
        pickle.dump(probe_seq_dict[dataset], f)
        pickle.dump(gallery_seq_dict[dataset], f)
        pickle.dump(seq_type, f)
        pickle.dump(gallery_y, f)
        pickle.dump(label, f)
        pickle.dump(probe_sequence_label_mask, f)
        pickle.dump([jj for jj in gallery_labels], f)
        pickle.dump(to_save, f)
        pickle.dump(fnames, f)
    '''

    result_dict = {}
    np.set_printoptions(precision=3, suppress=True)
    for ii in ranks:
        msg_mgr.log_info(f'===Rank-{ii+1}===')
        msg_mgr.log_info('%.3f ' % (np.mean(acc[0, :, :, ii])))
    result_dict["scalar/test_accuracy/NM"] = acc[0, :, :, 0]
    return result_dict, probe_x_failure_rank_indices


def identification_real_scene(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0'],
                        'GREW': ['01']}
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1'],
                      'GREW': ['02']}

    num_rank = 20
    acc = np.zeros([num_rank]) - 1.
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = label[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.cpu().sort(1)[1].numpy()
    acc = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                          0) * 100 / dist.shape[0], 2)
    msg_mgr.log_info('==Rank-1==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[0])))
    msg_mgr.log_info('==Rank-5==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[4])))
    msg_mgr.log_info('==Rank-10==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[9])))
    msg_mgr.log_info('==Rank-20==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[19])))
    return {"scalar/test_accuracy/Rank-1": np.mean(acc[0]), "scalar/test_accuracy/Rank-5": np.mean(acc[4])}


def identification_GREW_submission(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating GREW")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)
    gallery_seq_type = {'GREW': ['01', '02']}
    probe_seq_type = {'GREW': ['03']}
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = view[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.cpu().sort(1)[1].numpy()

    save_path = os.path.join(
        "GREW_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("GREW_result")
    with open(save_path, "w") as f:
        f.write("videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n")
        for i in range(len(idx)):
            r_format = [int(idx) for idx in gallery_y[idx[i, 0:20]]]
            output_row = '{}'+',{}'*20+'\n'
            f.write(output_row.format(probe_y[i], *r_format))
        print("GREW result saved to {}/{}".format(os.getcwd(), save_path))
    return


def evaluate_HID(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating HID")
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]

    feat = np.concatenate([probe_x, gallery_x])
    dist = cuda_dist(feat, feat, metric).cpu().numpy()
    msg_mgr.log_info("Starting Re-ranking")
    re_rank = re_ranking(dist, probe_x.shape[0], k1=6, k2=6, lambda_value=0.3)
    idx = np.argsort(re_rank, axis=1)

    save_path = os.path.join(
        "HID_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("HID_result")
    with open(save_path, "w") as f:
        f.write("videoID,label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print("HID result saved to {}/{}".format(os.getcwd(), save_path))
    return


def re_ranking(original_dist, query_num, k1, k2, lambda_value):
    # Modified from https://github.com/michuanhaohao/reid-strong-baseline/blob/master/utils/re_ranking.py
    all_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                            :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
        original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def identification_briar_yiyang(data, dataset, metric='euc', save_embeddings=True,
						 use_subjects_as_galleries=False):
    feature, label, seq_type, view = (data['embeddings'],
                                      data['labels'],
                                      data['types'],
                                      data['views'])
    label = np.array(label)

    if save_embeddings:
        import pickle
        pickle.dump({'embeddings': feature, 'labels': label, 'types': seq_type, 'views': view},
                    open('embeddings.pkl', 'wb'))

    probe_mask = np.array([False if 'controlled' in seq else True for seq in seq_type])

    to_save = {'gallery': {}, 'probe': {}}
    for f, l, s, v in zip(list(feature), label, seq_type, view):
        if l not in to_save['gallery'].keys():
            to_save['gallery'][l] = []
            to_save['probe'][l] = {}
        if s.startswith('controlled'):
            to_save['gallery'][l].append(f)
        else:
            to_save['probe'][l][f'{s}_{v}'] = f

    def remove_no_gallery(gallery):
        return {k: v for k, v in gallery.items() if v}

    def remove_no_gallery_from_probe(gallery_subjects, probe):
        return {k: v for k, v in probe.items() if k in gallery_subjects}

    to_save['gallery'] = remove_no_gallery(to_save['gallery'])
    to_save['probe'] = remove_no_gallery_from_probe(to_save['gallery'].keys(),
                                                    to_save['probe'])

    if use_subjects_as_galleries:
        gallery_collapsed = []
        label_collapsed = []
        for l, g in to_save['gallery'].items():
            label_collapsed.append(l)
            gallery_collapsed.append(np.stack(g).mean(0))
        gallery_y = np.array(label_collapsed)
        gallery_x = np.stack(gallery_collapsed)

    print(f'Gallery Subjects: {sum([1 for l in to_save["gallery"].values() if len(l) > 0])}')
    print(f'Probe Subjects: {sum([1 for d in to_save["probe"].values() if len(d) > 0])}')

    probe_seq_dict = {'BTS1': [[seq for seq in set(seq_type) if 'controlled' not in seq],
                               [seq for seq in set(seq_type) if 'close_range' in seq],
                               [seq for seq in set(seq_type) if '100m' in seq],
                               [seq for seq in set(seq_type) if '200m' in seq],
                               [seq for seq in set(seq_type) if '270m' in seq],
                               [seq for seq in set(seq_type) if '300m' in seq],
                               [seq for seq in set(seq_type) if '370m' in seq],
                               [seq for seq in set(seq_type) if '400m' in seq],
                               [seq for seq in set(seq_type) if '490m' in seq],
                               [seq for seq in set(seq_type) if '500m' in seq],
                               [seq for seq in set(seq_type) if '600m' in seq],
                               [seq for seq in set(seq_type) if '800m' in seq],
                               [seq for seq in set(seq_type) if '1000m' in seq],
                               [seq for seq in set(seq_type) if 'uav' in seq],
                               [seq for seq in set(seq_type) if 'face' in seq],
                               [seq for seq in set(seq_type) if 'wb' in seq],
                               [seq for seq in set(seq_type) if 'rand' in seq],
                               [seq for seq in set(seq_type) if 'struct' in seq],
                               [seq for seq in set(seq_type) if 'stand' in seq]]}

    gallery_seq_dict = {'BTS1': [[seq for seq in set(list(seq_type)) if 'controlled' in seq]]}

    num_rank = 10
    acc = np.zeros([len(probe_seq_dict[dataset]), 1, 1, num_rank]) - 1.

    probe_sequence_label_mask = np.zeros(len(seq_type), dtype=bool)
    gallery_labels = to_save['gallery'].keys()
    for probe_label in to_save['probe'].keys():
        if probe_label in gallery_labels:
            probe_sequence_label_mask[np.isin(label, probe_label)] = True
    print(f'probe sequences: {np.sum(probe_sequence_label_mask)}/{probe_sequence_label_mask.shape[0]}')

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            gseq_mask = np.isin(seq_type, gallery_seq)
            if not use_subjects_as_galleries:
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

            pseq_mask = np.isin(seq_type, probe_seq) & probe_sequence_label_mask & probe_mask
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            dist = cuda_dist(probe_x, gallery_x, metric)
            idx = dist.sort(1)[1].cpu().numpy()
            pred_by_seq = gallery_y[idx]  # [num_probe, num_gallery]
            pred_by_subj = pred_by_seq[:, :len(to_save['gallery'])]  # just to get the shape and dtype right
            for i in range(pred_by_seq.shape[0]):
                pred_by_subj[i] = pred_by_seq[i][np.sort(np.unique(pred_by_seq[i], return_index=True)[1])]

            pred_by_subj = pred_by_subj[:, :num_rank]
            acc[p, :, :, :] = np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == pred_by_subj, 1) > 0, 0) * 100 / dist.shape[0]
    result_dict = {}
    with np.printoptions(precision=2, suppress=True):
        for i in [0, 2, 4, 9]:
            print(f'===Rank-{i + 1}===')
            print(f'All: {np.mean(acc[0, :, :, i]):.2f},\t'
                  f'CR: {np.mean(acc[1, :, :, i]):.2f}\t'
                  f'100: {np.mean(acc[2, :, :, i]):.2f}\t'
                  f'200: {np.mean(acc[3, :, :, i]):.2f}\t'
                  f'270: {np.mean(acc[4, :, :, i]):.2f}\t'
                  f'300: {np.mean(acc[5, :, :, i]):.2f}\t'
                  f'370: {np.mean(acc[6, :, :, i]):.2f}\t'
                  f'400: {np.mean(acc[7, :, :, i]):.2f},\n'
                  f'490: {np.mean(acc[8, :, :, i]):.2f}\t'
                  f'500: {np.mean(acc[9, :, :, i]):.2f}\t'
                  f'600: {np.mean(acc[10, :, :, i]):.2f}\t'
                  f'800: {np.mean(acc[11, :, :, i]):.2f}\t'
                  f'1000: {np.mean(acc[12, :, :, i]):.2f}\t'
                  f'uav: {np.mean(acc[13, :, :, i]):.2f}\t'
                  f'face: {np.mean(acc[14, :, :, i]):.2f}\t'
                  f'wb: {np.mean(acc[15, :, :, i]):.2f}\n'
                  f'rand: {np.mean(acc[16, :, :, i]):.2f}\t'
                  f'struct: {np.mean(acc[17, :, :, i]):.2f}\t'
                  f'stand: {np.mean(acc[18, :, :, i]):.2f}\t')
    print(f'All: {[round(x, 2) for x in acc[0, 0, 0]]}\n'
          f'CR : {[round(x, 2) for x in acc[1, 0, 0]]}\n'
          f'100: {[round(x, 2) for x in acc[2, 0, 0]]}\n'
          f'200: {[round(x, 2) for x in acc[3, 0, 0]]}\n'
          f'270: {[round(x, 2) for x in acc[4, 0, 0]]}\n'
          f'300: {[round(x, 2) for x in acc[5, 0, 0]]}\n'
          f'370: {[round(x, 2) for x in acc[6, 0, 0]]}\n'
          f'400: {[round(x, 2) for x in acc[7, 0, 0]]}\n'
          f'490: {[round(x, 2) for x in acc[8, 0, 0]]}\n'
          f'500: {[round(x, 2) for x in acc[9, 0, 0]]}\n'
          f'600: {[round(x, 2) for x in acc[10, 0, 0]]}\n'
          f'800: {[round(x, 2) for x in acc[11, 0, 0]]}\n'
          f'1km: {[round(x, 2) for x in acc[12, 0, 0]]}\n'
          f'uav: {[round(x, 2) for x in acc[13, 0, 0]]}\n'
          f'fac: {[round(x, 2) for x in acc[14, 0, 0]]}\n'
          f'wb : {[round(x, 2) for x in acc[15, 0, 0]]}\n'
          f'rdn: {[round(x, 2) for x in acc[16, 0, 0]]}\n'
          f'str: {[round(x, 2) for x in acc[17, 0, 0]]}\n'
          f'std: {[round(x, 2) for x in acc[18, 0, 0]]}\n'
    )
    result_dict["scalar/test_accuracy/NM"] = acc[0, :, :, 0]
    return result_dict


def identification_briar_view_yiyang(data, dataset, metric='euc',
									 save_embeddings=True,
									 use_subjects_as_galleries=False):
    feature, label, seq_type, view = (data['embeddings'],
                                      data['labels'],
                                      data['types'],
                                      data['views'])
    label = np.array(label)

    if save_embeddings:
        import pickle
        pickle.dump({'embeddings': feature, 'labels': label, 'types': seq_type, 'views': view},
                    open('embeddings.pkl', 'wb'))

    probe_mask = np.array([False if 'controlled' in seq else True for seq in seq_type])

    to_save = {'gallery': {}, 'probe': {}}
    for f, l, s, v in zip(list(feature), label, seq_type, view):
        if l not in to_save['gallery'].keys():
            to_save['gallery'][l] = []
            to_save['probe'][l] = {}
        if s.startswith('controlled'):
            to_save['gallery'][l].append(f)
        else:
            to_save['probe'][l][f'{s}_{v}'] = f

    def remove_no_gallery(gallery):
        return {k: v for k, v in gallery.items() if v}

    def remove_no_gallery_from_probe(gallery_subjects, probe):
        return {k: v for k, v in probe.items() if k in gallery_subjects}

    to_save['gallery'] = remove_no_gallery(to_save['gallery'])
    to_save['probe'] = remove_no_gallery_from_probe(to_save['gallery'].keys(),
                                                    to_save['probe'])

    if use_subjects_as_galleries:
        gallery_collapsed = []
        label_collapsed = []
        for l, g in to_save['gallery'].items():
            label_collapsed.append(l)
            gallery_collapsed.append(np.stack(g).mean(0))
        gallery_y = np.array(label_collapsed)
        gallery_x = np.stack(gallery_collapsed)

    print(f'Gallery Subjects: {sum([1 for l in to_save["gallery"].values() if len(l) > 0])}')
    print(f'Probe Subjects: {sum([1 for d in to_save["probe"].values() if len(d) > 0])}')

    views_set = sorted(list(set([v[:3] for v in view])))
    probe_seq_dict = {'BTS1': [[v for v in set(view) if v.startswith(distinct_view)] for distinct_view in views_set]}

    gallery_seq_dict = {'BTS1': [[seq for seq in set(list(seq_type)) if 'controlled' in seq]]}

    num_rank = 10
    acc = np.zeros([len(probe_seq_dict[dataset]), 1, 1, num_rank]) - 1.

    probe_sequence_label_mask = np.zeros(len(seq_type), dtype=bool)
    gallery_labels = to_save['gallery'].keys()
    for probe_label in to_save['probe'].keys():
        if probe_label in gallery_labels:
            probe_sequence_label_mask[np.isin(label, probe_label)] = True
    print(f'probe sequences: {np.sum(probe_sequence_label_mask)}/{probe_sequence_label_mask.shape[0]}')

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            gseq_mask = np.isin(seq_type, gallery_seq)
            if not use_subjects_as_galleries:
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

            pseq_mask = np.isin(view, probe_seq) & probe_sequence_label_mask & probe_mask
            print(f'{views_set[p]}: {sum(pseq_mask)}')
            if sum(pseq_mask) <= 0:
                continue
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            dist = cuda_dist(probe_x, gallery_x, metric)
            idx = dist.sort(1)[1].cpu().numpy()
            pred_by_seq = gallery_y[idx]  # [num_probe, num_gallery]
            pred_by_subj = pred_by_seq[:, :len(to_save['gallery'])]  # just to get the shape and dtype right
            for i in range(pred_by_seq.shape[0]):
                pred_by_subj[i] = pred_by_seq[i][np.sort(np.unique(pred_by_seq[i], return_index=True)[1])]

            pred_by_subj = pred_by_subj[:, :num_rank]
            acc[p, :, :, :] = np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == pred_by_subj, 1) > 0, 0) * 100 / dist.shape[0]
    result_dict = {}
    with np.printoptions(precision=2, suppress=True):
        for i in [0, 2, 4, 9]:
            print(f'===Rank-{i + 1}===')
            for j in range(len(views_set)):
                  print(f'{views_set[j]}: {np.mean(acc[j, :, :, i]):.2f},\t')
    for j in range(len(views_set)):
        print(f'views_set[j]: {[round(x, 2) for x in acc[j, 0, 0]]}\n')

    result_dict["scalar/test_accuracy/NM"] = acc[0, :, :, 0]
    return result_dict
