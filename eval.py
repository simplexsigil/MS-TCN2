#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py
import os

import numpy as np
import argparse
from data_utils import *

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


label2idx = {
    "walk": 0,
    "fall": 1,
    "fallen": 2,
    "sit_down": 3,
    "sitting": 4,
    "lie_down": 5,
    "lying": 6,
    "stand_up": 7,
    "standing": 8,
    "other": 9,
}

idx2label = {v: k for k, v in label2idx.items()}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="breakfast")
    parser.add_argument('--split', default='1')
    parser.add_argument('--target_fps', default='10', type=int)

    args = parser.parse_args()

    dataset_root = "/pfs/work8/workspace/ffhk/scratch/kf3609-ws/data/omnifall"

    ground_truth_path = os.path.join(dataset_root, "segmentation_annotations", "labels", "full.csv")
    recog_path = "./results/"+args.dataset+"/split_"+args.split+"/"
    file_list = os.path.join(dataset_root, "segmentation_annotations", "splits", "test_cs.csv")

    df = pd.read_csv(file_list)
    first_column_name = df.columns[0]
    list_of_videos = df[first_column_name].tolist()
    list_of_videos = [i for i in list_of_videos if i not in excluded_samples]

    gts_segments = load_temporal_labels(ground_truth_path)

    overlap = [.1, .25, .5]

    test_list = [
        "all",
        "caucafall",
        "cmdfall",
        "edf",
        "gmdcsa24",
        "le2i",
        "mcfd",
        "occu",
        "up_fall",
        "OOPS",  # only for test!
    ]

    for subset in test_list:
        print("#####Evaluating subset: {}".format(subset))
        if subset == "all":
            list_of_videos_subset = list_of_videos
        else:
            list_of_videos_subset = [i for i in list_of_videos if subset in i]

        if len(list_of_videos_subset) == 0:
            print("No videos for subset {}. Skipping.".format(subset))
            continue

        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct = 0
        total = 0
        edit = 0

        for vid in list_of_videos_subset:
            recog_file = recog_path + vid.split('/')[-1].split('.')[0]
            recog_content = read_file(recog_file).split('\n')[1].split()

            # Prepare labels
            num_frames = len(recog_content)
            gt_segments = gts_segments[vid]
            gt = convert_segments_to_segmentations(gt_segments, num_frames, fps=args.target_fps, default_class_id=9)
            gt_content = [idx2label[i] for i in gt]

            for i in range(len(gt_content)):
                total += 1
                if gt_content[i] == recog_content[i]:
                    correct += 1

            edit += edit_score(recog_content, gt_content)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1

        print("Acc: %.4f" % (100*float(correct)/total))
        print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos_subset)))
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])

            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100
            print('F1@%0.2f: %.4f' % (overlap[s], f1))

if __name__ == '__main__':
    main()
