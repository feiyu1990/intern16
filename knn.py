import string
import csv
import operator
import os
import numpy as np
from collections import defaultdict
import json
from collections import Counter
# import matplotlib.pyplot as plt
# from nltk.corpus import wordnet as wn
import random
root = '/mnt/ilcompf3d1/user/yuwang/docker_shared/fotolia_knn/'
import shutil

def create_hash():
    csv_path = '/mnt/ilcompf3d1/user/jianmzha2/data/expTagging/imgList6M.csv'
    check_path = '../fotolia/imgIDList_6m.txt'
    check_img_id = []
    with open(check_path) as f:
        for line in f:
            check_img_id.append(int(line))
    mapping = dict()
    with open(csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            mapping[idx] = row[0].split(' ')[0]
            img_id = int(row[0].split('/')[-1].split('.')[0])
            if img_id != check_img_id[idx]:
                raise ValueError('IDX Not matched!')
    with open(root + 'idx_matching.json', 'w') as f:
        json.dump(mapping, f)
    mapping_inverse = dict([(j, i) for i, j in mapping.items()])
    with open(root + 'idx_matching_inverse.json', 'w') as f:
        json.dump(mapping_inverse, f)


def image_feature_from_pretrained():
    with open(root + 'idx_matching_inverse.json') as f:
        mapping= json.load(f)
    split_file = root + '../fotolia/training/train_minlen2_50w.json'
    img_info = json.load(open(split_file))['images']
    split_dict = defaultdict(list)
    for img in img_info:
        idx = int(mapping[img['file_path'][6:]])
        split_dict[img['split']].append(idx)
    feature_path = root + 'fotolia6M_train_feat.bin'
    f = open(feature_path, 'r')
    data = np.fromfile(f, dtype=np.float32)
    data_reshape = data.reshape((len(data)/1024, 1024))
    print data.shape
    print data
    print data_reshape
    training_data = data_reshape[split_dict['train']]
    test_data = data_reshape[split_dict['test']]
    val_data = data_reshape[split_dict['val']]
    print training_data.shape, test_data.shape, val_data.shape
    # np.savez('../fotolia_knn/train_minlen2_50w.npz', training=training_data, test=test_data, val=val_data)
    training_data.tofile('../fotolia_knn/train_minlen2_50w.bin')
    test_data.tofile('../fotolia_knn/test_minlen2_50w.bin')
    val_data.tofile('../fotolia_knn/val_minlen2_50w.bin')


def read_knn_result():
    split_file = root + '../fotolia/training/train_minlen2_50w.json'
    img_info = json.load(open(split_file))['images']
    img_files = []
    for i in img_info:
        if i['split'] == 'train':
            img_files.append((i['file_path'], i['caption_ground']))
    nn_info = []
    img_list = []
    with open(root + '../fotolia_knn/subimg_list.txt') as f:
        for line in f:
            img_list.append(line.split('.')[0])

    with open(root + 'subimg_test_knn_result.txt') as f:
        for idx, line in enumerate(f):
            temp1 = []
            for i in xrange(10):
                temp = {}
                img_id, score = (int(line.split(',')[2 * i]), float(line.split(',')[2 * i + 1]))
                print img_id, score
                shutil.copy('/mnt/ilcompf2d1/data/fotolia17M/image/' + img_files[img_id][0], '../fotolia_knn/subimages/' + img_list[idx] + '_nn' +str(i) + '.jpg')
                temp['caption_ground'] = img_files[img_id][1]
                temp['file_path'] = img_files[img_id][0]
                temp['score'] = score
                temp1.append(temp)
            print temp1
            nn_info.append(temp1)
    save_knn_file = root + 'subimg_test_knn_info.json'
    json.dump(nn_info, open(save_knn_file, 'w'))

if __name__ == '__main__':
    # create_hash()
    # image_feature_from_pretrained()
    read_knn_result()