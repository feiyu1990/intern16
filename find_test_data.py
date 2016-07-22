import string
import csv
import operator
import os
import numpy as np
from collections import defaultdict
import json
from collections import Counter
import random
import shutil
root = '/mnt/ilcompf3d1/user/yuwang/docker_shared/fotolia/'


# test data that not in training set.
def test_data():
    word_per_caption = defaultdict(int)
    temp_wordcount = []

    dict_17_to_foto = dict()
    dict_6_to_foto_inv = dict()
    with open(root + 'trainIdx.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            dict_17_to_foto[int(line[0])] = int(line[1])
    with open(root + 'imgIDList_6m.txt') as f:
        for idx, line in enumerate(f):
            dict_6_to_foto_inv[dict_17_to_foto[int(line)]] = int(line)
    print len(dict_6_to_foto_inv)
    dict_17_to_foto_inv = dict([(i,j) for j,i in dict_17_to_foto.items()])
    lstm_imgs = []
    caption_file = '/mnt/ilcompf3d1/user/zlin/data/fotolia28m/data/DATASEARCH-144.csv'
    ii = 0
    with open(caption_file, 'r') as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            if int(line[0]) in dict_6_to_foto_inv or line[1] not in ['2', '3']:
                continue
            try:
                caption = str(line[2])
            except:
                continue
            if len(caption.split(' ')) <= 2:
                continue
            word_count = len(line[2].split(' '))
            word_per_caption[word_count] += 1
            temp_wordcount.append(word_count)
            temp = {}
            try:
                img_file = str(dict_17_to_foto_inv[int(line[0])]).zfill(9)
                temp['file_path'] = img_file[:3] + '/' + img_file[3:6] + '/' + img_file + '.jpg'
                if not os.path.isfile('/mnt/ilcompf2d1/data/fotolia17M/image/train/'+temp['file_path']):
                    print temp['file_path']
                    continue
                temp['id'] = dict_17_to_foto_inv[int(line[0])]
                temp['captions'] = [caption]
                lstm_imgs.append(temp)
            except:
                continue
            ii += 1
            if ii % 100000 == 0:
                print ii, line, dict_17_to_foto_inv[int(line[0])]
                # json.dump(lstm_imgs, open('test_raw_sample.json','w'))
                # break
    print len(lstm_imgs)
    # lst_imgs_sample = random.sample(lstm_imgs, 50000)
    json.dump(lstm_imgs, open(root + 'test_raw_all.json','w'))


def clean_test_data():
    train_id_list_jiam = set()
    with open(root + 'test_raw_all.json', 'r') as f:
        lstm_imgs = json.load(f)
    lstm_imgs_clean = []
    for idx, img in enumerate(lstm_imgs):
        s = img['captions'][0]
        try:
            temp = str(s).lower().translate(None, string.punctuation).strip().split()
        except:
            continue
        lstm_imgs_clean.append(img)
    json.dump(lstm_imgs_clean, open(root + 'test_raw_all_clean.json','w'))
    print len(lstm_imgs), len(lstm_imgs_clean)
    with open('/mnt/ilcompf3d1/user/jianmzha2/fotolia/trainTaskJPOrig/data/cleanPseudoClass/trainIdxmodelX') as f:
        for line in f:
            train_id_list_jiam.add(int(line.split('/')[-1].split('.')[0]))
    for i in lstm_imgs:
        if i['id'] in train_id_list_jiam:
            raise ValueError('Training and test overlapped!')

    # json.dump(lst_imgs_sample, open('test_raw.json','w'))


def create_test_dataset():
    data = json.load(open('test_raw.json'))
    data_sampled = random.sample(data, 500)
    for i in data_sampled:
        try:
            shutil.copy(i['file_path'], 'test_data/' + ('_').join(str(i['captions'][0]).translate(None, string.punctuation).split(' '))+'.jpg')
        except:
            continue


def add_user_info_to_test():
    dict_stock_to_user = dict()
    dict_17_to_user = dict()
    dict_fotolia_to_17 = dict()
    meta_root = '/mnt/ilcompf3d1/user/zlin/data/fotolia28m/data/mid/'
    file_list = [file_name for file_name in os.listdir(meta_root) if file_name.endswith('.csv')]
    for file_name in file_list:
        with open(meta_root + file_name) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                dict_stock_to_user[int(row[0])] = int(row[1])
    print len(dict_stock_to_user)
    caption_file = '/mnt/ilcompf3d1/user/zlin/data/fotolia28m/data/DATASEARCH-144.csv'
    dict_fotolia_to_caption = dict()
    with open(caption_file) as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            dict_fotolia_to_caption[int(line[0])] = line[2]
    with open(root + 'trainIdx.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            dict_17_to_user[int(line[0])] = (int(line[1]), dict_stock_to_user[int(line[1])])
            dict_fotolia_to_17[int(line[1])] = int(line[0])
    print len(dict_17_to_user)
    with open(root + 'dict_fotolia17_to_user.json', 'w') as f:
        json.dump(dict_17_to_user, f)
    with open(root + 'train_raw_neuraltalk_cleaned.json') as f:
        training_info = json.load(f)
    with open(root + 'test_raw_all_clean.json') as f:
        test_info = json.load(f)
    print len(test_info)
    training_user_img_dict = defaultdict(list)
    for i in training_info:
        temp = dict_17_to_user[i['id']]
        # print temp
        training_user_img_dict[temp[1]].append(temp[0])
    training_user_img_dict_np = dict()
    for i, j in training_user_img_dict.items():
        training_user_img_dict_np[i] = np.asarray(j)

    # for i, j in training_user_img_dict.items():
    #     print i, len(j)
    test_cleaned_info = []
    test_dumped_info = []
    for idx, i in enumerate(test_info):
        j = i.copy()
        fotolia_id, usr_id = dict_17_to_user[i['id']]
        if usr_id in training_user_img_dict_np:
            training_fotolia_list = training_user_img_dict_np[usr_id]
            temp = np.abs(training_fotolia_list-fotolia_id)
            nn_fotolia_id = training_fotolia_list[np.argmin(temp)]
            j['nn_fotolia_id'] = dict_fotolia_to_17[nn_fotolia_id]
            temp_id = str(dict_fotolia_to_17[nn_fotolia_id]).zfill(9)
            j['nn_fotolia_file_path'] = temp_id[:3] + '/' + temp_id[3:6] + '/' + temp_id + '.jpg'
            j['nn_fotolia_distance'] = np.min(temp)
            j['nn_caption'] = dict_fotolia_to_caption[nn_fotolia_id]
            # if j['nn_fotolia_distance'] < 100:
                # test_dumped_info.append(j)
                # continue
        test_cleaned_info.append(j)
    print len(test_cleaned_info), len(test_dumped_info)
    # json.dump(test_cleaned_info, open(root + 'test_raw_cleaned_with_usr.json','w'))
    # json.dump(test_dumped_info, open(root + 'test_raw_dumped_with_usr.json','w'))
    json.dump(test_cleaned_info, open(root + 'test_raw_with_usrinfo.json','w'))



def filter_test_with_user(n_threshold):
    no_repeat_user = []
    no_close_stream = []
    test_info = json.load(open(root + 'test_raw_with_usrinfo.json'))
    print len(test_info)
    for idx, i in enumerate(test_info):
        if 'nn_fotolia_id' not in i:
            no_repeat_user.append(i)
        elif i['nn_fotolia_distance'] > n_threshold:
            no_close_stream.append(i)
        if idx % 10000 == 0:
            print idx
    # no_close_stream.extend(no_repeat_user)
    print len(no_close_stream), len(no_repeat_user)
    json.dump(no_close_stream, open(root + 'test_raw_no_closestream.json','w'))
    # json.dump(no_repeat_user, open(root + 'test_raw_no_repeatusr.json','w'))



def test_data_from_event():
    dict_name2 = {'ThemePark':1, 'UrbanTrip':2, 'BeachTrip':3, 'NatureTrip':4,
             'Zoo':5,'Cruise':6,'Show':7,
            'Sports':8,'PersonalSports':9,'PersonalArtActivity':10,
            'PersonalMusicActivity':11,'ReligiousActivity':12,
            'GroupActivity':13,'CasualFamilyGather':14,
            'BusinessActivity':15, 'Architecture':16, 'Wedding':17, 'Birthday':18, 'Graduation':19, 'Museum':20,'Christmas':21,
            'Halloween':22, 'Protest':23}
    all_img_paths = set()
    for event_name in dict_name2:
        with open('/mnt/ilcompf3d1/user/yuwang/event_curation/baseline_all_0509/' + event_name + '/guru_training_path.txt') as f:
            for line in f:
                all_img_paths.add(line.split(' ')[0])
    print len(all_img_paths)
    sampled_ = random.sample(all_img_paths, 500)
    for i in sampled_:
        shutil.copy('/mnt/ilcompf3d1/user/yuwang/' + '/'.join(i.split('/')[4:]), root + '../test_imgs_event/')

if __name__ == "__main__":
    # clean_test_data()
    add_user_info_to_test()
    # filter_test_with_user(n_threshold=1000)
    # test_data()
    # test_data_from_event()