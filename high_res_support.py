import json
import csv
import os
root = '/mnt/ilcompf3d1/user/yuwang/docker_shared/fotolia/'
high_root = '/mnt/ilcompf1d0/data/stock/images_tmp/00/'
from PIL import Image

def high_res_mapping():
    dict_17_to_foto = dict()
    count_invalid = 0
    with open(root + 'trainIdx.csv', 'r') as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            temp = str(int(line[1])).zfill(8)
            temp_path = '/'.join([temp[2*i:2*(i+1)] for i in xrange(3)])
            file_path = os.path.join(high_root, temp_path)
            try:
                file_name = [i for i in os.listdir(file_path) if i.startswith(str('1000_F_' + str(int(line[1]))))][0]
                dict_17_to_foto[int(line[0])] = os.path.join(temp_path, file_name)
                if idx % 100000 == 0:
                    print idx, os.path.join(temp_path, file_name)
            except:
                count_invalid += 1
                if count_invalid % 1000 == 0:
                    print count_invalid
                continue
    with open(root + 'dict_fotolia17_to_high_res.json', 'w') as f:
        json.dump(dict_17_to_foto, f)
    print len(dict_17_to_foto), count_invalid


def high_res_training():
    img_info_ori = json.load(open('../fotolia/train_raw_neuraltalk_cleaned.json'))
    high_res_path = json.load(open('../fotolia/dict_fotolia17_to_high_res.json'))
    img_info_high = []
    for i in img_info_ori:
        if str(i['id']) in high_res_path:
            i['low_res_path'] = i['file_path']
            i['file_path'] = high_res_path[str(i['id'])]
            img_info_high.append(i)
    print len(img_info_high), len(img_info_ori)
    json.dump(img_info_high, open('../fotolia_highres/train_raw_neuraltalk_cleaned.json', 'w'))


def high_res_test():
    img_info_ori = json.load(open('../fotolia/test_raw_no_closestream.json'))
    high_res_path = json.load(open('../fotolia/dict_fotolia17_to_high_res.json'))
    img_info_high = []
    for i in img_info_ori:
        if str(i['id']) in high_res_path:
            i['low_res_path'] = i['file_path']
            i['file_path'] = high_res_path[str(i['id'])]
            img_info_high.append(i)
    print len(img_info_high), len(img_info_ori)
    json.dump(img_info_high, open('../fotolia_highres/test_raw_no_closestream.json', 'w'))


def check_high_res_exist(name):
    info = json.load(open('../fotolia_highres/' + name + '.json'))
    info_cleaned = []
    for idx, img in enumerate(info):
        try:
            Image.open(high_root + img['file_path'])
            info_cleaned.append(img)
        except:
            print idx, img['file_path']
        if idx % 1000 == 0:
            print idx
    print len(info), len(info_cleaned)
    json.dump(info_cleaned, open('../fotolia_highres/' + name + '_imgcheck.json', 'w'))


if __name__ == '__main__':
    # high_res_training()
    # high_res_test()
    # check_high_res_exist('train_raw_neuraltalk_cleaned')
    check_high_res_exist('test_raw_no_closestream')