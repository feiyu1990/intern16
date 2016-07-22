import numpy as np
import json

def combine_multi_methods(input_html, json_file_list, new_json_name):
    temp = []
    for json_file in json_file_list:
        with open(json_file) as f:
            temp.append(json.load(f))
    caption_list = temp[0]
    caption_ground = None
    for idx, i in enumerate(caption_list):
        # if 'caption_ground' in caption_list[idx]:
        #     caption_ground = caption_list[idx]['caption_ground']
        for j in range(1, len(temp)):
            caption_list[idx]['caption'] += '<br><span class="caption' + str(j + 1) + '">' + temp[j][idx]['caption'] + '</span>'
            # if caption_ground and caption_ground != temp[j][idx]['caption_ground']:
            #     print caption_ground
            #     print temp[j][idx]['caption_ground']
            #     raise ValueError('GROUND NOT MATCHED!')
    with open(new_json_name, 'w') as f:
        json.dump(caption_list, f)

    lines = []
    with open(input_html) as f:
        for line in f:
            lines.append(line)
    with open(new_json_name.split('.')[0] + '.html', 'w') as f:
        for line in lines:
            if 'vis.json?t=' in line:
                line = line.replace('vis.json', new_json_name.split('/')[-1])
            if '$(first_json)' in line:
                line = line.replace('$(first_json)', json_file_list[0].split('/')[1])
                line = line.replace('$(second_json)', json_file_list[1].split('/')[1])
            f.write(line)


def combine_knn_result(input_json):
    with open(root + 'test_minlen2_50w_knn_info.json') as f:
        knn_info = json.load(f)
    with open(root + 'dict_fotolia17_to_user.json') as f:
        id_to_user = json.load(f)
    with open(input_json) as f:
        all_info = json.load(f)
    all_info_new = []
    for j,i in zip(knn_info, all_info):
        i['usr_id'] = id_to_user[str(i['image_id'])][1]
        i['nn_visual'] = []
        for idx, ii in enumerate(j):
            if id_to_user[str(int(ii['file_path'].split('/')[-1].split('.')[0]))][1] == i['usr_id']:
                i['nn_visual'].append('SAME USER! ' + str(ii['score']) + '<br>' + str(ii['caption_ground']))
            else:
                i['nn_visual'].append(str(ii['score']) + '<br>' + str(ii['caption_ground']))
        all_info_new.append(i)
    with open(input_json, 'w') as f:
        json.dump(all_info_new, f)


def vis_knn(knn_json, img_list_file, caption_json, out_json):
    file_list = []
    with open(img_list_file) as f:
        for line in f:
            file_list.append(line.split('.')[0])
    with open(root + '../fotolia/dict_fotolia17_to_user.json') as f:
        id_to_user = json.load(f)
    caption_predict_info = json.load(open(caption_json))
    knn_info = json.load(open(knn_json))
    img_all_info = []
    idx = 1
    for img, img_name in zip(knn_info, file_list):
        if idx > 100:
            break
        temp = {}
        if img_name == str(idx):
            img_this = caption_predict_info[idx - 1]
            temp['caption_ground'] = img_this['caption_ground']
            temp['caption'] = img_this['caption']
            temp['img_name'] = img_name
            # usr_id = id_to_user[str(img_this['image_id'])][1]
            idx += 1
        temp['nn_visual'] = []

        for j in img:
            # print j['file_path']
            # print id_to_user[str(int(j['file_path'].split('/')[-1].split('.')[0]))]
            # print j['score']
            # print j['caption_ground']
            temp['nn_visual'].append(str(j['score']) + '<br>' + str(j['caption_ground']))
        img_all_info.append(temp)
    with open(out_json, 'w') as f:
        json.dump(img_all_info, f)


if __name__ == '__main__':
    root = 'vis-fotolia-noclosestream/'
    # combine_multi_methods(root + 'index.html', [root + 'vis-haystack.json', root + 'vis-vgg.json',
    #                                             root + 'vis-phrase.json'], root + 'vis-haystack-vgg-phrase.json')
    # combine_multi_methods(root + 'index.html', [root + 'vis-haystack.json', root + 'vis-vgg.json'],
    #                       root + 'vis-haystack-vgg.json')
    # combine_multi_methods(root + 'index-nn-new.html', [root + 'vis-haystack.json', root + 'vis-vgg.json'],
    #                       root + 'vis-haystack-vgg-nn.json')
    # combine_knn_result(root + 'vis-haystack-vgg-nn.json')
    root = '../fotolia_knn/'
    vis_knn(root + 'subimg_test_knn_info.json', root + 'subimg_list.txt', root + 'vis-haystack.json', root + 'vis-knn.json')
