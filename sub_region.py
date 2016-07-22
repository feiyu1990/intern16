import json
from PIL import Image, ImageDraw
import numpy as np


# root = '../external_package/densecap/vis/data/'
root = '/home/densecap/vis/data/'
save_root = '/home/fotolia_knn/subimages/'


def from_dense_cap(json_file):
    with open('/home/fotolia_knn/dict_high_res.json') as f:
        img_highres_path = json.load(f)

    with open(json_file) as f:
        result = json.load(f)
    result_imgs = result['results']
    img_list = []
    for result_img in result_imgs:
        img_size = Image.open(root + result_img['img_name']).size
        temp = result_img['img_name'].split('/')[-1].split('_')[2].zfill(8)
        temp_path = '/'.join([temp[2*i:2*(i+1)] for i in xrange(3)])
        ori_img = Image.open('/data_highres/' + temp_path + '/' + result_img['img_name'])
        ori_img_size = ori_img.size
        scale = float(ori_img_size[0]) / img_size[0]
        boxes = np.asarray(result_img['boxes'])
        scores = np.asarray(result_img['scores'])
        keep = np.where(scores > 0)[0]
        boxes_keep = boxes[keep,:]
        boxes_keep = list(np.asarray(boxes_keep * scale, dtype=int))
        print boxes_keep
        idx = 0
        for (x, y, w, h) in boxes_keep:
            if float(w * h) / (ori_img_size[0] * ori_img_size[1]) < 0.01:
                continue
            if idx > 4:
                break
            x -= int(0.1 * w)
            w += int(0.1 * w)
            y -= int(0.1 * h)
            h += int(0.1 * h)
            box = (max(0, x), max(0, y), min(x + w, ori_img_size[0]), min(y + h, ori_img_size[1]))
            dr = ImageDraw.Draw(ori_img)
            dr.rectangle(list(box), outline='red')
        ori_img.save(save_root + str(img_highres_path[temp_path + '/' + result_img['img_name']]) + '_all.jpg')

    #         img_crop = ori_img.crop(box)
    #         img_crop.save(save_root + str(img_highres_path[temp_path + '/' + result_img['img_name']]) + '_' + str(idx) + '.jpg')
    #         img_list.append(str(img_highres_path[temp_path + '/' + result_img['img_name']]) + '_' + str(idx) + '.jpg')
    #         idx += 1
    #     ori_img.save(save_root + str(img_highres_path[temp_path + '/' + result_img['img_name']]) + '.jpg')
    #     img_list.append(str(img_highres_path[temp_path + '/' + result_img['img_name']]) + '.jpg')
    #
    # with open(save_root + '../subimg_list.txt', 'w') as f:
    #     for line in img_list:
    #         f.write(line + '\n')


if __name__ == '__main__':
    # temp = json.load(open('/home/neuraltalk2/vis/vis.json'))
    # img_id_list = []
    # for i in temp:
    #     img_id_list.append(i['image_id'])
    # high_res_path = json.load(open('../fotolia/dict_fotolia17_to_high_res.json'))
    # print len(high_res_path)
    # img_highres_path = dict()
    # # f = open('/home/test_img_list_all_highres.txt', 'w')
    # for idx, i in enumerate(img_id_list):
    #     try:
    #         img_highres_path[high_res_path[str(i)]] = idx + 1
    # #         f.write(img_highres_path[idx + 1] + '\n')
    #     except:
    #         print idx, i
    # # f.close()
    # with open('/home/fotolia_knn/dict_high_res.json', 'w') as f:
    #     json.dump(img_highres_path, f)

    from_dense_cap('/home/densecap/vis/data/results.json')