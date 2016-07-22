
caffe_root = '/root/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import json
root = '/home/fotolia/'
import numpy as np
from scipy.misc import imread, imresize
import h5py


def extract_feature(img_path):
        model_name = '/home/neuraltalk2/model/haystack.prototxt'
        weight_name = '/home/neuraltalk2/model/haystack.caffemodel'
        caffe.set_mode_gpu()
        net = caffe.Net(model_name,
                    weight_name,
                    caffe.TEST)
        # mean = np.array([103.939, 116.779, 123.68]) #BGR
        # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        # transformer.set_transpose('data', (2,0,1))
        # transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        # transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
        # transformer.set_mean('data', mean)  # the reference model has channels in BGR order instead of RGB
        temp_ = np.asarray(json.load(open(root + '/test_caffe_img.json')))
        temp = temp_[:,[2,1,0],:,:]
        net.blobs['data'].reshape(1,3,224, 224)
        net.blobs['data'].data[...] = temp
        # temp = caffe.io.load_image(img_path)
        # print temp.shape
        # net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img_path))
        input_data = net.blobs['data'].data[:]
        out = net.forward()
        print input_data.shape
        print out['fc7'].shape
        # temp = list(out['fc7'].)
        # print len(temp), temp
        # json.dump(out['fc7'].flatten(), open(root + '/test_caffe_result.json', 'w'))
        np.save(root + '/test_caffe_result.npy', out['fc7'].flatten())
        np.save(root + '/test_caffe_data.npy', input_data)
        # np.save(root + '/test_caffe_data_raw.npy', temp)

def extract_raw_img(img_path):
        I = imread(img_path)
        print I.shape
        np.save(root + '/test_caffe_data_raw_torch.npy', I)
        try:
            Ir = imresize(I, (224,224))
        except:
            raise
        Ir = Ir.transpose(2,0,1)
        print Ir.shape
        f = h5py.File(root + '/test_caffe_input_torch.h5', "w")
        dset = f.create_dataset("images", (1,3,224,224), dtype='uint8')
        dset[0] = Ir
        f.close()


if __name__ == '__main__':
    img_path = '/data/fotolia17M/image/train/000/233/000233460.jpg'
    extract_feature(img_path)
    # extract_raw_img(img_path)