import mxnet as mx
import cv2
import numpy as np
from collections import namedtuple

model_prefix ='./model/reid_resnet-34_baili_256x128-20180520'
epoch = 52

im_width = 128
im_height = 256

#model_prefix = 'lmks-model/lmks-and-attr/resnet18'


def generateMod(model_prefix, epoch, im_width, im_height):
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, im_height, im_width))], \
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)


def extract_feature(im_patch, mod):
    #initialize model
    im = cv2.resize(im_patch,(im_width, im_height))
    im_patch = im.transpose(2,0,1)
    im_input = np.zeros((1,3,im_height,im_width))
    im_input[0,:,:,:] = im_patch 
    Batch = namedtuple('Batch',['data'])
    mod.forward(Batch([mx.nd.array(im_input)]))
    feature = mod.get_outputs()[0].asnumpy()
    return feature

def generateDets(detections_in, model_prefix, image_filenames):
    frame_indices = detections_in[:, 0].astype(np.int)
    min_frame_idx = frame_indices.astype(np.int).min()
    max_frame_idx = frame_indices.astype(np.int).max()
    mod = generateMod(model_prefix, 52, 128, 256)
    detections = []
    for frame_idx in range(min_frame_idx, max_frame_idx + 1):
        print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
        mask = frame_indices == frame_idx
        rows = detections_in[mask]

        if frame_idx not in image_filenames:
            print("WARNING could not find image for frame %d" % frame_idx)
            continue
        bgr_image = cv2.imread(
        image_filenames[frame_idx], cv2.IMREAD_COLOR)
        for row in rows:
            x = np.maximum(0, int(row[2]))
            y = np.maximum(0, int(row[3]))
            w = int(row[4])
            h = int(row[5])
            tempImg = bgr_image[y:y + h, x:x + w]
            tempFeat = extract_feature.extract_feature(tempImg, mod)
            tempDet = np.concatenate((row, tempFeat[0, :]), axis=0)
            detections.append(tempDet)
    return detections