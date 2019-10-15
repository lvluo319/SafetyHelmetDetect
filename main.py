# -*- coding: utf-8 -*-

import os
from gluoncv import data, utils
from mxnet import gluon, nd
import mxnet as mx
import cv2

def main():
    ctx = mx.cpu()
    classes = ['hat', 'person']
    net = gluon.SymbolBlock.imports(symbol_file='./model/darknet53-symbol.json', input_names=['data'],
                                    param_file='./model/darknet53-0000.params', ctx=ctx)

    # This must be a fullpath
    cap = cv2.VideoCapture('E:/work and learn/learning/deep learning/code/SafetyHelmetDetect_master/vedio/street.mp4')

    number = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = nd.array(frame[..., ::-1])
        x, orig_img = data.transforms.presets.yolo.transform_test(frame, short=512)
        x = x.as_in_context(ctx)
        box_ids, scores, bboxes = net(x)
        if [1.] in list(box_ids[0].asnumpy()):
            # person confidences
            index_person = [i for i, x in enumerate(list(box_ids[0].asnumpy())) if x == [1.]]
            list_scores = list(scores[0].asnumpy())
            person_scores = [list_scores[i] for i in index_person]
            if max(person_scores) > 0.4:
                # draw bounding boxes on pic.
                ax = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=classes, thresh=0.4)
                cv2.imwrite('./results/' + str(number) + '.jpg', orig_img[..., ::-1])
                number += 1

    cv2.destroyAllWindows()

if __name__ == '__main__':
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    main()

