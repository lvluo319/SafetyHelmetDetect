# -*- coding: utf-8 -*-

import os
from gluoncv import model_zoo, data, utils
from mxnet import gluon
import mxnet as mx
import cv2

def main():
    ctx = mx.cpu()

    # net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=False)

    classes = ['hat', 'person']
    # initialize net
    # for param in net.collect_params().values():
    #     if param._data is not None:
    #         continue
    #     param.initialize()
    # net.reset_class(classes)
    # net.collect_params().reset_ctx(ctx)
    #
    # net.load_parameters('./model/darknet.params', ctx=ctx)

    net = gluon.SymbolBlock.imports(symbol_file='./model/darknet53-symbol.json', input_names=['data'],
                                    param_file='./model/darknet53-0000.params', ctx=ctx)


    no = 1
    cap = cv2.VideoCapture('vedio.mp4')
    # for file in os.listdir('./image'):
    while (cap.isOpened()):
        ret, frame = cap.read()
        # frame = './image/' + file
        x, orig_img = data.transforms.presets.yolo.load_test(frame, short=416)
        x = x.as_in_context(ctx)
        box_ids, scores, bboxes = net(x)
        if [1.] in list(box_ids[0].asnumpy()):
            # person confidences
            index_person = [i for i, x in enumerate(list(box_ids[0].asnumpy())) if x == [1.]]
            list_scores = list(scores[0].asnumpy())
            person_scores = [list_scores[i] for i in index_person]
            if max(person_scores) > 0.4:
                # draw bounding boxes on pic.
                # ax = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes,
                #                             thresh=0.4)
                ax = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=classes, thresh=0.4)
                cv2.imwrite('./results/' + str(no) + '.jpg', orig_img[..., ::-1])
                no += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    main()

