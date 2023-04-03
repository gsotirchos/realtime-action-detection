import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data import (CLASSES, AnnotationTransform, BaseTransform, UCF24Detection,
                  detection_collate, v2)
from layers.box_utils import decode, nms
from ssd import build_ssd
from torchinfo import summary

CUDA = False
BASENET = "./rgb-ssd300_ucf24_120000.pth"  # pretrained model parameter file
DATASET_PATH = "./ucf24/"  # dataset directory (needs that '/')
NUM_CLASSES = len(CLASSES) + 1  # +1 'background' class
BATCH_SIZE = 1
CUTOFF = 20
SSD_DIM = 300  # input size for SSD
NUM_WORKERS = 0  # number of workers used in dataloading
MEANS = (104, 117, 123)  # 'only support voc now'
CONF_THRESH = 0.01  # confidence threshold for evaluation
NMS_THRESH = 0.45  # non-maxima suppression threshold
TOP_K = 20  # top k confidence scores, for non-maxima suppression


if CUDA and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect_actions(net, dataset):
    """ Test a SSD network on an Action image database. """

    val_data_loader = data.DataLoader(dataset,
                                      BATCH_SIZE,
                                      num_workers=NUM_WORKERS,
                                      shuffle=False,
                                      collate_fn=detection_collate,
                                      pin_memory=True)
    if CUDA:
        torch.cuda.synchronize()

    num_batches = len(val_data_loader)
    print('Number of images: ', len(dataset),
          '\nNumber of batches: ', num_batches)

    detections_boxes = [[] for _ in range(NUM_CLASSES)]
    batch_iterator = None
    with torch.no_grad():
        # iterate over batches
        for batch_ind in range(len(val_data_loader)):
            if not batch_iterator:
                batch_iterator = iter(val_data_loader)
            if CUDA:
                torch.cuda.synchronize()

            print('- Batch: ', batch_ind, '/', num_batches)

            images, targets, img_indexs = next(batch_iterator)
            height, width = images.size(2), images.size(3)

            if CUDA:
                images = images.cuda()
            output = net(images)

            loc_data = output[0]
            conf_preds = output[1]
            prior_data = output[2]

            # iterate over samples in a batch
            for b in range(BATCH_SIZE):
                # print('-- Sample: ', b)
                decoded_boxes = decode(loc_data[b].data,
                                       prior_data.data,
                                       v2['variance']).clone()
                conf_scores = net.softmax(conf_preds[b]).data.clone()

                # iterate over all classes
                for cl_ind in range(1, NUM_CLASSES):
                    class_scores = conf_scores[:, cl_ind].squeeze()
                    conf_mask = class_scores.gt(CONF_THRESH)
                    class_scores = class_scores[conf_mask].squeeze()
                    if class_scores.dim() == 0 or class_scores.nelement() == 0:
                        detections_boxes[cl_ind - 1].append(np.asarray([]))
                        continue

                    boxes = decoded_boxes.clone()
                    l_mask = conf_mask.unsqueeze(1).expand_as(boxes)
                    boxes = boxes[l_mask].view(-1, 4)

                    # indices of highest scoring and non-overlapping
                    # boxes per class, after nms
                    ids, counts = nms(boxes,
                                      class_scores,
                                      NMS_THRESH,
                                      TOP_K)
                    class_scores = class_scores[ids[:counts]].cpu().numpy()
                    boxes = boxes[ids[:counts]].cpu().numpy()
                    boxes[:, 0] *= width
                    boxes[:, 2] *= width
                    boxes[:, 1] *= height
                    boxes[:, 3] *= height

                    for ik in range(boxes.shape[0]):
                        boxes[ik, 0] = max(0, boxes[ik, 0])
                        boxes[ik, 2] = min(width, boxes[ik, 2])
                        boxes[ik, 1] = max(0, boxes[ik, 1])
                        boxes[ik, 3] = min(height, boxes[ik, 3])

                    # append (num_dets) * (4 + 1) size array, so that
                    # class_detections will be of shape:
                    # (classes) * (samples) * (# dets. in sample for class) * (5)
                    class_detections = np.hstack((
                        boxes,
                        class_scores[:, np.newaxis])
                    ).astype(np.float32, copy=True)
                    detections_boxes[cl_ind - 1].append(class_detections)

            if batch_ind == CUTOFF:
                return detections_boxes

    return detections_boxes


def main():
    # Load pre-trained model
    net = build_ssd(SSD_DIM, NUM_CLASSES)  # initialize SSD
    if CUDA:
        net.load_state_dict(torch.load(BASENET))
    else:
        net.load_state_dict(torch.load(BASENET,
                                       map_location=torch.device('cpu')))
    summary(net)
    net.eval()
    if CUDA:
        net = net.cuda()
        cudnn.benchmark = True
    print('=== Finished loading model!')

    # Load dataset
    dataset = UCF24Detection(DATASET_PATH,
                             'test',  # use the test split list
                             BaseTransform(SSD_DIM, MEANS),
                             AnnotationTransform(),
                             input_type="rgb",
                             full_test=True)
    if CUDA:
        torch.cuda.synchronize()
    print('=== Finished loading dataset!')

    # Generate per-frame detections
    tt0 = time.perf_counter()
    detections_boxes = detect_actions(net, dataset)
    print('=== detection_boxes.shape: ',
          len(detections_boxes),
          '*',
          len(detections_boxes[0]),
          '* N *',
          len(detections_boxes[0][0][0]))

    if CUDA:
        torch.cuda.synchronize()
    print('Complete set time: {:0.2f}'.format(time.perf_counter() - tt0))


if __name__ == '__main__':
    main()
