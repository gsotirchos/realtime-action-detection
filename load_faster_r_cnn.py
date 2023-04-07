import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torchinfo import summary

from data import (CLASSES, AnnotationTransform, BaseTransform, UCF24Detection,
                  detection_collate, v2)
from layers.box_utils import decode, nms
from ssd import build_ssd

CUDA = False
BASENET = "./rgb-ssd300_ucf24_120000.pth"  # pretrained model parameter file
DATASET_PATH = "./ucf24/"  # dataset directory (needs that '/')
NUM_CLASSES = len(CLASSES) + 1  # +1 'background' class
CUTOFF = 2000
SSD_DIM = 300  # input size for SSD
NUM_WORKERS = 0  # number of workers used in dataloading
MEANS = (104, 117, 123)  # 'only support voc now'
CONF_THRESH = 0.01  # confidence threshold for evaluation
NMS_THRESH = 0.45  # non-maximum suppression threshold
TOP_K = 20  # top k confidence scores, for non-maximum suppression


if CUDA and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect_actions(net, dataset):
    """ Test a SSD network on an Action image database. """

    val_data_loader = data.DataLoader(dataset,
                                      num_workers=NUM_WORKERS,
                                      shuffle=False,
                                      collate_fn=detection_collate,
                                      pin_memory=True)
    if CUDA:
        torch.cuda.synchronize()

    num_samples = len(val_data_loader)
    print('Number of images: ', len(dataset),
          '\nNumber of batches: ', num_samples)

    detections = [[] for _ in range(NUM_CLASSES)]
    sample_itr = None
    with torch.no_grad():
        # iterate over samples
        for sample_ind in range(num_samples):
            if not sample_itr:
                sample_itr = iter(val_data_loader)
            if CUDA:
                torch.cuda.synchronize()

            print('- Sample: ', sample_ind, '/', num_samples)

            # get the sample's data
            images, targets, img_indexs = next(sample_itr)
            height, width = images.size(2), images.size(3)

            if CUDA:
                images = images.cuda()
            output = net(images)

            conf_scores, decoded_boxes = get_scores_and_boxes(output, net)

            # iterate over all classes
            for cl_ind in range(1, NUM_CLASSES):
                class_detections = get_class_detections(
                    cl_ind,
                    conf_scores,
                    decoded_boxes,
                    height, width)

                detections[cl_ind - 1].append(class_detections)

            if sample_ind == CUTOFF:
                return detections

    return detections


def get_scores_and_boxes(output, net):
    """ Retrieve the confidence scores and bounding boxes
    from the net's output. """
    # split the the output to:
    loc_data = output[0]  # loc layers' output
    conf_preds = output[1]  # confidence predictions
    prior_data = output[2]  # prior boxes

    # use the loc data to refine the prior boxes' coordinates
    decoded_boxes = decode(loc_data[0].data,
                           prior_data.data,
                           v2['variance']
                           ).clone()

    # apply softmax to the confidence predictions
    conf_scores = net.softmax(conf_preds[0]).data.clone()

    return conf_scores, decoded_boxes


def get_class_detections(cl_ind, conf_scores, decoded_boxes, height, width):
    """ Process and retrieve the per-class confidence scores
    and bounding boxes. """
    class_scores = conf_scores[:, cl_ind].squeeze()

    # filter the class scores with the confidence threshold
    conf_mask = class_scores.gt(CONF_THRESH)
    class_scores = class_scores[conf_mask].squeeze()
    if class_scores.dim() == 0 or class_scores.nelement() == 0:
        return np.asarray([])

    # filter the bounding boxes with the confidence threshold
    class_boxes = decoded_boxes.clone()
    l_mask = conf_mask.unsqueeze(1).expand_as(class_boxes)
    class_boxes = class_boxes[l_mask].view(-1, 4)

    # apply non-maximum suppression
    # indices of top k highest scoring and non-overlapping
    # boxes per class, after nms
    ids, counts = nms(class_boxes,
                      class_scores,
                      NMS_THRESH,
                      TOP_K)

    class_scores = class_scores[ids[:counts]].cpu().numpy()
    class_boxes = class_boxes[ids[:counts]].cpu().numpy()
    class_boxes[:, 0] *= width
    class_boxes[:, 2] *= width
    class_boxes[:, 1] *= height
    class_boxes[:, 3] *= height

    for ik in range(class_boxes.shape[0]):
        class_boxes[ik, 0] = max(0, class_boxes[ik, 0])
        class_boxes[ik, 2] = min(width, class_boxes[ik, 2])
        class_boxes[ik, 1] = max(0, class_boxes[ik, 1])
        class_boxes[ik, 3] = min(height, class_boxes[ik, 3])

    # append (num_dets) * (4 + 1) size array, so that
    # class_detections will be of shape:
    # (classes) * (samples) * (# dets. in sample for class) * (5)
    class_detections = np.hstack((
        class_boxes,
        class_scores[:, np.newaxis])
    ).astype(np.float32, copy=True)

    return class_detections


def setup_backbone(split='train'):

    # load pre-trained model
    net = build_ssd(SSD_DIM, NUM_CLASSES)  # initialize SSD
    if CUDA:
        net.load_state_dict(torch.load(BASENET))
    else:
        net.load_state_dict(torch.load(BASENET,
                                       map_location=torch.device('cpu')))

    # print a summary of the loaded network's architecture
    summary(net)

    net.eval()

    if CUDA:
        net = net.cuda()
        cudnn.benchmark = True
    print('=== Finished loading model!')

    # load dataset
    dataset = UCF24Detection(DATASET_PATH,
                             split,  # use the test split list
                             BaseTransform(SSD_DIM, MEANS),
                             AnnotationTransform(),
                             input_type="rgb",
                             full_test=True)
    if CUDA:
        torch.cuda.synchronize()
    print('=== Finished loading dataset!')

    return net, dataset


def main():
    # load pre-trained model
    net = build_ssd(SSD_DIM, NUM_CLASSES)  # initialize SSD
    if CUDA:
        net.load_state_dict(torch.load(BASENET))
    else:
        net.load_state_dict(torch.load(BASENET,
                                       map_location=torch.device('cpu')))

    # print a summary of the loaded network's architecture
    summary(net)

    net.eval()

    if CUDA:
        net = net.cuda()
        cudnn.benchmark = True
    print('=== Finished loading model!')

    # load dataset
    dataset = UCF24Detection(DATASET_PATH,
                             'test',  # use the test split list
                             BaseTransform(SSD_DIM, MEANS),
                             AnnotationTransform(),
                             input_type="rgb",
                             full_test=True)
    print(dataset)
    return
    if CUDA:
        torch.cuda.synchronize()
    print('=== Finished loading dataset!')

    # generate per-frame detections
    tt0 = time.perf_counter()
    detections = detect_actions(net, dataset)
    print('=== detection_boxes.shape: ',
          len(detections),
          '*',
          len(detections[0]),
          '* N *',
          len(detections[0][0][0]))

    if CUDA:
        torch.cuda.synchronize()
    print('Complete set time: {:0.2f}'.format(time.perf_counter() - tt0))


if __name__ == '__main__':
    main()
