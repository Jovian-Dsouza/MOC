from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np

import shutil
import os

def setup_dir(dir):
    shutil.rmtree(dir, ignore_errors=True)
    os.mkdir(dir)

def _gather_feature(feature, index, index_all=None):
    # dim = channel = 2*K
    # feature b, h*w , c
    # index  b, N --> b, N, c
    if index_all is not None:
        index0 = index_all
    else:
        dim = feature.size(2)
        index0 = index.unsqueeze(2).expand(index.size(0), index.size(1), dim)
    feature = feature.gather(1, index0)
    # feature --> b, N, 2*K
    return feature


def _tranpose_and_gather_feature(feature, index, index_all=None):
    # b,c,h,w --> b,h,w,c
    feature = feature.permute(0, 2, 3, 1).contiguous()
    # b,h,w,c --> b,h*w,c
    feature = feature.view(feature.size(0), -1, feature.size(3))
    feature = _gather_feature(feature, index, index_all=index_all)
    # feature --> b, N, 2*K
    return feature

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep

def _topN(scores, N=40):
    batch, cat, height, width = scores.size()

    # each class, top N in h*w    [b, c, N]
    topk_scores, topk_index = torch.topk(scores.view(batch, cat, -1), N)

    topk_index = topk_index % (height * width)
    topk_ys = (topk_index / width).int().float()
    topk_xs = (topk_index % width).int().float()

    # cross class, top N    [b, N]
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), N)

    topk_classes = (topk_ind / N).int()
    topk_index = _gather_feature(topk_index.view(batch, -1, 1), topk_ind).view(batch, N)
    topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, N)
    topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, N)

    return topk_score, topk_index, topk_classes, topk_ys, topk_xs

def moc_decode(heat, wh, mov, K, N):
    '''
    returns detection (batch_size, N, 4*k+2)
    where each row is a vector of size 4*k+2 in this format
    [ (x1, y1, x2, y2) * K, score, class ]
    '''

    batch, cat, height, width = heat.size()

    # perform 'nms' on heatmaps
    heat = _nms(heat)
    scores, index, classes, ys, xs = _topN(heat, N=N)

    mov = _tranpose_and_gather_feature(mov, index)
    mov = mov.view(batch, N, 2 * K)

    mov_copy = mov.clone()
    mov_copy = mov_copy.view(batch, N, K, 2)
    index_all = torch.zeros((batch, N, K, 2))
    xs_all = xs.clone().unsqueeze(2).expand(batch, N, K)
    ys_all = ys.clone().unsqueeze(2).expand(batch, N, K)
    xs_all = xs_all + mov_copy[:, :, :, 0]
    ys_all = ys_all + mov_copy[:, :, :, 1]
    xs_all[:, :, K // 2] = xs
    ys_all[:, :, K // 2] = ys

    xs_all = xs_all.long()
    ys_all = ys_all.long()

    index_all[:, :, :, 0] = xs_all + ys_all * width
    index_all[:, :, :, 1] = xs_all + ys_all * width
    index_all[index_all < 0] = 0
    index_all[index_all > width * height - 1] = width * height - 1
    index_all = index_all.view(batch, N, K * 2).long()

    # gather wh in each location after movement
    wh = _tranpose_and_gather_feature(wh, index, index_all=index_all)
    wh = wh.view(batch, N, 2 * K)

    classes = classes.view(batch, N, 1).float()
    scores = scores.view(batch, N, 1)
    xs = xs.view(batch, N, 1)
    ys = ys.view(batch, N, 1)
    bboxes = []
    for i in range(K):
        bboxes.extend([xs + mov[..., 2 * i:2 * i + 1] - wh[..., 2 * i:2 * i + 1] / 2,
                        ys + mov[..., 2 * i + 1:2 * i + 2] - wh[..., 2 * i + 1:2 * i + 2] / 2,
                        xs + mov[..., 2 * i:2 * i + 1] + wh[..., 2 * i:2 * i + 1] / 2,
                        ys + mov[..., 2 * i + 1:2 * i + 2] + wh[..., 2 * i + 1:2 * i + 2] / 2])
    bboxes = torch.cat(bboxes, dim=2)
    detections = torch.cat([bboxes, scores, classes], dim=2)
    return detections

def post_process(detections, num_classes, K, hm_size=(72, 72), output_size=(288, 288)):
    '''
    returns predictions which a list having len = batch_size
    for each batch, we have dict containing the class label as the keys
    for each class in the dict, we have a np.array of shape (N, 4*k+1)
    containing the top N, bounding box prediction and score for each prediction
    '''
    # Post process
    width = output_size[1]
    height = output_size[0]
    output_width = hm_size[1]
    output_height = hm_size[0]

    detections = detections.detach().cpu().numpy()

    results = []
    for batch in range(detections.shape[0]):
        top_preds = {}
        for j in range((detections.shape[2] - 2) // 2):
            # tailor bbox to prevent out of bounds
            detections[batch, :, 2 * j] = np.maximum(0, np.minimum(width - 1, detections[batch, :, 2 * j] / output_width * width))
            detections[batch, :, 2 * j + 1] = np.maximum(0, np.minimum(height - 1, detections[batch, :, 2 * j + 1] / output_height * height))
        classes = detections[batch, :, -1]
        # gather bbox for each class
        for c in range(num_classes):
            inds = (classes == c)
            top_preds[c] = detections[batch, inds, :4 * K + 1].astype(np.float32)
        results.append(top_preds)
    return results