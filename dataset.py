import math 
import pytorch_lightning as pl
import torch
import os
import pickle
import numpy as np

from torch.utils import data
from utils.ACT_utils import tubelet_in_out_tubes, tubelet_has_gt
from utils.gaussian_hm import gaussian_radius, draw_umich_gaussian

import torchvision.transforms.functional as F
from torchvision.io import read_image
import kornia as K

class UCFDataset(data.Dataset):
    def __init__(self,
                 root_dir, 
                 mode, # train or val
                 pkl_filename = 'UCF101v2-GT.pkl', 
                 K=7, 
                 down_ratio=4,
                 mean=[0.40789654, 0.44719302, 0.47026115],
                 std=[0.28863828, 0.27408164, 0.27809835], 
                 spatial_resolution=[192, 256], # (h, w)
                 max_objs=128,
                 max_brightness = 0.3,
                 min_contrast = 0.5,
        ):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.K = K
        self.spatial_resolution = spatial_resolution
        self.down_ratio = down_ratio
        self.mean = mean
        self.std = std
        self.max_objs = max_objs
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.spatial_detection_resolution = (spatial_resolution[0]//down_ratio,
                                                spatial_resolution[1]//down_ratio)
        # Load pickle data
        # labels, _nframes, _train_videos, _test_videos, _gttubes, _resolution
        pkl_file = os.path.join(root_dir, pkl_filename)
        with open(pkl_file, 'rb') as fid:
            pkl = pickle.load(fid, encoding='iso-8859-1')
        for k in pkl:
            setattr(self, ('_' if k != 'labels' else '') + k, pkl[k])
        
        self.num_classes = len(self.labels)

        self._indices = []
        video_list = self._train_videos[0] if mode == 'train' else self._test_videos[0]

        for v in video_list:
            vtubes = sum(self._gttubes[v].values(), [])
            self._indices += [(v, i) for i in range(1, self._nframes[v] + 2 - self.K, self.K)
                              if tubelet_in_out_tubes(vtubes, i, self.K) and tubelet_has_gt(vtubes, i, self.K)]
    
    def __len__(self):
        return len(self._indices)
    
    def imagefile(self, v, i):
        return os.path.join(self.root_dir, 'rgb-images', v, '{:0>5}.jpg'.format(i))
    
    def make_gttbox(self, frame, v, flip=False):
        '''
        v is the video name
        gt_bbox[ilabel] -> a list of numpy array, 
            with shape (K, 4)
            each one is <x1> <y1> <x2> <y2>
        # The list represents the number of instances of a particular class

        self._gttubes[v][label] = list(tubes)
        tubes = numpy array with shape (nframes, 5), 
                each col is in this format: <frame index> <x1> <y1> <x2> <y2>
        '''
        h, w = self._resolution[v]
        gt_bbox = {}
        for ilabel, tubes in self._gttubes[v].items():
            for t in tubes:
                if frame not in t[:, 0]: #frames = t[:, 0]
                    continue
                assert frame + self.K - 1 in t[:, 0]
                t = t.copy()

                if flip:
                    xmin = w - t[:, 3]
                    t[:, 3] = w - t[:, 1]
                    t[:, 1] = xmin
                
                boxes = t[(t[:, 0] >= frame) * (t[:, 0] < frame + self.K), 1:5]
                assert boxes.shape[0] == self.K

                boxes[:, 0] = (boxes[:, 0] / w) * self.spatial_detection_resolution[1] 
                boxes[:, 2] = (boxes[:, 2] / w) * self.spatial_detection_resolution[1] 
                boxes[:, 1] = (boxes[:, 1] / h) * self.spatial_detection_resolution[0] 
                boxes[:, 3] = (boxes[:, 3] / h) * self.spatial_detection_resolution[0] 

                if ilabel not in gt_bbox:
                    gt_bbox[ilabel] = []
                gt_bbox[ilabel].append(boxes)
        return gt_bbox

    def draw_ground_truths(self, gt_bbox):
        output_h = self.spatial_detection_resolution[0]
        output_w =  self.spatial_detection_resolution[1]

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, self.K * 2), dtype=np.float32)
        mov = np.zeros((self.max_objs, self.K * 2), dtype=np.float32)
        index = np.zeros((self.max_objs), dtype=np.int64)
        index_all = np.zeros((self.max_objs, self.K * 2), dtype=np.int64)
        mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = 0
        for ilabel in gt_bbox:
            for itube in range(len(gt_bbox[ilabel])):
                key = self.K // 2
                # key frame's bbox height and width （both on the feature map）
                key_h, key_w = gt_bbox[ilabel][itube][key, 3] - gt_bbox[ilabel][itube][key, 1], gt_bbox[ilabel][itube][key, 2] - gt_bbox[ilabel][itube][key, 0]
                # create gaussian heatmap
                radius = gaussian_radius((math.ceil(key_h), math.ceil(key_w)))
                radius = max(0, int(radius))

                # ground truth bbox's center in key frame
                center = np.array([(gt_bbox[ilabel][itube][key, 0] + gt_bbox[ilabel][itube][key, 2]) / 2, (gt_bbox[ilabel][itube][key, 1] + gt_bbox[ilabel][itube][key, 3]) / 2], dtype=np.float32)
                center_int = center.astype(np.int32)
                assert 0 <= center_int[0] and center_int[0] <= output_w and 0 <= center_int[1] and center_int[1] <= output_h

                # draw ground truth gaussian heatmap at each center location
                draw_umich_gaussian(hm[ilabel], center_int, radius)

                for i in range(self.K):
                    center_all = np.array([(gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2,  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2], dtype=np.float32)
                    center_all_int = center_all.astype(np.int32)
                    # wh is ground truth bbox's height and width in i_th frame
                    wh[num_objs, i * 2: i * 2 + 2] = 1. * (gt_bbox[ilabel][itube][i, 2] - gt_bbox[ilabel][itube][i, 0]), 1. * (gt_bbox[ilabel][itube][i, 3] - gt_bbox[ilabel][itube][i, 1])
                    # mov is ground truth movement from i_th frame to key frame
                    mov[num_objs, i * 2: i * 2 + 2] = (gt_bbox[ilabel][itube][i, 0] + gt_bbox[ilabel][itube][i, 2]) / 2 - \
                        center_int[0],  (gt_bbox[ilabel][itube][i, 1] + gt_bbox[ilabel][itube][i, 3]) / 2 - center_int[1]
                    # index_all are all frame's bbox center position
                    index_all[num_objs, i * 2: i * 2 + 2] = center_all_int[1] * output_w + center_all_int[0], center_all_int[1] * output_w + center_all_int[0]
                # index is key frame's boox center position
                index[num_objs] = center_int[1] * output_w + center_int[0]
                # mask indicate how many objects in this tube
                mask[num_objs] = 1
                num_objs = num_objs + 1

        return hm, wh, mov, index, index_all, mask

    def __getitem__(self, id):
        v, frame = self._indices[id]
        video = [read_image(self.imagefile(v, frame + i)) for i in range(self.K)]
        video = torch.stack(video, dim=0) #(k, 3, h, w)
        video = F.resize(video, dataset.spatial_resolution)
        video = video/255.0

        flip = False
        if self.mode == 'train':
            flip = np.random.random() > 0.5
            if flip:
                video = K.geometry.transform.hflip(video)
               
            brightness = torch.FloatTensor(1).uniform_(-self.max_brightness, self.max_brightness)
            video = K.enhance.adjust_brightness(video, brightness_factor=brightness)

            contrast = torch.FloatTensor(1).uniform_(self.min_contrast, 1.0)
            video = K.enhance.adjust_contrast(video, contrast_factor=contrast)

        video = F.normalize(video, self.mean, self.std).permute(1, 0, 2, 3)

        gt_bbox = self.make_gttbox(frame, v, flip)

        hm, wh, mov, index, index_all, mask = self.draw_ground_truths(gt_bbox)
        return {
            'video': video, #(c, k, h, w)
            'hm': hm,  #(num_classes, h/down, w/down)
            'mov': mov, #(max_objs, 2*k)
            'wh': wh, #(max_objs, 2*k)
            'mask': mask, #(max_objs)
            'index': index,  #(max_objs)
            'index_all': index_all #(max_objs, 2*k)
        }

if __name__ == '__main__':
    dataset = UCFDataset(root_dir='ucf24',
                        mode='train',
                        down_ratio=4,
                        K=7, 
                        spatial_resolution=[192, 256],
                        mean=[0.40789654, 0.44719302, 0.47026115],
                        std=[0.28863828, 0.27408164, 0.27809835], 
            )
    print(len(dataset))
