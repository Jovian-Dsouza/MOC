from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from utils.utils import _tranpose_and_gather_feature
import torch.nn.functional as F


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        output = self.model(batch['input'])[0]
        loss, loss_stats = self.loss(output, batch)
        return output, loss, loss_stats


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(torch.nn.Module):
    '''torch.nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, index, target, index_all=None):
        pred = _tranpose_and_gather_feature(output, index, index_all=index_all)
        # pred --> b, N, 2*K
        # mask --> b, N ---> b, N, 2*K
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # print(pred.shape)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class DiceLoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.0 * intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class MOCLoss(torch.nn.Module):
    def __init__(self, loss_type='focal', hm_lambda=1, wh_lambda=1, mov_lambda=0.1):
        super().__init__()
        self.crit_hm = None
        if loss_type == 'dice':
            self.crit_hm = DiceLoss()
        elif loss_type == 'focal':
            self.crit_hm = FocalLoss()
        else:
            assert self.crit_hm is not None, "Select loss fucntion"
            
        self.crit_mov = RegL1Loss()
        self.crit_wh = RegL1Loss()

        self.hm_lambda = hm_lambda
        self.wh_lambda = wh_lambda
        self.mov_lambda = mov_lambda
        
        self.crit_dice = DiceLoss()

    def forward(self, output, batch):
        output['hm'] = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1 - 1e-4)

        hm_loss = self.crit_hm(output['hm'], batch['hm'])

        mov_loss = self.crit_mov(output['mov'], batch['mask'],
                                 batch['index'], batch['mov'])

        wh_loss = self.crit_wh(output['wh'], batch['mask'],
                               batch['index'], batch['wh'],
                               index_all=batch['index_all'])

        dice_loss = self.crit_dice(output['hm'], batch['hm'])

        loss = self.hm_lambda * hm_loss + self.wh_lambda * wh_loss + self.mov_lambda * mov_loss

        loss_stats = {
                        'loss': loss, 
                        'loss_hm': hm_loss,
                        'loss_mov': mov_loss, 
                        'loss_wh': wh_loss,
                        'dice_loss': dice_loss,
                    }
        return loss, loss_stats

