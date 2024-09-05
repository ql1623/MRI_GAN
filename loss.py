import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian

##### losses #####

# https://github.com/omigeft/RVSC-Medical-Image-Segmentation/blob/master/utils/dice_score.py#L25
def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    intersection = 2 * (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    union = torch.where(union == 0, intersection, union)

    dice = (intersection + epsilon) / (union + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input, target, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)



class GradientDifferenceLoss_w_Attention(nn.Module):
    def __init__(self, filter_type):
        super(GradientDifferenceLoss_w_Attention, self).__init__()
        if filter_type == "sobel":
            self.filter_h = torch.tensor([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # make [1,1,3,3]
            self.filter_w = torch.tensor([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif filter_type == "prewitt":  
            self.filter_h = torch.tensor([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # make [1,1,3,3]
            self.filter_w = torch.tensor([[-1, -1, -1],[0, 0, 0],[1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

    def forward(self, pred, target, seg_output):
        attn_map = torch.sigmoid(seg_output).round()
        self.filter_h = self.filter_h.to(target.device)
        self.filter_w = self.filter_w.to(target.device)

        pred_grad_h = F.conv2d(pred, self.filter_h, padding=1)
        pred_grad_w = F.conv2d(pred, self.filter_w, padding=1)
        
        target_grad_h = F.conv2d(target, self.filter_h, padding=1)
        target_grad_w = F.conv2d(target, self.filter_w, padding=1)
        
        pred_grad_h = attn_map * (torch.abs(pred_grad_h))
        pred_grad_w = attn_map * (torch.abs(pred_grad_w))
        target_grad_h = attn_map * (torch.abs(target_grad_h))
        target_grad_w = attn_map * (torch.abs(target_grad_w))
        
        grad_diff_h = torch.mean((target_grad_h - pred_grad_h)**2)
        grad_diff_w = torch.mean((target_grad_w - pred_grad_w)**2)
        
        return grad_diff_h + grad_diff_w

    