import torch
import numpy as np
import os
from loss import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

        
def set_require_grad(model, require_grad=False):
    """ to set if model's weights are to be updated / frozen
    Parameters:
        model (nn.Module): the model to set whether gradient calculation is needed
        require_grad (bool): if gradient is needed
    """
    for param in model.parameters():
        param.requires_grad = require_grad

# schedule learning rate
def lambda_lr(epoch, lr_start_epoch, num_epochs):    
    if epoch < lr_start_epoch:
            return 1.0
    else:
        return max(0.0, 1.0 - float(epoch - lr_start_epoch) / float(num_epochs - lr_start_epoch))


def save_checkpoint_v2(epoch, model, optimizer, scheduler, checkpoint_dir, dir_name, save_filename):
    save_dir = os.path.join(checkpoint_dir, "chkpt_" + dir_name)
    save_path = os.path.join(save_dir, save_filename)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)

def load_checkpoint_v2(model, optimizer, scheduler, checkpoint_dir, dir_name, epoch_num, model_type="gen", model_or_opt="model"):
    """ epoch_num (int): the epoch number to load
        model_type (str, optional): use one of "gen", "disc", or "seg". Defaults to "gen".
        model_or_opt (str, optional): use one of "model", "opt", or "both". Defaults to "model".

    Returns:
        if model_or_opt == "opt", return checkpoint['epoch'], optimizer, scheduler 
        if model_or_opt == "model", return checkpoint['epoch'], model 
        if model_or_opt == "both", return checkpoint['epoch'], model, optimizer, scheduler
    """
    save_dir = os.path.join(checkpoint_dir, "chkpt_" + dir_name)
    if model_type=="gen":
        save_path = os.path.join(save_dir, f"{epoch_num}_net_G.pth")
        print("Loaded gen from: ", save_path)
    elif model_type=="disc":
        save_path = os.path.join(save_dir, f"{epoch_num}_net_D.pth")
        print("Loaded disc from: ", save_path)
    elif model_type=="seg":
        save_path = os.path.join(save_dir, f"{epoch_num}_net_S.pth")
        print("Loaded seg from: ", save_path)
    else:
        raise Exception("model_type defined in wrong format")
    if not os.path.exists(save_path):
        print("No such checkpoint is in this directory")
         
    # checkpoint = torch.load(save_path, map_location=config.DEVICE)
    checkpoint = torch.load(save_path)
    if model_or_opt == "opt":
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint['epoch'], optimizer, scheduler 
    elif model_or_opt == "model":
        model.load_state_dict(checkpoint["state_dict"])
        return checkpoint['epoch'], model 
    elif model_or_opt == "both":
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint['epoch'], model, optimizer, scheduler
    else:
        raise Exception("Variable 'model_or_opt' defined in wrong format")


"""Test Utils used in testing only"""

def normalize_image(image):
    """Normalize the image to the range [-1, 1]."""
    min_val = image.min()
    max_val = image.max()
    normalized_image = 2 * (image - min_val) / (max_val - min_val) - 1
    return normalized_image.float()

    

def dice_coeff_np(seg_fake, seg_target, epsilon=1e-6):
    intersection = 2 * (seg_fake * seg_target).sum(axis=(-1,-2))
    union = (seg_fake).sum(axis=(-1,-2)) + (seg_target).sum(axis=(-1,-2))
    union = np.where(union == 0, intersection, union)
    dice_coeff = (intersection + epsilon) / (union + epsilon)
    return dice_coeff


def evaluate_images(pred_images, real_images, pred_seg, real_seg): 

    real_seg_binary = real_seg.float()
    pred_seg_binary = (torch.sigmoid(pred_seg) > 0.5).float()
    dice_score = dice_coeff(pred_seg_binary, real_seg_binary, reduce_batch_first=False)
    
    pred_image = torch.squeeze(pred_images).cpu().numpy()
    real_image = torch.squeeze(real_images).cpu().numpy()
    real_seg_mask = torch.squeeze(real_seg_binary).bool().cpu().numpy()
    
    # rescale from [-1,1] to [0,2] to [0,1] 
    pred_image = (pred_image + 1) / 2
    real_image = (real_image + 1) / 2
    
    dr = np.max([pred_image.max(), real_image.max()]) - np.min([pred_image.min(), real_image.min()])
    avg_ssim = ssim(real_image, pred_image, data_range=dr)
    avg_psnr = psnr(real_image, pred_image, data_range=dr)
    avg_mse = mse(real_image, pred_image)

    if real_seg_mask.any():
        y_indices, x_indices = np.where(real_seg_mask)
        min_y, max_y = y_indices.min(), y_indices.max()
        min_x, max_x = x_indices.min(), x_indices.max()
        
        min_y = max(min_y, 0)
        max_y = min(max_y, pred_image.shape[0] - 1)
        min_x = max(min_x, 0)
        max_x = min(max_x, pred_image.shape[1] - 1)
        
        pred_image_cropped = pred_image[min_y:max_y + 1, min_x:max_x + 1]
        real_image_cropped = real_image[min_y:max_y + 1, min_x:max_x + 1]

        if pred_image_cropped.size > 0 and real_image_cropped.size > 0:
            dr_local = np.ptp([pred_image_cropped, real_image_cropped])
            avg_local_ssim = ssim(real_image_cropped, pred_image_cropped, data_range=dr_local)
            avg_local_psnr = psnr(real_image_cropped, pred_image_cropped, data_range=dr_local)
            avg_local_mse = mse(pred_image_cropped, real_image_cropped)
    else:
        avg_local_ssim = 0
        avg_local_psnr = 0
        avg_local_mse = 0

    avg_dice = dice_score.item()
    return avg_ssim, avg_psnr, avg_mse, avg_dice, avg_local_ssim, avg_local_psnr, avg_local_mse

if __name__ == "__main__":
    pass