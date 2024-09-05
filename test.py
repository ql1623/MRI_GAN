import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MRI_dataset

from utils import *

from dataset import MRI_dataset  

from model_options import Options
        
def test():
    parser = Options()
    options = parser.parse()
    options.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # define model
    num_encoders = 2
    if num_encoders == 2:
        import models.network_2_encode as models 
    elif num_encoders == 3:
       import models.network_3_encode as models 
    else:
        raise Exception("Number of encoder defined not within range")
    num_features = options.NUM_FEATURES
    seg_num_features = options.NUM_SEG_FEATURES
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features, pre_out_channels=seg_num_features)
    seg = models.SegmentationNetwork(input_ngf=seg_num_features, output_channels=1)
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen, options.GPU_IDS)
        seg = nn.DataParallel(seg, options.GPU_IDS)
    
    gen.to(options.DEVICE)
    seg.to(options.DEVICE)
    
    # load trained model
    epoch_to_load = options.LOAD_EPOCH
    _, gen = load_checkpoint_v2(gen, None, None, options.SAVE_CHECKPOINT_DIR, options.LOAD_RESULTS_DIR_NAME, epoch_to_load, model_type="gen", model_or_opt="model")
    _, seg = load_checkpoint_v2(seg, None, None, options.SAVE_CHECKPOINT_DIR, options.LOAD_RESULTS_DIR_NAME, epoch_to_load, model_type="seg", model_or_opt="model")
    
    # dataset
    test_dataset = MRI_dataset(options.INPUT_MODALITIES, options.DATA_PNG_DIR, transform=True, train=False, fold=options.FOLD, input_comb=options.INPUT_COMB_SEED) 
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=options.NUM_WORKERS)
    print(f"FOLD {options.FOLD}")
    print("-------------------------------")

    
    for index, images_labels in enumerate(test_loader):
        image_A, image_B, real_target_C, real_seg, in_out_ohe = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4]
        input_1_labels, input_2_labels, target_labels = in_out_ohe[:,0,:].to(options.DEVICE), in_out_ohe[:,1,:].to(options.DEVICE), in_out_ohe[:,2,:].to(options.DEVICE)
        image_A, image_B, real_target_C, real_seg = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), real_seg.to(options.DEVICE)
            
        x_concat = torch.cat((image_A, image_B), dim=1)
        
        with torch.no_grad():
            # generate
            pred_target, _, _, fusion_features= gen(x_concat, input_1_labels, input_2_labels, target_labels)
            # segment
            pred_seg = seg(fusion_features)
            avg_ssim, avg_psnr, avg_nmse, avg_dice, avg_local_ssim, avg_local_psnr, avg_local_nmse = evaluate_images(pred_target, real_target_C, pred_seg, real_seg) 
                
            print("[" + f"Batch {index+1}: NMSE: {avg_nmse:.6f} | SSIM: {avg_ssim:.6f} | PSNR: {avg_psnr:.6f} | DICE: {avg_dice:.6f} | Local_NMSE: {avg_local_nmse:.6f} | Local_SSIM: {avg_local_ssim:.6f} | Local_PSNR: {avg_local_psnr:.6f} " + "]")
              

if __name__ == "__main__":
    test()