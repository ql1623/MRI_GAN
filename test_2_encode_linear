import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import *
from loss import *

from dataset import MRI_dataset  
import models.network_2_encode_linear as models

from model_options import Options
        
def test():

    parser = Options()
    options = parser.parse()
    checkpoint_dir = os.path.join(os.getcwd(), options.SAVE_CHECKPOINT_DIR)
    options.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    num_features = options.NUM_FEATURES
    seg_num_features = options.NUM_SEG_FEATURES
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features, pre_out_channels=seg_num_features)
    disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512])
    seg = models.SegmentationNetwork(input_ngf=seg_num_features, output_channels=1)
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen, options.GPU_IDS) 
        disc = nn.DataParallel(disc, options.GPU_IDS)
        seg = nn.DataParallel(seg, options.GPU_IDS)
    
    gen.to(options.DEVICE)
    disc.to(options.DEVICE)
    seg.to(options.DEVICE)
    
    # define optimiser for model
    opt_disc = optim.Adam(disc.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2)) 
    opt_gen = optim.Adam(gen.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    opt_seg = optim.Adam(seg.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    
    # learning rate decay
    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_seg = optim.lr_scheduler.LambdaLR(opt_seg, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))

    # attention hyperparam decay
    lambda_gdl_attn_func = lambda epoch: 0.01 + (10.0 - 0.01) * min(1, max(0, (epoch - 10) / (100 - 10)))

    # losses
    criterion_L1 = nn.L1Loss()    
    criterion_GAN_BCE = nn.BCEWithLogitsLoss()
    criterion_GAN_L1 = nn.L1Loss()
    criterion_GAN_GDL_ATT = GradientDifferenceLoss_w_Attention(filter_type="sobel")
    criterion_SEG_BCE = nn.BCEWithLogitsLoss()
    
    train_dataset = MRI_dataset(options.INPUT_MODALITIES, options.DATA_PNG_DIR, transform=True, train=True, fold=options.FOLD) 
    train_loader = DataLoader(train_dataset, batch_size=options.BATCH_SIZE, shuffle=True, num_workers=options.NUM_WORKERS)


    print(f"FOLD {options.FOLD}")
    print("-------------------------------")
    
    for epoch in range(options.NUM_EPOCHS):
    # for epoch in range(1):
        # count = 0
        loop = tqdm(train_loader, leave=True)
        lambda_gdl_attn = lambda_gdl_attn_func(epoch)
        for index, images_labels in enumerate(loop):
            # if count >= 5:
            #     break
            # else:
            #     count +=1
        image_A, image_B, real_target_C, real_seg, in_out_ohe = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4]
        input_1_labels, input_2_labels, target_labels = in_out_ohe[:,0,:].to(options.DEVICE), in_out_ohe[:,1,:].to(options.DEVICE), in_out_ohe[:,2,:].to(options.DEVICE)
        image_A, image_B, real_target_C, real_seg = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), real_seg.to(options.DEVICE)
            
        x_concat = torch.cat((image_A, image_B), dim=1)
        
        with torch.no_grad():
            # generate
            # pred_target, pred_image_A_recon, pred_image_B_recon, fusion_features= gen(x_concat, target_labels)
            pred_target, _, _, fusion_features= gen(x_concat, input_1_labels, input_2_labels, target_labels)
            # # segment
            pred_seg = seg(fusion_features)
            
            avg_ssim, avg_psnr, avg_nmse, avg_dice, avg_local_ssim, avg_local_psnr, avg_local_nmse, error_metrics = evaluate_images_binary_v3(pred_target, real_target_C, pred_seg, real_seg, run_id, index, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME) 
                
            print("[" + f"Batch {index+1}: NMSE: {avg_nmse:.6f} | SSIM: {avg_ssim:.6f} | PSNR: {avg_psnr:.6f} | DICE: {avg_dice:.6f} | Local_NMSE: {avg_local_nmse:.6f} | Local_SSIM: {avg_local_ssim:.6f} | Local_PSNR: {avg_local_psnr:.6f} " + "]")
            
            
if __name__ == "__main__":
    test()
    
