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


def train():
    # pass in cmd arguments for run
    parser = Options()
    options = parser.parse()
    options.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = os.path.join(os.getcwd(), options.SAVE_CHECKPOINT_DIR)

    # define model
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
    
    # for epoch in range(options.NUM_EPOCHS):
    for epoch in range(1):
        count = 0
        loop = tqdm(train_loader, leave=True)
        lambda_gdl_attn = lambda_gdl_attn_func(epoch)
        for index, images_labels in enumerate(loop):
            if count >= 5:
                break
            else:
                count +=1
            image_A, image_B, real_target_C, real_seg, in_out_ohe = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4]
            input_1_labels, input_2_labels, target_labels = in_out_ohe[:,0,:].to(options.DEVICE), in_out_ohe[:,1,:].to(options.DEVICE), in_out_ohe[:,2,:].to(options.DEVICE)
            image_A, image_B, real_target_C, real_seg = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), real_seg.to(options.DEVICE)
            
            x_concat = torch.cat((image_A, image_B), dim=1)
            # generation
            target_fake, image_A_recon, image_B_recon, fusion_features = gen(x_concat, input_1_labels, input_2_labels, target_labels)
            
            # segmentation
            seg_target_fake = seg(fusion_features)

            # ----- backward of disc ----- 
            # -- Disc loss for fake --
            set_require_grad(disc, True)
            opt_disc.zero_grad()
            pred_disc_fake = disc(target_fake.detach(), image_A, image_B) # as dont want to backward this 
            fake_labels = torch.zeros_like(pred_disc_fake)
            loss_D_fake = criterion_GAN_BCE(pred_disc_fake, fake_labels) # D(G(x))
            
            # -- Disc loss for real --
            pred_disc_real = disc(real_target_C, image_A, image_B)
            real_labels = torch.ones_like(pred_disc_real)
            loss_D_real = criterion_GAN_BCE(pred_disc_real, real_labels) # D(x)
            
            # get both loss and backprop
            loss_D = (loss_D_fake + loss_D_real) / 2
            loss_D.backward()
            opt_disc.step()
            
            # ----- backward of seg ----- 
            # loss for segmentation
            set_require_grad(disc, False)
            opt_seg.zero_grad()
            loss_S_BCE = criterion_SEG_BCE(seg_target_fake, real_seg) # S(G(x))
            loss_S_DICE = dice_loss(torch.sigmoid(torch.squeeze(seg_target_fake, dim=1)), torch.squeeze(real_seg, dim=1).float(), multiclass=False)
            
            loss_S = options.LAMBDA_SEG_BCE * loss_S_BCE + options.LAMBDA_SEG_DICE * loss_S_DICE
            loss_S.backward(retain_graph=True)
            opt_seg.step()
            
            # ----- backward of gen ----- 
            opt_gen.zero_grad()
            
            pred_disc_fake = disc(target_fake, image_A, image_B) # D(G(x))
            
            # loss for GAN
            loss_G_BCE = criterion_GAN_BCE(pred_disc_fake, torch.ones_like(pred_disc_fake))
            loss_G_L1 = criterion_GAN_L1(target_fake, real_target_C) 
            
            # loss for reconstucting unet
            loss_G_reconA = criterion_L1(image_A_recon, image_A)
            loss_G_reconB = criterion_L1(image_B_recon, image_B)
            
            # loss for gradient difference between pred and real
            loss_G_GDL_ATTN = criterion_GAN_GDL_ATT(target_fake, real_target_C, seg_target_fake.detach())

            loss_G = (options.LAMBDA_GAN_BCE * loss_G_BCE + 
                    options.LAMBDA_GAN_L1 * loss_G_L1 + 
                    lambda_gdl_attn * loss_G_GDL_ATTN +
                    options.LAMBDA_RECON_A * loss_G_reconA + 
                    options.LAMBDA_RECON_B * loss_G_reconB 
                    )
        
            loss_G.backward()
            opt_gen.step()

            loop.set_description(f"Epoch [{epoch+1}/{options.NUM_EPOCHS}]: Batch [{index+1}/{len(train_loader)}]")
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), loss_S=loss_S.item())
            
        scheduler_disc.step()
        scheduler_gen.step()
        scheduler_seg.step()

        if options.SAVE_MODEL:
            if (epoch+1) > 100 and (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
            # if (epoch+1) == 1:
                save_checkpoint_v2(epoch, gen, opt_gen, scheduler_gen, checkpoint_dir, options.SAVE_RESULTS_DIR_NAME, f"{epoch+1}_net_G.pth")
                save_checkpoint_v2(epoch, disc, opt_disc, scheduler_disc, checkpoint_dir, options.SAVE_RESULTS_DIR_NAME, f"{epoch+1}_net_D.pth")
                save_checkpoint_v2(epoch, seg, opt_seg, scheduler_seg, checkpoint_dir, options.SAVE_RESULTS_DIR_NAME, f"{epoch+1}_net_S.pth")
            
    
if __name__ == "__main__":
    train()

        
