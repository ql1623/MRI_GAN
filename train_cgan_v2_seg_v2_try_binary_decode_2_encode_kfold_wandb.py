import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler
from torchsummary import summary

# from sklearn.model_selection import KFold
import numpy as np
import scipy.io as scio
import time
import datetime
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd

from collections import OrderedDict

import matplotlib.pyplot as plt
# from utils.config_reader import Config
# from train_options import Options
# import train_options as config
from utils.utils import *
from utils.loss import *
from utils.dice import *

import wandb

import model.cgan.generator_seg_v2_try_v2_decode_some_batch_2_encode_embed as models 
from dataset_png_v3_seg_v2_multiclass_seg_kfold_ohe_2 import MRI_dataset  

from train_options_v3 import TrainOptions

"""Have Segmentation Network separately and not inside GAN, removed other dataset version as will be using version 4 anyways""" 



def train():
    # torch.autograd.set_detect_anomaly(True)
    # config = Config("./utils/params.yaml")
    # config = Options.parse()
    # print(options.BATCH_SIZE)
    wandb.login()
    
    start_time = time.time()
    
    parser = TrainOptions()
    options = parser.parse()
    options.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project="t1_t2_flair_gan", name=options.SAVE_RESULTS_DIR_NAME, config=options)
    # print(options.GPU_IDS)

    num_features = options.NUM_FEATURES
    seg_num_features = options.NUM_SEG_FEATURES

    # num_features = 8
    # seg_num_features = 8
    # options.BATCH_SIZE = 8
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features, pre_out_channels=seg_num_features)
    # gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=num_features, pre_out_channels=seg_num_features)
    # summary(gen, (2, 128, 128))
    disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512])
    # if multiclass:
    #     seg = models.SegmentationNetwork(input_ngf=seg_num_features, output_channels=4)
    # else:
    seg = models.SegmentationNetwork(input_ngf=seg_num_features, output_channels=1)
    # seg = models.SegmentationNetwork(input_ngf=seg_num_features, output_channels=1)
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen, options.GPU_IDS) # DistributedDataParallel?
        disc = nn.DataParallel(disc, options.GPU_IDS)
        seg = nn.DataParallel(seg, options.GPU_IDS)
    
    gen.to(options.DEVICE)
    disc.to(options.DEVICE)
    seg.to(options.DEVICE)
        
    # initialize weights inside
    # gen.apply(initialize_weights)
    # disc.apply(initialize_weights)
    
    opt_disc = optim.Adam(disc.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2)) 
    opt_gen = optim.Adam(gen.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    opt_seg = optim.Adam(seg.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    
    # learning rate decay
    scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    scheduler_seg = optim.lr_scheduler.LambdaLR(opt_seg, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))

    lambda_gdl_attn_func = lambda epoch: 0.01 + (10.0 - 0.01) * min(1, max(0, (epoch - 10) / (100 - 10)))

    criterion_L1 = nn.L1Loss()    
    criterion_GAN_BCE = nn.BCEWithLogitsLoss()
    criterion_GAN_L1 = nn.L1Loss()
    # criterion_GDL = GradientDifferenceLoss_v2(filter_type="sobel")
    criterion_GAN_GDL_ATT = GradientDifferenceLoss_v2_w_Attention(filter_type="sobel", is_multiclass=False)
    # elif filter_type == "canny":
    #     criterion_GDL = GradientDifferenceLossCanny()
    #     criterion_GAN_GDL_ATT = GradientDifferenceLossCanny_w_Attention(is_multiclass=multiclass)
    # if seg_class == "binary":
    criterion_SEG_BCE = nn.BCEWithLogitsLoss()
    
    train_dataset = MRI_dataset(options.INPUT_MODALITIES, options.DATA_PNG_DIR, transform=True, train=True, fold=options.FOLD, seg_class="binary") 
    train_loader = DataLoader(train_dataset, batch_size=options.BATCH_SIZE, shuffle=True, num_workers=options.NUM_WORKERS)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=options.NUM_WORKERS)
    
    run_id = datetime.datetime.now().strftime("run_%H:%M:%S_%d/%m/%Y")
    parser.save_options(run_id, options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, train=True)
    # save_config(config, run_id, options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, train=True)

    print(f"FOLD {options.FOLD}")
    print("-------------------------------")
    
    for epoch in range(options.NUM_EPOCHS):
    # for epoch in range(15):
        # count = 0
        loop = tqdm(train_loader, leave=True)
        epoch_losses = []
        
        lambda_gdl_attn = lambda_gdl_attn_func(epoch)
        # import pdb; pdb.set_trace()
        for index, images_labels in enumerate(loop):
            # if count >=1:
            #     break
            # else:
            #     count +=1               
            # image_A, image_B, real_target_C, real_seg, in_out_comb, in_out_ohe, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4], images_labels[5], images_labels[6]
            image_A, image_B, real_target_C, real_seg, in_out_ohe = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[5]
            # import pdb; pdb.set_trace() 
            # in_out_comb = in_out_comb.to(options.DEVICE)
            # input_1_labels, input_2_labels, target_labels = in_out_to_ohe_label_v2(in_out_comb, 3)
            input_1_labels, input_2_labels, target_labels = in_out_ohe[:,0,:].to(options.DEVICE), in_out_ohe[:,1,:].to(options.DEVICE), in_out_ohe[:,2,:].to(options.DEVICE)
            # import pdb; pdb.set_trace()
            image_A, image_B, real_target_C, real_seg = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), real_seg.to(options.DEVICE)
            
            x_concat = torch.cat((image_A, image_B), dim=1)
            # generate
            # import pdb; pdb.set_trace()
            target_fake, image_A_recon, image_B_recon, fusion_features = gen(x_concat, input_1_labels, input_2_labels, target_labels)
            # target_fake, image_A_recon, image_B_recon, fusion_features, attn_map = gen(x_concat, target_labels)
            
            # segment
            seg_target_fake = seg(fusion_features)

            # ----- backward of disc ----- 
            # -- Disc loss for fake --
            set_require_grad(disc, True)
            opt_disc.zero_grad()
            pred_disc_fake = disc(target_fake.detach(), image_A, image_B) # as dont want to backward this 
            # import pdb; pdb.set_trace()
            fake_labels = torch.zeros_like(pred_disc_fake)
            fake_labels_flipped = flip_labels(fake_labels, flip_prob=0.05)
            loss_D_fake = criterion_GAN_BCE(pred_disc_fake, fake_labels_flipped) # D(G(x))
            
            # -- Disc loss for real --
            pred_disc_real = disc(real_target_C, image_A, image_B)
            real_labels = torch.ones_like(pred_disc_real)
            real_labels_flipped = flip_labels(real_labels, flip_prob=0.05)
            loss_D_real = criterion_GAN_BCE(pred_disc_real, real_labels_flipped) # D(x)
            
            # get both loss and backprop
            loss_D = (loss_D_fake + loss_D_real) / 2
            loss_D.backward()
            opt_disc.step()
            # print("disc")
            
            # ----- backward of seg ----- 
            # loss for segmentation
            set_require_grad(disc, False)
            opt_seg.zero_grad()
            loss_S_BCE = criterion_SEG_BCE(seg_target_fake, real_seg) # S(G(x))
            # import pdb; pdb.set_trace()
            # if multiclass:
            #     loss_S_DICE = dice_loss(
            #         torch.softmax(seg_target_fake, dim=1).float(),
            #         F.one_hot(real_seg, num_classes=4).permute(0, 3, 1, 2).float(), # real_seg should be [b,h,w]
            #         multiclass=True
            #     )
            # else: 
                # import pdb; pdb.set_trace()
            loss_S_DICE = dice_loss(torch.sigmoid(torch.squeeze(seg_target_fake, dim=1)), torch.squeeze(real_seg, dim=1).float(), multiclass=False)
            
            loss_S = options.LAMBDA_SEG_BCE * loss_S_BCE + options.LAMBDA_SEG_DICE * loss_S_DICE
            loss_S.backward(retain_graph=True)
            opt_seg.step()
            # print("seg")
            
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
            # print("gen")

            loop.set_description(f"Epoch [{epoch+1}/{options.NUM_EPOCHS}]: Batch [{index+1}/{len(train_loader)}]")
            loop.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), loss_S=loss_S.item())
            
            batch_loss_dict = {
                "batch": index+1,
                "G_GAN": loss_G_BCE.item(),
                "G_L1": loss_G_L1.item(),
                "G_reA": loss_G_reconA.item(),
                "G_reB": loss_G_reconB.item(),
                "G_GDL_ATTN": loss_G_GDL_ATTN.item(),
                "D_real": loss_D_real.item(),
                "D_fake": loss_D_fake.item(),
                "S_BCE": loss_S_BCE.item(),
                "S_DICE": loss_S_DICE.item(),
            }
            epoch_losses.append(batch_loss_dict)
            log_data = {k: v for k, v in batch_loss_dict.items() if k != "batch"}
            wandb.log(log_data, step=epoch + 1)        
                    
        scheduler_disc.step()
        scheduler_gen.step()
        scheduler_seg.step()

        log_loss_to_json(options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, run_id, epoch+1, epoch_losses)
        log_loss_to_txt(options.SAVE_CHECKPOINT_DIR, options.SAVE_RESULTS_DIR_NAME, run_id, epoch+1,
                        # loss_data=[loss_G_BCE, loss_G_L1, loss_G_reconA, loss_G_reconB, loss_G_GDL, loss_D_fake, loss_D_real, loss_S_BCE, loss_S_DICE], 
                        # loss_name=["G_BCE", "G_L1", "G_reA", "G_reB", "G_GDL", "D_fake", "D_real", "S_BCE", "S_DICE"]
                        loss_data=[loss_G_BCE, loss_G_L1, loss_G_reconA, loss_G_reconB, loss_G_GDL_ATTN, loss_D_fake, loss_D_real, loss_S_BCE, loss_S_DICE], 
                        loss_name=["G_BCE", "G_L1", "G_reA", "G_reB", "G_GDL_ATTN", "D_fake", "D_real", "S_BCE", "S_DICE"]
                        )

        if options.SAVE_MODEL:
            if (epoch+1) > 100 and (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
            # if (epoch+1)== 1:
                save_checkpoint_v2(epoch, gen, opt_gen, scheduler_gen, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_G.pth")
                save_checkpoint_v2(epoch, disc, opt_disc, scheduler_disc, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_D.pth")
            # if (epoch+1) > 10 and (epoch+1) % options.CHECKPOINT_INTERVAL == 0:
                save_checkpoint_v2(epoch, seg, opt_seg, scheduler_seg, checkpoint_dir=options.SAVE_CHECKPOINT_DIR, dir_name=options.SAVE_RESULTS_DIR_NAME, save_filename=f"{epoch+1}_net_S.pth")
                # print("checkpoint saved")

        target_fake_np = target_fake.detach().cpu().numpy()
        real_target_C_np = real_target_C.detach().cpu().numpy()
        # if multiclass:
        #     seg_target_fake_np = torch.argmax(torch.softmax(seg_target_fake, dim=1), dim=1).detach().cpu().numpy()
        #     real_seg_np = torch.unsqueeze(real_seg, dim=1).detach().cpu().numpy()
        #     seg_target_fake_mapped, real_seg_mapped = visualise_seg(seg_target_fake_np, real_seg_np)
        # else:
        seg_target_fake_mapped = (torch.sigmoid(seg_target_fake).round()).detach().cpu().numpy()
        real_seg_mapped = torch.unsqueeze(real_seg, dim=1).detach().cpu().numpy()
        
        # Log the first image of the batch
        wandb.log({
            "Target Fake": [wandb.Image(target_fake_np[0], caption="Target Fake")],
            "Real Target C": [wandb.Image(real_target_C_np[0], caption="Real Target C")],
            "Seg Target Fake": [wandb.Image(seg_target_fake_mapped[0], caption="Seg Target Fake")],
            "Real Seg": [wandb.Image(real_seg_mapped[0], caption="Real Seg")],
            # "Attn 1": [wandb.Image(attn_map_1[0], caption="Attn 1")],
            # "Attn 2": [wandb.Image(attn_map_2[0], caption="Attn 2")],
        })
        
    end_time = time.time()  
    
    print(f"Time taken: {end_time - start_time:.4f} seconds")
            

    
if __name__ == "__main__":
    train()

        
