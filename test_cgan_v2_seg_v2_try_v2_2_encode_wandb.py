import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import scipy.io as scio
import time
import datetime
from tqdm import tqdm
from torch.autograd import Variable
import torch.autograd as autograd

from collections import OrderedDict

import wandb

from dataset import MRI_dataset

from utils.utils import *

import model.cgan.generator_seg_v2_try_v2_decode_some_batch_2_encode_embed as models 
from dataset_png_v3_seg_v2_multiclass_seg_kfold_ohe_2 import MRI_dataset  

from train_options_v3 import TrainOptions

"""Have Segmentation Network separately and not inside GAN, take feature layers right before out_conv of fusion
generate 1 time only, 1 pass on each of 3 optimiser, 1st pass: disc opt, seg opt, gan opt
gan and seg = different optimiser, update disc -> update seg net -> update gan
removed other dataset version as will be using version 4 anyways""" 
        
def test(load_dir_name, input_comb_seed, fold):
    # wandb.login()
    
    parser = TrainOptions()
    options = parser.parse()
    options.LOAD_RESULTS_DIR_NAME = load_dir_name
    options.INPUT_COMB_SEED = input_comb_seed
    options.FOLD = fold
    options.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # print(options.GPU_IDS)
     
    # wandb.init(project="t1_t2_flair_gan_test_2", name=options.LOAD_RESULTS_DIR_NAME, config=options)
    
    num_features = options.NUM_FEATURES
    seg_num_features = options.NUM_SEG_FEATURES
    # print(f"Model architecture is {net_layer} layers, with gan = {num_features} feat, seg = {seg_num_features} feat")
    
    # use_attn = options.USE_ATTN
    gen = models.datasetGAN(input_channels=1, output_channels=1, ngf=32, pre_out_channels=16)
    # summary(gen, (2, 128, 128))
    # disc = models.Discriminator(in_channels=1, features=[32,64,128,256,512])
    seg = models.SegmentationNetwork(input_ngf=16, output_channels=1)
    
    if torch.cuda.device_count() > 1:
        gen = nn.DataParallel(gen, options.GPU_IDS) # DistributedDataParallel?
        # disc = nn.DataParallel(disc, options.GPU_IDS)
        seg = nn.DataParallel(seg, options.GPU_IDS)
    
    gen.to(options.DEVICE)
    # disc.to(options.DEVICE)
    seg.to(options.DEVICE)
    
    # opt_gen = optim.Adam(gen.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    # # opt_disc = optim.Adam(disc.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2)) 
    # opt_seg = optim.Adam(seg.parameters(), lr=options.LEARNING_RATE, betas=(options.B1, options.B2))  
    
    # scheduler_gen = optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    # # scheduler_disc = optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))
    # scheduler_seg = optim.lr_scheduler.LambdaLR(opt_seg, lr_lambda=lambda epoch: lambda_lr(epoch, options.LR_START_EPOCH, options.NUM_EPOCHS))

    epoch_to_load = options.LOAD_EPOCH
    _, gen = load_checkpoint_v2(gen, None, None, options.SAVE_CHECKPOINT_DIR, options.LOAD_RESULTS_DIR_NAME, 150, model_type="gen", model_or_opt="model")
    # disc, opt_disc = load_checkpoint(disc, opt_disc, options.SAVE_CHECKPOINT_DIR, options.LOAD_RESULTS_DIR_NAME, 150, model_type="disc", model_or_opt="model")
    _, seg = load_checkpoint_v2(seg, None, None, options.SAVE_CHECKPOINT_DIR, options.LOAD_RESULTS_DIR_NAME, 150, model_type="seg", model_or_opt="model")
    
    test_dataset = MRI_dataset(options.INPUT_MODALITIES, options.DATA_PNG_DIR, transform=True, train=False, fold=options.FOLD, seg_class="binary", comb_seed=options.INPUT_COMB_SEED) 
    # test_loader = DataLoader(test_dataset, batch_size=options.BATCH_SIZE, shuffle=True, num_workers=options.NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=options.NUM_WORKERS)
    
    run_id = datetime.datetime.now().strftime("run_%d-%m-%Y_%H-%M-%S")
    parser.save_options(run_id, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME, train=False)
    # save_config(config, run_id, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME, train=False)
    print(f"FOLD {options.FOLD}")
    print("-------------------------------")
    
    temp_fold_results = {"SSIM": {}, "PSNR": {}, "NMSE": {}, "DICE": {}, "Local_SSIM": {}, "Local_PSNR": {}, "Local_NMSE": {}, }
    patient_error_metrics = {
        "SSIM": [],
        "PSNR": [],
        "NMSE": [],
        "DICE": [],
        "Local_SSIM": [],
        "Local_PSNR": [],
        "Local_NMSE": [],
    }
    
    for index, images_labels in enumerate(test_loader):
        image_A, image_B, real_target_C, real_seg, in_out_comb, in_out_ohe, img_id = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[4], images_labels[5], images_labels[6]
        # image_A, image_B, real_target_C, real_seg, in_out_ohe = images_labels[0], images_labels[1], images_labels[2], images_labels[3], images_labels[5]
        # import pdb; pdb.set_trace()
        # in_out_comb = in_out_comb.to(options.DEVICE)
        # # if gan_version != 6:
        # target_labels = in_out_to_ohe_label(in_out_comb, 3).float()
        input_1_labels, input_2_labels, target_labels = in_out_ohe[:,0,:].to(options.DEVICE), in_out_ohe[:,1,:].to(options.DEVICE), in_out_ohe[:,2,:].to(options.DEVICE)
        image_A, image_B, real_target_C, real_seg = image_A.to(options.DEVICE), image_B.to(options.DEVICE), real_target_C.to(options.DEVICE), real_seg.to(options.DEVICE)
            
        x_concat = torch.cat((image_A, image_B), dim=1)
        
        with torch.no_grad():
            # generate
            # pred_target, pred_image_A_recon, pred_image_B_recon, fusion_features= gen(x_concat, target_labels)
            pred_target, _, _, fusion_features= gen(x_concat, input_1_labels, input_2_labels, target_labels)
            # # segment
            pred_seg = seg(fusion_features)
            
            # import pdb; pdb.set_trace()
            # if multiclass:
            #     avg_ssim, avg_psnr, avg_nmse, avg_dice, error_metrics = evaluate_images_multiclass(pred_target, real_target_C, pred_seg, real_seg, run_id, index, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME) 
            # else:
            avg_ssim, avg_psnr, avg_nmse, avg_dice, avg_local_ssim, avg_local_psnr, avg_local_nmse, error_metrics = evaluate_images_binary_v3(pred_target, real_target_C, pred_seg, real_seg, run_id, index, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME) 
                
            print("[" + f"Batch {index+1}: NMSE: {avg_nmse:.6f} | SSIM: {avg_ssim:.6f} | PSNR: {avg_psnr:.6f} | DICE: {avg_dice:.6f} | Local_NMSE: {avg_local_nmse:.6f} | Local_SSIM: {avg_local_ssim:.6f} | Local_PSNR: {avg_local_psnr:.6f} " + "]")
            
            # avg_batch_results = {"Batch SSIM": avg_ssim, 
            #                      "Batch PSNR": avg_psnr, 
            #                      "Batch MSE": avg_nmse,
            #                      "Batch DICE": avg_dice,
            #                      "Batch Local SSIM": avg_local_ssim, 
            #                      "Batch Local PSNR": avg_local_psnr, 
            #                      "Batch Local MSE": avg_local_nmse,}
            
            # if index % 32 == 0:
                # wandb.log(avg_batch_results, step=index + 1)
            pat_id = "_".join(img_id[0].split("_")[:2])
            
            if pat_id not in temp_fold_results["SSIM"]:
                for key in temp_fold_results:
                    temp_fold_results[key][pat_id] = []
                    
            temp_fold_results["SSIM"][pat_id].append(avg_ssim)
            temp_fold_results["PSNR"][pat_id].append(avg_psnr)
            temp_fold_results["NMSE"][pat_id].append(avg_nmse)
            temp_fold_results["DICE"][pat_id].append(avg_dice)
            # if avg_local_ssim != 0:
            temp_fold_results["Local_SSIM"][pat_id].append(avg_local_ssim)
            # if avg_local_psnr != 0:
            temp_fold_results["Local_PSNR"][pat_id].append(avg_local_psnr)
            # if avg_local_nmse != 0:
            temp_fold_results["Local_NMSE"][pat_id].append(avg_local_nmse)
            
            # log gen images to wandb at end of each epoch
            # import pdb; pdb.set_trace()
            pred_target_np = pred_target.detach().cpu().numpy()
            real_target_C_np = real_target_C.detach().cpu().numpy()
            # if multiclass:
            #     pred_seg_np = torch.argmax(torch.softmax(pred_seg, dim=1), dim=1).detach().cpu().numpy()
            #     real_seg_np = torch.unsqueeze(real_seg, dim=1).detach().cpu().numpy()
            #     pred_seg_mapped, real_seg_mapped = visualise_seg(pred_seg_np, real_seg_np)
            # else:
            pred_seg_mapped = (torch.squeeze(torch.sigmoid(pred_seg).round(), dim=1)).detach().cpu().numpy()
            real_seg_mapped = real_seg.detach().cpu().numpy()
            
            # log only first image of the batch
            # if index % 32 == 0:
            #     wandb.log({
            #         "Target Fake": [wandb.Image(pred_target_np[0], caption="Target Fake")],
            #         "Real Target C": [wandb.Image(real_target_C_np[0], caption="Real Target C")],
            #         "Seg Target Fake": [wandb.Image(pred_seg_mapped[0], caption="Seg Target Fake")],
            #         "Real Seg": [wandb.Image(real_seg_mapped[0], caption="Real Seg")],
            #         # "Attn 1": [wandb.Image(attn_map_1[0], caption="Attn 1")],
            #         # "Attn 2": [wandb.Image(attn_map_2[0], caption="Attn 2")],
            #     })
            image_A_np = torch.squeeze(image_A, dim=1).detach().cpu().numpy()
            image_B_np = torch.squeeze(image_B, dim=1).detach().cpu().numpy()
            pred_target_np = np.squeeze(pred_target_np, axis=1)
            real_target_C_np = np.squeeze(real_target_C_np, axis=1)
            real_seg_mapped = np.squeeze(real_seg_mapped, axis=1)
                
        # elif dataset_version == 2: 
        # save_results(img_id, target_labels, real_images_A, real_images_B, pred_target, real_target_C, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME)
        # elif dataset_version == 3:  
        # save_results_seg(run_id, img_id, in_out_comb, options.INPUT_MODALITIES, index, image_A, image_B, pred_target, real_target_C, pred_seg, real_seg, error_metrics, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME)
        # print("results was saved")
        save_results_seg_v4(img_id, in_out_comb, options.INPUT_MODALITIES, image_A_np, image_B_np, pred_target_np, real_target_C_np, pred_seg_mapped, real_seg_mapped, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME)
    
    for pat_id, metrics in temp_fold_results["SSIM"].items():
        # For each metric, calculate the mean, ignoring zeros where required
        for metric in patient_error_metrics.keys():
            values = temp_fold_results[metric][pat_id]

            if metric in ["Local_SSIM", "Local_PSNR", "Local_NMSE"]:
                # Exclude zeros for local metrics
                values = [v for v in values if v != 0]

            if values:
                mean_value = np.mean(values)
            else:
                mean_value = 0  # Handle case where all values were zero or the list is empty

            # Append the mean value to the patient_error_metrics dictionary
            patient_error_metrics[metric].append(mean_value)
            
    output_file_name = f"pat_f{options.FOLD}_avg_batch_results.json"
    output_file_path = os.path.join(options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME + "_test", output_file_name)
    
    with open(output_file_path, 'w') as outfile:
        json.dump(patient_error_metrics, outfile, indent=4)

    print(f"Patient metrics saved to {output_file_path}")
    
    dataset_fold_results = {
            "fold": options.FOLD,
            "results": {
                "SSIM": np.mean(temp_fold_results["SSIM"][pat_id]),
                "PSNR": np.mean(temp_fold_results["PSNR"][pat_id]),
                "NMSE": np.mean(temp_fold_results["NMSE"][pat_id]), 
                "DICE": np.mean(temp_fold_results["DICE"][pat_id]), 
                "Local_SSIM": np.mean(temp_fold_results["Local_SSIM"][pat_id]),
                "Local_PSNR": np.mean(temp_fold_results["Local_PSNR"][pat_id]),
                "Local_NMSE": np.mean(temp_fold_results["Local_NMSE"][pat_id]), 
            }
        }
    
    fold_results_path = os.path.join(options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME + "_test", 'fold_results.json')
    save_fold_results(fold_results_path, dataset_fold_results)
    
    # batch_results_file = os.path.join(options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME + "_test", 'fold_avg_batch_results.json')
    # save_avg_batch_results(batch_results_file, options.FOLD, temp_fold_results)
        
    generate_html_seg_v2(run_id, options.SAVE_RESULTS_DIR, options.LOAD_RESULTS_DIR_NAME, options.INPUT_MODALITIES)
    # print("html was generated")
    

if __name__ == "__main__":
    # test()
    test("binary_2_encode_gdl_attn_sobel_f0_wandb", (1,2,0), 0)
    test("binary_2_encode_gdl_attn_sobel_f1_wandb", (1,2,0), 1)
    test("binary_2_encode_gdl_attn_sobel_f2_wandb", (1,2,0), 2)
    test("binary_2_encode_gdl_attn_sobel_f3_wandb", (1,2,0), 3)
    test("binary_2_encode_gdl_attn_sobel_f4_wandb", (1,2,0), 4)
    
    # test("binary_3_encode_gdl_attn_sobel_f0_wandb", (0,2,1), 0)
    # test("binary_3_encode_gdl_attn_sobel_f1_wandb", (0,2,1), 1)
    # test("binary_3_encode_gdl_attn_sobel_f2_wandb", (0,2,1), 2)
    # test("binary_3_encode_gdl_attn_sobel_f3_wandb", (0,2,1), 3)
    # test("binary_3_encode_gdl_attn_sobel_f4_wandb", (0,2,1), 4)
    
    test("binary_2_encode_gdl_attn_canny_f0_wandb", (1,2,0), 0)
    test("binary_2_encode_gdl_attn_canny_f1_wandb", (1,2,0), 1)
    test("binary_2_encode_gdl_attn_canny_f2_wandb", (1,2,0), 2)
    test("binary_2_encode_gdl_attn_canny_f3_wandb", (1,2,0), 3)
    test("binary_2_encode_gdl_attn_canny_f4_wandb", (1,2,0), 4)
    
    test("binary_2_encode_no_gdl_attn_f0_wandb", (1,2,0), 0)
    test("binary_2_encode_no_gdl_attn_f1_wandb", (1,2,0), 1)
    test("binary_2_encode_no_gdl_attn_f2_wandb", (1,2,0), 2)
    test("binary_2_encode_no_gdl_attn_f3_wandb", (1,2,0), 3)
    test("binary_2_encode_no_gdl_attn_f4_wandb", (1,2,0), 4)
    
    # test("binary_2_encode_gdl_attn_sobel_res_f0_wandb", (0,2,1), 0)
    # test("binary_2_encode_gdl_attn_sobel_res_f1_wandb", (0,2,1), 1)
    # test("binary_2_encode_gdl_attn_sobel_res_f2_wandb", (0,2,1), 2)
    # test("binary_2_encode_gdl_attn_sobel_res_f3_wandb", (0,2,1), 3)
    # test("binary_2_encode_gdl_attn_sobel_res_f4_wandb", (0,2,1), 4)
    
    # test("binary_2_encode_conv_gdl_attn_sobel_f0_wandb", (0,2,1), 0)
    # test("binary_2_encode_conv_gdl_attn_sobel_f1_wandb", (0,2,1), 1)
    # test("binary_2_encode_conv_gdl_attn_sobel_f2_wandb", (0,2,1), 2)
    # test("binary_2_encode_conv_gdl_attn_sobel_f3_wandb", (0,2,1), 3)
    # test("binary_2_encode_conv_gdl_attn_sobel_f4_wandb", (0,2,1), 4)
    
    # single conv, sampling, gdl attn, sobel
    # python test_cgan_v2_seg_v2_try_v2_2_encode_wandb.py --load_dir_name "binary_2_encode_gdl_attn_sobel_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"
    # python test_cgan_v2_seg_v2_try_v2_2_encode_wandb_v2.py --load_dir_name "binary_2_encode_gdl_attn_sobel_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"
    # python test_cgan_v2_seg_v2_try_v2_3_encode_wandb.py --load_dir_name "binary_3_encode_gdl_attn_sobel_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"
    # python test_cgan_v2_seg_v2_try_v2_3_encode_wandb_v2.py --load_dir_name "binary_3_encode_gdl_attn_sobel_f0_wandb" --fold 0 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"

    # single conv, sampling, gdl attn, canny
    # python test_cgan_v2_seg_v2_try_v2_2_encode_wandb.py --load_dir_name "binary_2_encode_gdl_attn_canny_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"
    # python test_cgan_v2_seg_v2_try_v2_2_encode_wandb_v2.py --load_dir_name "binary_2_encode_gdl_attn_canny_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"

    # single conv, sampling, no gdl attn
    # python test_cgan_v2_seg_v2_try_v2_2_encode_wandb.py --load_dir_name "binary_2_encode_no_gdl_attn_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"
    # python test_cgan_v2_seg_v2_try_v2_2_encode_wandb_v2.py --load_dir_name "binary_2_encode_no_gdl_attn_f0_wandb" --fold 0 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"

    # single conv, sampling, gdl attn, sobel, res in modality encoder
    # python test_cgan_v2_seg_v2_try_v2_2_encode_res_wandb.py --load_dir_name "binary_2_encode_gdl_attn_sobel_res_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"
    # python test_cgan_v2_seg_v2_try_v2_2_encode_res_wandb_v2.py --load_dir_name "binary_2_encode_gdl_attn_sobel_res_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"

    # single conv, sampling, gdl attn, sobel, embed using conv2d
    # python test_cgan_v2_seg_v2_try_v2_2_encode_conv_wandb.py --load_dir_name "binary_2_encode_conv_gdl_attn_sobel_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"
    # python test_cgan_v2_seg_v2_try_v2_2_encode_conv_wandb_v2.py --load_dir_name "binary_2_encode_conv_gdl_attn_sobel_f4_wandb" --fold 4 --seg_class "binary" --norm_type "instance" --num_features 32 --batch_size 1 --data_png_dir "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold" --load_epoch 150 --comb_seed "0,2,1"
