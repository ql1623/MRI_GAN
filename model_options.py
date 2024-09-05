import torch
import argparse
import os

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Model Options")
        self._initialize()

    def _initialize(self):
        self.parser.add_argument('--data_png_dir', type=str, default="/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold")
        # self.parser.add_argument('--data_png_dir', type=str, default="/data/")
        self.parser.add_argument('--device', type=str, default="cpu")
        self.parser.add_argument('--gpu_ids', type=str, default="0")
        
        self.parser.add_argument('--input_modalities', '--input_mods', type=str, default="t1_t2_flair")
        self.parser.add_argument('--input_comb_seed', '--comb_seed', type=str, default=None) # should only have string of 3 numbers 0,1,2 in diff comb
        self.parser.add_argument('--fold', type=int, default=0)  
        
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--num_workers', type=int, default=4)
        self.parser.add_argument('--num_epochs', type=int, default=150)
        self.parser.add_argument('--num_features', type=int, default=32)
        self.parser.add_argument('--num_seg_features', type=int, default=16)
        
        self.parser.add_argument('--learning_rate', '--lr', type=float, default=2e-4)
        self.parser.add_argument('--lr_start_epoch', type=int, default=50)
        self.parser.add_argument('--lr_decay', type=float, default=0.95)
        
        self.parser.add_argument('--lambda_gan_l1', type=float, default=10.0) # was 25, maybe try 10 or 1 not sure, was 10?
        self.parser.add_argument('--lambda_gan_bce', type=float, default=10.0) # was 10, maybe try 1?
        self.parser.add_argument('--lambda_recon_a', type=float, default=20.0) # was 1, maybe try 10?
        self.parser.add_argument('--lambda_recon_b', type=float, default=20.0) # was 1, maybe try 10?
        self.parser.add_argument('--lambda_recon_c', type=float, default=20.0) # was 1, maybe try 10?
        self.parser.add_argument('--lambda_gdl', type=float, default=0.1) # was 2, maybe try 0.1?
        self.parser.add_argument('--lambda_seg_bce', type=float, default=10.0) # was 5, maybe try 10?
        self.parser.add_argument('--lambda_seg_dice', type=float, default=5.0) # was 10, maybe try 10?
        
        self.parser.add_argument('--b1', type=float, default=0.5)
        self.parser.add_argument('--b2', type=float, default=0.999)
        
        self.parser.add_argument('--load_model', type=bool, default=False)
        self.parser.add_argument('--save_model', type=bool, default=True)
        
        self.parser.add_argument('--log_interval', type=int, default=5)
        self.parser.add_argument('--checkpoint_interval', type=int, default=5)
        self.parser.add_argument('--save_results_dir_name', '--save_dir_name', type=str, default="unnamed_model") # save name for checkpoint dir
        # self.parser.add_argument('--save_checkpoint_dir', type=str, default="/rds/general/user/ql1623/home/datasetGAN/checkpoints")
        self.parser.add_argument('--save_checkpoint_dir', type=str, default="checkpoints")
        self.parser.add_argument('--save_results_dir', type=str, default="results")

        ### arguments for testing only 
        self.parser.add_argument('--load_results_dir_name', '--load_dir_name', type=str, default="unnamed_model") # for test.py
        self.parser.add_argument('--load_epoch', type=int, default=150) # for test.py
        
    def parse(self):
        self.options = self.parser.parse_args()
        self.options.gpu_ids = list(map(int, self.options.gpu_ids.split(',')))

        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and self.options.gpu_ids == "0":
            self.options.gpu_ids = list(range(torch.cuda.device_count()))

        if self.options.input_comb_seed is not None:
            self.options.input_comb_seed = tuple(map(int, self.options.input_comb_seed.split(',')))

        self.uppercased_options = argparse.Namespace()
        for attr, value in vars(self.options).items():
            setattr(self.uppercased_options, attr.upper(), value)
        return self.uppercased_options

if __name__ == "__main__":
    pass