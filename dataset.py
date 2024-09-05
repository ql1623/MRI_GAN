import os
import numpy as np
import torch
import torchvision.transforms.functional as F
import torch.nn.functional as func
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from PIL import Image
import random

# to binarize segmentation mask
class Make_binary:
    def __call__(self, image):
        return torch.where(image != 0, torch.tensor(1.0), torch.tensor(0.0))

# getting transform params to sync image and segmentation transform
def get_transform_params():
    new_h = new_w = 286
    crop_size = 256
    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

# get transform for modality image
def get_transforms(grayscale=True, params=None, resize_method=transforms.InterpolationMode.BICUBIC, convert=True, is_train=True):
    transform_list = []
    resize_train_shape = [286, 286]
    resize_test_shape = [256, 256]
    crop_size = 256

    if is_train:
        transform_list.append(transforms.Resize(resize_train_shape, resize_method))

        if params is None:
            transform_list.append(transforms.RandomCrop(crop_size))
            transform_list.append(transforms.RandomHorizontalFlip())
        else:
            transform_list.append(transforms.Lambda(lambda img: img.crop((params['crop_pos'][0], params['crop_pos'][1], params['crop_pos'][0] + crop_size, params['crop_pos'][1] + crop_size))))
            if params['flip']:
                transform_list.append(transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)))
    else:
        transform_list.append(transforms.Resize(resize_test_shape, resize_method))
        
    if convert:
        transform_list.append(transforms.ToTensor())
        if grayscale:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

        
    return transforms.Compose(transform_list)

# get transform for segmentation image
def get_seg_transforms(params, resize_method=Image.NEAREST, seg_class="binary", is_train=True):
    transform_list = []
    resize_train_shape = [286, 286]
    resize_test_shape = [256, 256]
    crop_size = 256

    if is_train:
        transform_list = [
            transforms.Resize(resize_train_shape, resize_method),
            transforms.Lambda(lambda img: img.crop((params['crop_pos'][0], params['crop_pos'][1], params['crop_pos'][0] + crop_size, params['crop_pos'][1] + crop_size))),
            transforms.Lambda(lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) if params['flip'] else img),
        ]
    else:
        transform_list = [
            transforms.Resize(resize_test_shape, resize_method)
        ]

    transform_list.append(transforms.ToTensor())
    transform_list.append(Make_binary()) 
    
    return transforms.Compose(transform_list)

class MRI_dataset(Dataset):
    def __init__(self, input_modalities, data_png_dir, transform=True, train=True, fold=0, input_comb=None):
        self.input_modalities = input_modalities # string about modality set, e.g. "t1_t2_flair"
        self.data_png_dir = data_png_dir
        self.transform = transform
        self.train = train
        self.fold = fold
        self.input_comb = input_comb # if specified, denotes input-output modality combination
        
        if self.train:
            txt_file_path = os.path.join(data_png_dir, f'train_patient_fold_{fold}.txt')
        else:
            txt_file_path = os.path.join(data_png_dir, f'test_patient_fold_{fold}.txt')
        
        with open(txt_file_path, 'r') as file:
            patient_ids = [line.strip() for line in file.readlines()]

        modalities = self.input_modalities.split("_")
        mod_A, mod_B, mod_C = modalities[0], modalities[1], modalities[2]

        self.img_paths = {
            0: os.path.join(self.data_png_dir, mod_A),
            1: os.path.join(self.data_png_dir, mod_B),
            2: os.path.join(self.data_png_dir, mod_C)
        }
        self.seg_img_paths = os.path.join(self.data_png_dir, "seg")
        
        self.img_lists = os.listdir(self.seg_img_paths)
        
        self.valid_triplets = [
            "t1_t1ce_t2",
            "t1_t1ce_flair",
            "t1_t2_flair",
            "t1ce_t2_flair"
        ]
        self.in_out_combinations = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 2, 0)
        ]
        
        self.all_comb_img_lists = []
        
        if self.input_comb is not None:
            for pat_id in patient_ids:
                img_list = os.listdir(os.path.join(self.seg_img_paths, pat_id))
                for img_name in img_list:
                    self.all_comb_img_lists.append((os.path.join(pat_id, img_name), self.input_comb))
        else:          
            for pat_id in patient_ids:
                img_list = os.listdir(os.path.join(self.seg_img_paths, pat_id))
                for img_name in img_list:
                    for comb in self.in_out_combinations:
                        self.all_comb_img_lists.append((os.path.join(pat_id, img_name), comb))
        
        self.total_len = len(self.all_comb_img_lists)

    def check_input_seq(self, input_seq):
        input_term = input_seq.split("_")
        
        if len(input_term) == 3:
            input_set = set(input_term)
            for triplet in self.valid_triplets:
                if input_set == set(triplet.split('_')):
                    return True
            return False

        else:
            print("Input Sequence / Modality Direction specified is in wrong format")
            return False


    def __getitem__(self, idx):
        if not self.check_input_seq(self.input_modalities):
            print("Invalid input modalities format")
            return None, None, None, None
        
        img_filepath, (img_A_ori_idx, img_B_ori_idx, img_C_ori_idx) = self.all_comb_img_lists[idx]

        pil_A = Image.open(os.path.join(self.img_paths[img_A_ori_idx], img_filepath))
        pil_B = Image.open(os.path.join(self.img_paths[img_B_ori_idx], img_filepath))
        pil_C = Image.open(os.path.join(self.img_paths[img_C_ori_idx], img_filepath))
        pil_seg = Image.open(os.path.join(self.seg_img_paths, img_filepath))
        
        if self.transform:
            transform_params = get_transform_params()

            transform = get_transforms(grayscale=True, params=transform_params, convert=True, is_train=self.train)
            seg_transform = get_seg_transforms(params=transform_params, is_train=self.train)
            
            tensor_A = transform(pil_A)
            tensor_B = transform(pil_B)
            tensor_C = transform(pil_C)
            tensor_seg = seg_transform(pil_seg)
            
        else:
            tensor_A = torch.tensor(np.array(pil_A, dtype=np.float32))
            tensor_B = torch.tensor(np.array(pil_B, dtype=np.float32))
            tensor_C = torch.tensor(np.array(pil_C, dtype=np.float32))
            tensor_seg = torch.tensor(np.array(pil_seg, dtype=np.float32))
            
        in_out_comb = torch.tensor([img_A_ori_idx, img_B_ori_idx, img_C_ori_idx])
        in_out_ohe = func.one_hot(in_out_comb, 3).float()
        
        return tensor_A, tensor_B, tensor_C, tensor_seg, in_out_ohe, 
    
    def __len__(self):
        return self.total_len
    

if __name__ == "__main__":
    # data_dir = "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold"
    data_dir = "/rds/general/user/ql1623/home/datasetGAN/data20_z_kfold"
    input_mod_str = "t1_t2_flair"
    train_data = MRI_dataset(input_mod_str, data_dir, transform=True, train=False, fold=0, input_comb=None)
    print(len(train_data))