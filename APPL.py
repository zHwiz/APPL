# -*- coding: utf-8 -*-
# Author: Wenzhen Zhang
# Time: 2024/6/13 11:06


# %% setup environment
import numpy as np

import os
import cv2

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
from segment_anything.modeling.common import LayerNorm2d
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import factorizer as ft
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Type
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()



class IVUSDataset(Dataset):
    def __init__(self, data_root="./IVUSdataset/", bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "label256_training_1")
        self.img_path = join(data_root, "image256_training")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "*.png"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load png image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_256 = cv2.imread(join(self.img_path, img_name))  # (256, 256, 3)
        img_1024 = cv2.resize(img_256, (1024, 1024))
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024/255, (2, 0, 1))

        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = cv2.imread(self.gt_path_files[index], 0)  # multiple labels [0, 1,4,5...], (256,256)
        gt = cv2.resize(gt, (1024,1024),interpolation=cv2.INTER_NEAREST)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )

        gt2D = np.uint8(gt == 2)

        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y2_indices, x2_indices = np.where(gt == 2)# wall
        y1_indices, x1_indices = np.where(gt == 1)# [0 2 1] lumen
        x2_min, x2_max = np.min(x2_indices), np.max(x2_indices)
        y2_min, y2_max = np.min(y2_indices), np.max(y2_indices)

        x1_min, x1_max = np.min(x1_indices), np.max(x1_indices)
        y1_min, y1_max = np.min(y1_indices), np.max(y1_indices)


        x_left = random.randint(x2_min, x1_min)
        x_right = random.randint(x1_max, x2_max)
        y_up = random.randint(y2_min, y1_min)
        try:
            y_down = random.randint(y1_max, y2_max)
        except:
            print(img_name)
            y_down = random.randint(y2_max, y1_max)

        x1_negative = random.randint(x1_min, x1_max)


        point_coords = np.array([[x_left, 512],[x_right, 512],[512, y_up],[512, y_down],[x1_negative, 512]])
        point_labels = np.array([1,1,1,1,0])

        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(point_coords).float(),
            torch.tensor(point_labels).int(),
            img_name,
        )



# Create an elliptical mask
def create_elliptical_mask(kernel_size):
    mask = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size // 2, kernel_size // 2)
    axes = (kernel_size // 2, kernel_size // 3)  # Half axis length (horizontal axis, vertical axis)
    y, x = np.ogrid[:kernel_size, :kernel_size]
    mask[((x - center[0]) / axes[0]) ** 2 + ((y - center[1]) / axes[1]) ** 2 <= 1] = 1
    return torch.tensor(mask)

# inflation operation
def elliptical_dilation(tensor, mask):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, 1, h, w)
    padding = mask.shape[0] // 2
    dilated_tensor = F.conv2d(tensor, mask.unsqueeze(0).unsqueeze(0).float(), padding=padding)
    dilated_tensor = F.relu(dilated_tensor)
    dilated_tensor = dilated_tensor.view(b, c, h, w)
    return dilated_tensor

# corrosion operations
def elliptical_erosion(tensor, mask):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, 1, h, w)
    padding = mask.shape[0] // 2
    eroded_tensor = -F.conv2d(-tensor, mask.unsqueeze(0).unsqueeze(0).float(), padding=padding)
    eroded_tensor = F.relu(eroded_tensor)
    eroded_tensor = eroded_tensor.view(b, c, h, w)
    return eroded_tensor

# 定义闭运算操作：先膨胀后腐蚀
def elliptical_morphological_closing(tensor, kernel_size):
    device = tensor.device
    mask = create_elliptical_mask(kernel_size).to(device)  # Move the mask to the same device
    dilated_tensor = elliptical_dilation(tensor, mask)
    closed_tensor = elliptical_erosion(dilated_tensor, mask)
    return closed_tensor

class mp_RefinedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()


        self.conv_mp = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            LayerNorm2d(out_channels),
            nn.GELU()
        )


        self.multihead_attn = nn.MultiheadAttention(embed_dim=out_channels, num_heads=8)
        self.LN = LayerNorm2d(out_channels)
        self.linear_act = torch.nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.GELU()
        )

    def forward(self, original_embed, mp_embed_pre):
        # Morphological closure operation
        image_embedding_mp = elliptical_morphological_closing(mp_embed_pre, kernel_size=5)


        mp_feat = self.conv_mp(image_embedding_mp)


        b, c, h, w = mp_feat.shape
        mp_feat = mp_feat.view(b, c, h * w).permute(2, 0, 1)  # (h * w, b, c)

        A_feat = original_embed.view(b, c, h * w).permute(2, 0, 1)  # (h * w, b, c)

        attn_output, _ = self.multihead_attn(A_feat, mp_feat, mp_feat)

        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)

        attn_output = self.LN(attn_output)

        attn_output = self.linear_act(attn_output.permute(0, 2, 3, 1))
        image_embedding = attn_output.permute(0, 3, 1, 2)

        return image_embedding, image_embedding_mp



# %% set up model
class APPL(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        self.prompt_linear = torch.nn.Sequential(
            nn.Linear(in_features=256, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256)
        )


        self.factorizer_encoder = ft.FactorizerStage(
            in_channels=768,
            out_channels=768,
            spatial_size=(64, 64),
            depth=2,
            nmf=(
                ft.FactorizerSubblock,
                {
                    "tensorize": (ft.SWMatricize, {"head_dim": 8, "patch_size": 8}),
                    "act": nn.ReLU,
                    "factorize": ft.NMF,
                    "rank": 1,
                    "num_iters": 5,
                    "init": "uniform",
                    "solver": "mu",
                    "dropout": 0.1,
                },
            ),
            mlp=(ft.MLP, {"ratio": 2, "dropout": 0.1}),
        )

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        self.mp_refined = mp_RefinedBlock(in_channels=768, out_channels=256)


    def forward(self, image, point_coords, point_labels):


        x = self.image_encoder.patch_embed(image)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed


        for blk in self.image_encoder.blocks:
            x = blk(x)


            x = self.factorizer_encoder(x.permute(0, 3, 1, 2))
            x = x.permute(0, 2, 3, 1)


        image_embeddingA = self.image_encoder.neck(x.permute(0, 3, 1, 2))

        # 形态学闭运算增强
        image_embedding, image_embedding_mp = self.mp_refined(image_embeddingA, x.permute(0, 3, 1, 2))  # (B, 256, 64, 64)# (B, 256, 64, 64)

        with torch.no_grad():
            point_coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=image.device)
            point_labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=image.device)



        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords_torch, point_labels_torch),
            # points=None,
            boxes=None,
            masks=None,
        )


        prompt_features = self.prompt_linear(sparse_embeddings[:,:4,:])
        prompt_features = torch.cat((prompt_features, sparse_embeddings[:,-1:,:]), dim=1)



        # mask_decoder
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=prompt_features,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )


        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )


        return ori_res_masks



