import numpy as np
import torch
from torch.utils import data
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
from utils.create_tilling import generate_tiling


class Data1(data.Dataset):
    def __init__(self, large_image_path, large_gt_path, large_mask_path, w_size, dilation=False):
        self.image_path = large_image_path
        self.gt_path    = large_gt_path
        self.mask_path  = large_mask_path
        self.w_size     = w_size
        self.dilation = dilation
        self.image_path    = np.array(generate_tiling(self.image_path, w_size=self.w_size))
        self.gt_path       = np.array(generate_tiling(self.gt_path,    w_size=self.w_size))
        self.mask_path = np.array(generate_tiling(self.mask_path, w_size=self.w_size))
        print('Window_size: {}, Generate {} image patches, {} gt patches, and {} mask patches.'.format(
            w_size, len(self.image_path), len(self.gt_path), len(self.mask_path)))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        img    = self.image_path[index]
        labels = self.gt_path[index]
        mask= self.mask_path[index]

        img = img / 255.
        img = np.array(img, dtype=np.float32)

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        labels = labels / 255.
        labels = labels.astype(np.uint8)

        mask=mask / 255.
        mask = mask.astype(np.uint8)

        if self.dilation:
            struct1 = ndimage.generate_binary_structure(2, 2)
            labels = binary_dilation(labels, structure=struct1).astype(np.uint8)

        labels = torch.from_numpy(np.array([labels])).float()
        mask = torch.from_numpy(mask).float()

        return img, labels, mask
