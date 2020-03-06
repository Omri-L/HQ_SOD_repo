import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torch

class DUTS_SOD_dataset(Dataset):
    """DUTSDataset."""

    def __init__(self, input_image_name_list, scale=8, input_size=(256, 256), lr_input_size=True):
        self.input_image_name_list = input_image_name_list # list of guidance images (RGB)
        self.scale = scale #resize scale
        self.input_size = input_size
        self.lr_input_size = lr_input_size # low resolution prediction back to the same input image size

    def __len__(self):
        return len(self.input_image_name_list)

    def __getitem__(self, idx):
        input_img_name = self.input_image_name_list[idx]
        hr_name = input_img_name.replace("Image", "Mask")
        hr_name = hr_name.replace(".jpg", ".png")

        # read and convert the guided image
        guide_rgb = cv2.imread(input_img_name)
        guide_rgb_orig_shape = guide_rgb.shape
        guide_rgb = np.array(Image.fromarray(guide_rgb).resize(self.input_size, Image.BICUBIC))
        guide_rgb = guide_rgb.astype('float32') / 255.0
        guide_rgb = np.transpose(guide_rgb, (2, 0, 1))

        # read and convert the high resolution saliency image
        hr_orig = cv2.imread(hr_name, cv2.IMREAD_GRAYSCALE)
        hr = np.array(Image.fromarray(hr_orig).resize(self.input_size, Image.BICUBIC))
        hr = hr.astype('float32') / 255.0
        hr = np.expand_dims(hr, 0)

        # read and convert the low resolution saliency image
        input_scaled_size = (int(self.input_size[0] / self.scale), int(self.input_size[1] / self.scale))
        lr = np.array(Image.fromarray(hr_orig).resize(input_scaled_size, Image.BICUBIC))
        if self.lr_input_size:
            lr = np.array(Image.fromarray(lr).resize(self.input_size, Image.BICUBIC))

        lr = lr.astype('float32') / 255.0
        lr = np.expand_dims(lr, 0)

        guide_rgb = torch.from_numpy(guide_rgb)
        lr = torch.from_numpy(lr)
        hr = torch.from_numpy(hr)

        sample = {'guidance': guide_rgb, 'lr': lr, 'gt': hr}
        
        return sample
    
        """
        return:
            sample:
            guidance (np.array float): H x W x 3 
            lr ((np.array float)): H x W x 1
            gt ((np.array float)): H x W x 1
            
        """