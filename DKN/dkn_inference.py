import numpy as np
import cv2
import argparse
from DKN.models import *
import torch
import os
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--force_cpu',  default=False, action='store_true',
                    help='force running on cpu even if cuda available')
parser.add_argument('--rgb_folder',  default=r'./images/',
                    help='rgb folder path')
parser.add_argument('--mask_folder',  default=r'./images/',
                    help='SOD mask folder path')
parser.add_argument('--image_name',  default='ILSVRC2012_test_00000181', help='name of low resolution depth image')
parser.add_argument('--k', type=int, default=3, help='size of kernel')
parser.add_argument('--d', type=int, default=15, help='size of grid area')
parser.add_argument('--scale', type=int, default=8, help='scale factor')
parser.add_argument('--parameter',
                    default=r'./saved_model/' +
                            r'20200220182901-lr_1e-05-batch_size_4-k_3-d_15-s_8_data_fraction-0.4_lres_basnet-False/' +
                            r'parameter16', help='name of parameter file')
parser.add_argument('--model',  default='FDKN', help='choose model FDKN or DKN')
parser.add_argument('--output_folder',  default='DKN_results/', help='output results folder')

opt = parser.parse_args()
print(opt)

# create output folder if not exsits:
if not os.path.exists(opt.output_folder):
    os.mkdir(opt.output_folder)

# use cuda or cpu
use_cuda = True if torch.cuda.is_available() else False
use_cuda = True if not opt.force_cpu and use_cuda else False
device = torch.device("cuda" if use_cuda else "cpu")

# select model
if opt.model == 'FDKN':
    net = FDKN(kernel_size=opt.k, filter_size=opt.d, residual=True, device=device).to(device)
elif opt.model == 'DKN':
    net = DKN(kernel_size=opt.k, filter_size=opt.d, residual=True, device=device).to(device)

# load pre-trained model
if not use_cuda:
    net.load_state_dict(torch.load(opt.parameter, map_location='cpu'))
else:
    net.load_state_dict(torch.load(opt.parameter))
net.eval()
print('parameter \"%s\" has loaded'%opt.parameter)

# input image names (assume rgb image with 'jpg' and mask image as 'png')
rgb_img_path = opt.rgb_folder + '/' + opt.image_name + '.jpg'
mask_img_path = opt.mask_folder + '/' + opt.image_name + '.png'

net_input_size = (256, 256) # constant
resized_input_size = (32, 32) # constant

# read rgb image and save it
rgb = cv2.imread(rgb_img_path)
orig_rgb_shape = rgb.shape
cv2.imwrite(opt.output_folder + '/' + opt.image_name + '_rgb.jpg', rgb)

# resize the rgb input image to 256x256, normalize by 255
rgb = np.array(Image.fromarray(rgb).resize(net_input_size, Image.BICUBIC))
rgb = rgb.astype('float32') / 255.0
rgb = np.transpose(rgb, (2, 0, 1))

# read the mask input image and save it
mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
cv2.imwrite(opt.output_folder + '/' + opt.image_name + '_mask.png', mask)

# resize the input image 256x256 (consider it as the high resolution ground truth)
hr = np.array(Image.fromarray(mask).resize(net_input_size, Image.BICUBIC))

# resize the high res input to 32x32 and back to 256x256 (consider it as the low resolution input)
lr = np.array(Image.fromarray(hr).resize(resized_input_size, Image.BICUBIC))
lr = np.array(Image.fromarray(lr).resize(net_input_size, Image.BICUBIC))
lr = lr.astype('float32')/255.0
lr = np.expand_dims(lr, 0)

# prepare inputs for network:
image = torch.from_numpy(np.expand_dims(rgb, 0)).to(device)
lr = torch.from_numpy(np.expand_dims(lr, 0)).to(device)

# call network
with torch.no_grad():
    out_img, out_sig_img = net((image, lr))

# prepare outputs:
out_img = out_img.cpu().numpy()
out_sig_img = out_sig_img.cpu().numpy()
lr = lr.cpu().numpy()

# save the output image and the resized version for comparison:
out_img_name = opt.output_folder + '/' + opt.image_name + '_out.png'
resized_img_name = opt.output_folder + '/' + opt.image_name + '_resized.png'

out_img = cv2.resize((out_img[0,0]).astype('float32') * 255, (orig_rgb_shape[1],orig_rgb_shape[0]))
cv2.imwrite(out_img_name, out_img)

resized_img = cv2.resize(lr[0, 0].astype('float32') * 255, (orig_rgb_shape[1], orig_rgb_shape[0]))
cv2.imwrite(resized_img_name, resized_img)

print("DKN inference finish")