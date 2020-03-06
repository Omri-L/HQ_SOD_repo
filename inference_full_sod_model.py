import numpy as np
import argparse
from sod_model import *
import os
from PIL import Image
import cv2
import time

# a list of input images
input_images = ['ILSVRC2012_test_00000003', 'ILSVRC2012_test_00000105', 'ILSVRC2012_test_00000307',
                'ILSVRC2012_test_00000425', 'ILSVRC2012_test_00000776', 'ILSVRC2012_test_00000896',
                'ILSVRC2012_test_00001149', 'ILSVRC2012_test_00001157', 'ILSVRC2012_test_00001306']

for input_image in input_images:

    parser = argparse.ArgumentParser()
    # general params for device:
    parser.add_argument('--force_cpu',  default=False, action='store_true',
                        help='force running on cpu even if cuda available')
    parser.add_argument('--rgb_folder',  default=r'../datasets/DUTS/DUTS-TE/DUTS-TE-Image/',
                        help='name of rgb image')
    parser.add_argument('--mask_folder',  default=r'../datasets/DUTS/DUTS-TE/DUTS-TE-Mask/',
                        help='name of low resolution depth image')
    parser.add_argument('--image_name',  default=input_image, help='name of low resolution depth image')
    parser.add_argument('--output_folder',  default='final_results/', help='output folder')
    # parser.add_argument('--pre_trained_low_res_sod_params',
    #                     default=r'./saved_models/BASNET/basnet_bsi.pth', help='path to low res sod params')
    # parser.add_argument('--pre_trained_low_res_sod_params',
    #                     default=r'./saved_models/FCN/FCN_large.pth', help='path to low res sod params')
    # parser.add_argument('--merged_model_params',
    #                     default=None, help='path to SOD_model')
    # parser.add_argument('--pre_trained_fdkn_params',
    #                     default=r'./saved_models/FDKN/20200215001849_fdkn_on_basnet_epoch12', help='path to fdkn params')
    # parser.add_argument('--pre_trained_fdkn_params',
    #                     default=r'E:\DeepLearning\sod_upsampling_master\DKN\parameter\FDKN_8x', help='path to fdkn params')
    parser.add_argument('--merged_model_params',
                        default=r'./saved_models/Unet16_upsampled_FDKN/20200222021729_unet16_up_fdkn_parameter3',
                        help='path to SOD_model (or None)')
    parser.add_argument('--pre_trained_fdkn_params',
                        default=None, help='path to fdkn params')
    parser.add_argument('--pre_trained_low_res_sod_params',
                        default=None, help='path to low res sod params')

    # params for FDKN model:
    parser.add_argument('--k', type=int, default=3, help='size of kernel')
    parser.add_argument('--d', type=int, default=15, help='size of grid area')
    parser.add_argument('--scale', type=int, default=8, help='scale factor')

    opt = parser.parse_args()
    print(opt)

    # use cuda or cpu
    use_cuda = True if torch.cuda.is_available() else False
    use_cuda = True if not opt.force_cpu and use_cuda else False
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device: {}".format(device))

    # define network
    net = SOD_model(fdkn_kernel_size=opt.k,
                    fdkn_filter_size=opt.d,
                    device=device,
                    lres_model_params=opt.pre_trained_low_res_sod_params,
                    fdkn_model_params=opt.pre_trained_fdkn_params,
                    scale=opt.scale).to(device)

    # load pre-trained weights
    if opt.merged_model_params is not None:
        if not use_cuda:
            net.load_state_dict(torch.load(opt.merged_model_params, map_location='cpu'))
        else:
            net.load_state_dict(torch.load(opt.merged_model_params))
        print('merged_model_params \"%s\" has loaded' % opt.merged_model_params)

    net.eval()

    # create folder for results
    if not os.path.exists(opt.output_folder):
        os.mkdir(opt.output_folder)

    # input image path (assumes rgb image in jpg format and mask image in png format)
    rgb_img_path = opt.rgb_folder + '/' + opt.image_name + '.jpg'
    mask_img_path = opt.mask_folder + '/' + opt.image_name + '.png'

    print("rgb_img_path: {}".format(rgb_img_path))
    print("mask_img_path: {}".format(mask_img_path))

    # read mask image and rgb image and save
    mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(opt.output_folder + '/' + opt.image_name + '_mask.png', mask)
    rgb = cv2.imread(rgb_img_path)
    cv2.imwrite(opt.output_folder + '/' + opt.image_name + '_rgb.jpg', rgb)

    input_net_size = (256, 256)
    scaled_input_size = (32, 32)

    orig_rgb_shape = rgb.shape

    # resize the rgb image to 256x256 and normalize it by 255
    rgb = np.array(Image.fromarray(rgb).resize(input_net_size, Image.BICUBIC))
    rgb = rgb.astype('float32') / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    image = torch.from_numpy(np.expand_dims(rgb, 0)).to(device)

    # call network
    with torch.no_grad():
        out, out_sig, out_lr_upsampled, out_lr_upsampled_sig, out_lr32_sig = net(image)

    # prepare outputs
    out = out.cpu().numpy()
    out_sig = out_sig.cpu().numpy()
    out_lr32_sig = out_lr32_sig.cpu().numpy()

    # output image paths
    out_img_name = opt.output_folder + '/' + opt.image_name + 'hr_out.png'
    out_sig_img_name = opt.output_folder + '/' + opt.image_name + 'hr_out_sig.png'
    out_lr_sig_img_name = opt.output_folder + '/' + opt.image_name + 'lr_out_sig.png'

    # save output images
    out_img = np.array(Image.fromarray((out[0, 0]).astype('float32') * 255).resize((orig_rgb_shape[1],
                                                                                    orig_rgb_shape[0]), Image.BICUBIC))
    cv2.imwrite(out_img_name, out_img)


    out_sig_img = np.array(Image.fromarray((out_sig[0, 0]).astype('float32') * 255).resize((orig_rgb_shape[1],
                                                                                    orig_rgb_shape[0]), Image.BICUBIC))
    cv2.imwrite(out_sig_img_name, out_sig_img)


    out_lr_sig_img = np.array(Image.fromarray((out_lr32_sig[0, 0]).astype('float32') * 255).resize((orig_rgb_shape[1],
                                                                                    orig_rgb_shape[0]), Image.BICUBIC))
    cv2.imwrite(out_lr_sig_img_name, out_lr_sig_img)

    del out, out_sig, out_lr_upsampled, out_lr_upsampled_sig, out_lr32_sig

    print("finish")

