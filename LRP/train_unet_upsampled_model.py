import torch
import numpy as np
import cv2
import argparse
import time
from sod_duts_dataloader import *
from LRP.unet_upsampled_model import *
import sod_utils
from torch.utils.data import DataLoader
from torchvision import transforms

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os
import hybrid_loss as h_loss
import glob
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--train_input_images_dir',  default=r'../../datasets/DUTS/DUTS-TR/DUTS-TR-Image/',
                    help='Train input images directory. GT should be in the same path in folder DUTS-TR-Mask')
parser.add_argument('--test_input_images_dir',  default=r'../../datasets/DUTS/DUTS-TE/DUTS-TE-Image/',
                    help='Test input images directory. GT should be in the same path in folder DUTS-TE-Mask')
parser.add_argument('--force_cpu', default=False, action='store_true', help='force cpu even if cuda is available')
parser.add_argument('--b_train', default=True, action='store_true', help='train or only test')
parser.add_argument('--lr', default='0.0001', type=float, help='learning rate')
parser.add_argument('--result', default='./result_unet', help='result folder')
parser.add_argument('--epoch', default=20, type=int, help='max epoch')
parser.add_argument('--cloud', default=False, action='store_true', help='run on google cloud or not')
parser.add_argument('--dataset_frac', type=float, default=1.0, help='data set fraction (partial dataset)')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--batch_size_test', type=int, default=1, help='testing batch size')
parser.add_argument('--save_sample_test', default=True, action='store_true', help='save samples from test')
# parser.add_argument('--parameter', default=r'saved_models/Unet16_upsampled/parameter6',
# help='name of pre-trained parameters file')
parser.add_argument('--parameter', default=None, help='name of pre-trained parameters file (or None)')

opt = parser.parse_args()
print(opt)

# use cuda or cpu:
use_cuda = True if torch.cuda.is_available() else False
use_cuda = True if not opt.force_cpu and use_cuda else False
device = torch.device("cuda" if use_cuda else "cpu")

# results folder:
s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-lr_%s-batch_size_%s-s_%s_data_fraction-%s' % (opt.result, s, opt.lr,
                                                                   opt.batch_size, 8, opt.dataset_frac)
if not os.path.exists(opt.result):
    os.mkdir(opt.result)
if not os.path.exists(result_root):
    os.mkdir(result_root)

# log
logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)

# create network
net = UNet16_upsampled(1, 64).to(device)

# load pre-train weights
if opt.parameter is not None:
    net.load_state_dict(torch.load(opt.parameter))

# define an optimizer
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10000*opt.dataset_frac, gamma=0.2)

net.train()

train_input_images_name_list = glob.glob(opt.train_input_images_dir + '*.jpg')
num_samples = int(opt.dataset_frac * len(train_input_images_name_list))
train_input_images_name_list = train_input_images_name_list[:num_samples]

print("number of files in train dataset: {}".format(len(train_input_images_name_list)))

duts_dataset = DUTS_SOD_dataset(input_image_name_list=train_input_images_name_list, scale=8, lr_input_size=False)
dataloader = DataLoader(duts_dataset, batch_size=opt.batch_size, shuffle=True)

@torch.no_grad()
def test(net, epoch_num):

    test_input_images_name_list = glob.glob(opt.test_input_images_dir + '*.jpg')
    num_samples = int(opt.dataset_frac * len(test_input_images_name_list))
    test_input_images_name_list = test_input_images_name_list[:num_samples]

    print("number of files in test dataset: {}".format(len(test_input_images_name_list)))

    duts_test_dataset = DUTS_SOD_dataset(input_image_name_list=test_input_images_name_list, scale=8, lr_input_size=False)
    dataloader = DataLoader(duts_test_dataset, batch_size=opt.batch_size_test, shuffle=False)

    net.eval()
    loss = np.zeros(len(dataloader))
    MAE_test_set_in_epoch = []
    F_beta_test_set_in_epoch = []
    relaxed_F_beta_test_set_in_epoch = []

    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):

        guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)
        out, out_sig, _ = net(guidance)
        loss[idx] = h_loss.hybrid_loss_saliency(out_sig, gt)

        t.set_description('[validate] hybrid_loss: %f' % loss[:idx+1].mean())
        t.refresh()

        if idx == 0 or idx == 1 or idx == 10:
            if opt.save_sample_test:
                cv2.imwrite(result_root + "/epoch_" + str(epoch_num) + "_test_" + str(idx) + "_output_hr_sigmoid.png",
                            sod_utils.normPRED(out_sig[0, 0].detach()).cpu().numpy().astype('float32') * 255)
                cv2.imwrite(result_root + "/test_" + str(idx) + "_gt.png",
                            gt[0, 0].detach().cpu().numpy().astype('float32') * 255)
                cv2.imwrite(result_root + "/test_" + str(idx) + "_guide_rgb.png",
                            guidance[0, 0].detach().cpu().numpy().astype('float32') * 255)

        for i in range(gt.size(1)):
            MAE, F_beta, relaxed_F_beta = sod_utils.calc_eval_measures(gt[i, 0, :, :].detach().cpu().numpy(),
                                                                       out_sig[i, 0, :, :].detach().cpu().numpy())
            MAE_test_set_in_epoch.append(MAE)
            F_beta_test_set_in_epoch.append(F_beta)
            relaxed_F_beta_test_set_in_epoch.append(relaxed_F_beta)

        t.set_description('[validate epoch_num:%d] loss: %.8f, MAE max, avg: %.8f %.8f' %
                          (epoch_num + 1, loss[:idx+1].mean(), max(MAE_test_set_in_epoch),
                           sum(MAE_test_set_in_epoch) / len(MAE_test_set_in_epoch)))

    max_MAE_test_set_per_epoch = max(MAE_test_set_in_epoch)
    avg_MAE_test_set_per_epoch = sum(MAE_test_set_in_epoch) / len(MAE_test_set_in_epoch)
    max_F_beta_test_set_per_epoch = max(F_beta_test_set_in_epoch)
    avg_F_beta_test_set_per_epoch = sum(F_beta_test_set_in_epoch) / len(F_beta_test_set_in_epoch)
    max_relaxed_F_beta_test_set_per_epoch = max(relaxed_F_beta_test_set_in_epoch)
    avg_relaxed_F_beta_test_set_per_epoch = sum(relaxed_F_beta_test_set_in_epoch) / len(relaxed_F_beta_test_set_in_epoch)

    return loss, max_MAE_test_set_per_epoch, avg_MAE_test_set_per_epoch, \
        max_F_beta_test_set_per_epoch, avg_F_beta_test_set_per_epoch, \
        max_relaxed_F_beta_test_set_per_epoch, avg_relaxed_F_beta_test_set_per_epoch


if opt.b_train:

    max_epoch = opt.epoch
    train_loss_per_epoch = []
    test_loss_per_epoch = []

    max_MAE_train_set_per_epoch = []
    avg_MAE_train_set_per_epoch = []
    max_MAE_test_set_per_epoch = []
    avg_MAE_test_set_per_epoch = []

    max_F_beta_train_set_per_epoch = []
    avg_F_beta_train_set_per_epoch = []
    max_F_beta_test_set_per_epoch = []
    avg_F_beta_test_set_per_epoch = []

    max_relaxed_F_beta_train_set_per_epoch = []
    avg_relaxed_F_beta_train_set_per_epoch = []
    max_relaxed_F_beta_test_set_per_epoch = []
    avg_relaxed_F_beta_test_set_per_epoch = []

    for epoch in range(max_epoch):
        net.train()
        running_loss = 0.0
        # MAE
        MAE_train_set_in_epoch = []
        max_MAE_train_set_in_epoch = 0
        avg_MAE_train_set_in_epoch = 0
        MAE_test_set_in_epoch = []
        max_MAE_test_set_in_epoch = 0
        avg_MAE_test_set_in_epoch = 0
        # F_beta
        F_beta_train_set_in_epoch = []
        max_F_beta_train_set_in_epoch = 0
        avg_F_beta_train_set_in_epoch = 0
        F_beta_test_set_in_epoch = []
        max_F_beta_test_set_in_epoch = 0
        avg_F_beta_test_set_in_epoch = 0
        # Relaxed F_beta
        relaxed_F_beta_train_set_in_epoch = []
        max_relaxed_F_beta_train_set_in_epoch = 0
        avg_relaxed_F_beta_train_set_in_epoch = 0
        relaxed_F_beta_test_set_in_epoch = []
        max_relaxed_F_beta_test_set_in_epoch = 0
        avg_relaxed_F_beta_test_set_in_epoch = 0

        t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
        for idx, data in enumerate(t):
            optimizer.zero_grad()
            scheduler.step()
            guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)

            out, out_sig, _ = net(guidance)

            loss = h_loss.hybrid_loss_saliency(out_sig, gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()

            for i in range(gt.size(1)):
                MAE, F_beta, relaxed_F_beta = sod_utils.calc_eval_measures(gt[i, 0, :, :].detach().cpu().numpy(),
                                                                           out_sig[i, 0, :, :].detach().cpu().numpy())
                MAE_train_set_in_epoch.append(MAE)
                F_beta_train_set_in_epoch.append(F_beta)
                relaxed_F_beta_train_set_in_epoch.append(relaxed_F_beta)

            if idx % int(len(dataloader)*0.1) == 0:
                loss_for_log = running_loss / (idx+1)
                t.set_description('[train epoch:%d] loss: %.8f, MAE max, avg: %.8f %.8f' %
                                      (epoch+1, loss_for_log, max(MAE_train_set_in_epoch),
                                       sum(MAE_train_set_in_epoch) / len(MAE_train_set_in_epoch)))
                t.refresh()

        max_MAE_train_set_in_epoch = max(MAE_train_set_in_epoch)
        avg_MAE_train_set_in_epoch = sum(MAE_train_set_in_epoch) / len(MAE_train_set_in_epoch)
        max_F_beta_train_set_in_epoch = max(F_beta_train_set_in_epoch)
        avg_F_beta_train_set_in_epoch = sum(F_beta_train_set_in_epoch) / len(F_beta_train_set_in_epoch)
        max_relaxed_F_beta_train_set_in_epoch = max(relaxed_F_beta_train_set_in_epoch)
        avg_relaxed_F_beta_train_set_in_epoch = sum(relaxed_F_beta_train_set_in_epoch) / len(relaxed_F_beta_train_set_in_epoch)

        time.sleep(int(60*opt.dataset_frac))
        test_loss, max_MAE_test_set_in_epoch, avg_MAE_test_set_in_epoch,\
        max_F_beta_test_set_in_epoch, avg_F_beta_test_set_in_epoch,\
            max_relaxed_F_beta_test_set_in_epoch, avg_relaxed_F_beta_test_set_in_epoch = test(net, epoch)

        mean_test_loss = test_loss.mean()
        mean_train_loss = running_loss/(idx+1)
        logging.info('epoch:%d ... train_hybrid_loss:%f ...test_hybrid_loss:%f' %
                         (epoch+1, mean_train_loss, mean_test_loss))

        train_loss_per_epoch.append(mean_train_loss)
        test_loss_per_epoch.append(mean_test_loss)
        sod_utils.plt_data(train_loss_per_epoch, test_loss_per_epoch,
                           "01_Loss", save_fig=True, save_dir=result_root + '/')

        max_MAE_train_set_per_epoch.append(max_MAE_train_set_in_epoch)
        avg_MAE_train_set_per_epoch.append(avg_MAE_train_set_in_epoch)
        max_MAE_test_set_per_epoch.append(max_MAE_test_set_in_epoch)
        avg_MAE_test_set_per_epoch.append(avg_MAE_test_set_in_epoch)
        sod_utils.plt_data(avg_MAE_train_set_per_epoch, avg_MAE_test_set_per_epoch,
                           '02_Average_MAE', save_fig=True, save_dir=result_root + '/')

        max_F_beta_train_set_per_epoch.append(max_F_beta_train_set_in_epoch)
        avg_F_beta_train_set_per_epoch.append(avg_F_beta_train_set_in_epoch)
        max_F_beta_test_set_per_epoch.append(max_F_beta_test_set_in_epoch)
        avg_F_beta_test_set_per_epoch.append(avg_F_beta_test_set_in_epoch)
        sod_utils.plt_data(max_F_beta_train_set_per_epoch, max_F_beta_test_set_per_epoch,
                           '03_Max_F_beta', save_fig=True, save_dir=result_root + '/')

        max_relaxed_F_beta_train_set_per_epoch.append(max_relaxed_F_beta_train_set_in_epoch)
        avg_relaxed_F_beta_train_set_per_epoch.append(avg_relaxed_F_beta_train_set_in_epoch)
        max_relaxed_F_beta_test_set_per_epoch.append(max_relaxed_F_beta_test_set_in_epoch)
        avg_relaxed_F_beta_test_set_per_epoch.append(avg_relaxed_F_beta_test_set_in_epoch)
        sod_utils.plt_data(avg_relaxed_F_beta_train_set_per_epoch, avg_relaxed_F_beta_test_set_per_epoch,
                           '04_Relaxed_Average_F_beta', save_fig=True, save_dir=result_root + '/')

        logging.info('epoch:%d ... MAE(max,avg): train:(%f,%f) ... test:(%f,%f)' % (epoch+1, max_MAE_train_set_in_epoch,
                                                                                    avg_MAE_train_set_in_epoch,
                                                                                    max_MAE_test_set_in_epoch,
                                                                                    avg_MAE_test_set_in_epoch))
        logging.info('epoch:%d ... F_beta(max,avg): train:(%f,%f) ... test:(%f,%f)' % (epoch+1,
                                                                                       max_F_beta_train_set_in_epoch,
                                                                                       avg_F_beta_train_set_in_epoch,
                                                                                       max_F_beta_test_set_in_epoch,
                                                                                       avg_F_beta_test_set_in_epoch))
        logging.info('epoch:%d ... Relaxed_F_beta(max,avg): train:(%f,%f) ... test:(%f,%f)' % (epoch+1,
                                                                                       max_relaxed_F_beta_train_set_in_epoch,
                                                                                       avg_relaxed_F_beta_train_set_in_epoch,
                                                                                       max_relaxed_F_beta_test_set_in_epoch,
                                                                                       avg_relaxed_F_beta_test_set_in_epoch))

        torch.save(net.state_dict(), "%s/parameter%d"%(result_root, epoch+1))
        time.sleep(int(30*opt.dataset_frac))

    sod_utils.plt_data(train_loss_per_epoch, test_loss_per_epoch,
                       "01_Loss", save_fig=True, save_dir=result_root + '/')
    sod_utils.plt_data(avg_MAE_train_set_per_epoch, avg_MAE_test_set_per_epoch,
                       '02_Average_MAE', save_fig=True, save_dir=result_root + '/')
    sod_utils.plt_data(max_F_beta_train_set_per_epoch, max_F_beta_test_set_per_epoch,
                       '03_Max_F_beta', save_fig=True, save_dir=result_root + '/')
    sod_utils.plt_data(avg_relaxed_F_beta_train_set_per_epoch, avg_relaxed_F_beta_test_set_per_epoch,
                       '04_Relaxed_Average_F_beta', save_fig=True, save_dir=result_root + '/')

    print("finish train")
else:
    test_loss, max_MAE_test_set_in_epoch, avg_MAE_test_set_in_epoch, \
    max_F_beta_test_set_in_epoch, avg_F_beta_test_set_in_epoch, \
    max_relaxed_F_beta_test_set_in_epoch, avg_relaxed_F_beta_test_set_in_epoch = test(net, 0)
    print("finish test")
