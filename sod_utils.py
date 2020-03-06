import torch
import sklearn.metrics as m
import matplotlib.pyplot as plt
import numpy as np
import cv2

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def MAE_measure(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)


def F_beta_measure(y_true, y_pred, beta=0.5477):
    return m.fbeta_score(y_true, y_pred, beta)


def relaxed_F_beta_measure(y_true, y_pred, beta=0.5477):
    y_pred_th = (y_pred > 0.5).astype(float)
    y_true_th = (y_true > 0.5).astype(float)

    kernel = np.ones((3, 3), np.float)
    y_pred_th_erode = (cv2.erode(y_pred_th, kernel, iterations=1)).astype(int)
    y_true_th_erode = (cv2.erode(y_true_th, kernel, iterations=1)).astype(int)

    xor_y_pred = (y_pred_th).astype(int) ^ y_pred_th_erode
    xor_y_true = (y_true_th).astype(int) ^ y_true_th_erode

    return m.fbeta_score(xor_y_true.flatten(), xor_y_pred.flatten(), beta)


def calc_eval_measures(hr_gt, hr_out):
    MAE = MAE_measure(hr_gt, hr_out)
    F_beta = F_beta_measure((hr_gt > 0.5).astype(int).flatten(), (hr_out > 0.5).astype(int).flatten())
    relax_F_beta = relaxed_F_beta_measure(hr_gt, hr_out)

    return MAE, F_beta, relax_F_beta


def plt_data(data_train, data_test, titleStr, save_fig=False, save_dir=''):
    fig1 = plt.figure()
    plt.xlabel('Epoch #')
    plt.plot(data_train, color='b', label='Train')
    titleStr_train = titleStr + '_Train'
    plt.title(titleStr_train)
    plt.legend(loc='upper right')
    if save_fig:
        fig1.savefig(save_dir + titleStr_train + '.png', dpi=100)

    fig2 = plt.figure()
    plt.xlabel('Epoch #')
    plt.plot(data_test, color='r', label='Test')
    titleStr_test = titleStr + '_Test'
    plt.title(titleStr_test)
    plt.legend(loc='upper right')
    if save_fig:
        fig2.savefig(save_dir + titleStr_test + '.png', dpi=100)

    plt.close('all')
