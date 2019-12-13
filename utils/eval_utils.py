import torch
import numpy as np
from models import model_utils
from utils import utils
import cv2
import os

def coef(h, w, est_pred, est_gt, phi_0):
    n = est_pred.shape[0]
    size = n*h*w
    xi_gt, zeta_gt = model_utils.para_module(h, w, est_gt[:, 0], est_gt[:, 1], phi_0)
    xi_pred, zeta_pred = model_utils.para_module(h, w, est_pred[:, 0], est_pred[:, 1], phi_0)
    error_xi, error_zeta = mse(xi_pred, xi_gt), mse(zeta_pred, zeta_gt)
    error_alpha, error_beta = mse(est_pred[:, 0], est_gt[:, 0]), mse(est_pred[:, 1], est_gt[:, 1])
    error = {'alpha_error': error_alpha.cpu().data.numpy(), 'beta_error': error_beta.cpu().data.numpy(), 
            'xi_error': error_xi.cpu().data.numpy(), 'zeta_error': error_zeta.cpu().data.numpy()}
    pred = {'xi':xi_pred.data, 'zeta':zeta_pred.data}
    gt = {'xi':xi_gt.data, 'zeta':zeta_gt.data}
    return pred, gt, error

def calInitSep(args, para, est_input, ref_input, rb_gt):
    if args.grayscale:
        r_gt = rb_gt[:, :1, :, :]
        b_gt = rb_gt[:, -1:, :, :]
    else:
        r_gt = rb_gt[:, :3, :, :]
        b_gt = rb_gt[:, -3:, :, :]      

    r_est, b_est = InitSep(args, para, est_input, ref_input)
    loss = {}
    loss['init_r_loss'] = mse(r_gt, r_est).cpu().data.numpy()
    loss['init_b_loss'] = mse(b_gt, b_est).cpu().data.numpy()
    init_tensor = torch.cat([r_est, b_est], 1)
    return init_tensor, loss

def calSep(args, para, est_input, ref_input, rb_gt, rb_pred):
    if args.grayscale:
        r_gt = rb_gt[:, :1, :, :]
        b_gt = rb_gt[:, -1:, :, :]
        r_pred = rb_pred[:, :1, :, :]
        b_pred = rb_pred[:, -1:, :, :]
    else:
        r_gt = rb_gt[:, :3, :, :]
        b_gt = rb_gt[:, -3:, :, :]
        r_pred = rb_pred[:, :3, :, :]
        b_pred = rb_pred[:, -3:, :, :]
    r_est, b_est = InitSep(args, para, est_input, ref_input)
    loss = {}
    # loss['init_r_error'] = mse(r_gt, r_est).cpu().data.numpy()
    # loss['init_b_error'] = mse(b_gt, b_est).cpu().data.numpy()
    loss['r_error'] = mse(r_gt, r_pred).cpu().data.numpy()
    loss['b_error'] = mse(b_gt, b_pred).cpu().data.numpy()
    loss['r_psnr'] = psnr_tensor(r_gt, r_pred).cpu().data.numpy()
    loss['b_psnr'] = psnr_tensor(b_gt, b_pred).cpu().data.numpy()
    loss['r_ssim'] = ssim_tensor(r_gt, r_pred).cpu().data.numpy()
    loss['b_ssim'] = ssim_tensor(b_gt, b_pred).cpu().data.numpy()

    results = torch.cat([est_input, rb_gt, r_est, b_est, rb_pred], 1)
    return results, loss

def saveSep(args, input_ref, pred, i):
    save_dir = os.path.join(args.log_dir, 'sep_real_results')
    utils.makeFile(save_dir)
    r = (pred.permute(0, 2, 3, 1).cpu().data.numpy()[0, :, :, :3].clip(0, 1))
    b = (pred.permute(0, 2, 3, 1).cpu().data.numpy()[0, :, :, 3:6].clip(0, 1))
    cv2.imwrite(os.path.join(save_dir, str(i)+'_r_out.png'), (r[:, :, 0:1]*255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, str(i)+'_b_out.png'), (b[:, :, 0:1]*255).astype(np.uint8))

def InitSep(args, para, est_input, ref_input):
    if args.est_r_b:
        if args.grayscale: 
            r_est = ref_input[:, -2:-1, :, :]
            b_est = ref_input[:, -1:, :, :]
        else: 
            r_est = ref_input[:, -6:-3, :, :]
            b_est = ref_input[:, -3:, :, :]
    else:
        phi_0 = args.phi_0
        h, w = ref_input.shape[2], ref_input.shape[3]
        alpha, beta = para[:, 0], para[:, 1]
        if args.grayscale:
            unpol, pol = est_input[:, :1, :, :], est_input[:, 1:2, :, :]
        else:
            unpol, pol = est_input[:, :3, :, :], est_input[:, 3:6, :, :]
        xi, zeta = para_module(h, w, alpha, beta, phi_0)
        r_est = (2 * ((2 - xi) * pol + (zeta - 1) * unpol) / (2 * zeta - xi)).clamp(0, 1)
        b_est = (2 * (zeta * unpol - xi * pol) / (2 * zeta - xi)).clamp(0, 1)
    return r_est, b_est
    
def mse(a, b):
    return ((a-b)**2).mean()

def ssim_tensor(t1, t2):
    k1, k2 = 0.01, 0.03
    c1, c2 = k1**2, k2**2
    t1_mean= t1.mean((2, 3))[:, :, None, None]
    t2_mean= t2.mean((2, 3))[:, :, None, None]
    t1_sigma = torch.sqrt(((t1-t1_mean)**2).mean((2, 3)))
    t2_sigma = torch.sqrt(((t2-t2_mean)**2).mean((2, 3)))
    sigma_12 = ((t1-t1_mean)*(t2-t2_mean)).mean((2, 3))
    img_ssim = (2*t1_mean*t2_mean+c1)*(2*sigma_12+c2)/(t1_mean**2+t2_mean**2+c1)/(t1_sigma**2+t2_sigma**2+c2)
    ssim = img_ssim.mean()
    return ssim
    
def psnr_tensor(t1, t2):
    Mse = ((t1 - t2)**2).mean((1, 2, 3))
    img_psnr = 10 * torch.log10(1./Mse)
    psnr = img_psnr.mean()
    return psnr
