import os
import torch
from models import model_utils
from utils import time_utils, eval_utils
import numpy as np

def get_itervals(args, split):
    if split not in ['train', 'val', 'test']:
        split = 'test'
    args_var = vars(args)
    disp_intv = args_var[split+'_disp']
    save_intv = args_var[split+'_save']
    stop_iters = args_var['max_'+split+'_iter']
    return disp_intv, save_intv, stop_iters

def test(args, split, loader, models, log, recorder):
    models[0].eval()
    models[1].eval()
    log.printWrite('---- Start %s: %d batches ----' % (split, len(loader)))
    timer = time_utils.Timer(args.time_sync)
    disp_intv, save_intv, stop_iters = get_itervals(args, split)
    with torch.no_grad():
        for i, sample in enumerate(loader):
            input_est, gt = model_utils.parseRefData(args, sample, timer, 'train')
            pred_est = models[0](input_est); timer.updateTime('Forward')
            input_ref = model_utils.getRefInput(args, pred_est, input_est)
            pred = models[1](input_ref); timer.updateTime('Forward')
            _, error = eval_utils.calSep(args, pred_est, input_est, input_ref, gt, pred); timer.updateTime('Crit')
            recorder.updateIter(split, error.keys(), error.values())
            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split':split, 'epoch':0, 'iters':iters, 'batch':len(loader), 
                       'timer':timer, 'recorder': recorder}
                log.printItersSummary(opt)
            if iters % save_intv == 0:
                img_list, nrow = prepareSave(args, pred_est, input_est, input_ref, gt, pred) 
                log.saveImgResults(img_list, 'test', 0, iters, nrow=nrow)
            timer.updateTime('Record')

            if stop_iters > 0 and iters >= stop_iters: break
            
    opt = {'split': split, 'epoch':0, 'recorder': recorder}
    log.printEpochSummary(opt)

def prepareSave(args, para, input, ref_input, rb_gt, rb_pred):
    img_list = [input, rb_gt]
    if args.show_init_sep:
        init_tensor, _ = eval_utils.calInitSep(args, para, input, ref_input, rb_gt)
        img_list.append(init_tensor)
    img_list.append(rb_pred)
    nrow = input.shape[0] if input.shape[0] < args.test_save_n else args.test_save_n
    return img_list, nrow
