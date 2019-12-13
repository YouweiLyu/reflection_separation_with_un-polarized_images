import os
import torch
from models import model_utils
from utils import eval_utils
import numpy as np

def test(args, split, loader, models, log):
    models[0].eval()
    models[1].eval()
    log.printWrite('---- Start Test %d Images ----' % (len(loader)))
    with torch.no_grad():
        for i, sample in enumerate(loader):
            input_est = model_utils.parseRealData(args, sample)
            pred_est = models[0](input_est)
            input_ref = model_utils.getRealRefInput(args, pred_est, sample)
            pred = models[1](input_ref)
            eval_utils.saveSep(args, input_ref, pred, i)
            
