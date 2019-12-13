import argparse
import os
import torch

class TestOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        #### Test Dataset ####
        self.parser.add_argument('--dataset_test',  default='data_loader')
        self.parser.add_argument('--data_dir_val',  default='./data/syn/')
        self.parser.add_argument('--grayscale',     default=False, action='store_true')
        self.parser.add_argument('--resume',        default=None)
        self.parser.add_argument('--val_batch',     default=1,       type=int)
        self.parser.add_argument('--epochs',        default=0,       type=int)

        #### Device Arguments ####
        self.parser.add_argument('--cuda',        default=True,  action='store_false')
        self.parser.add_argument('--multi_gpu',   default=False, action='store_true')
        self.parser.add_argument('--time_sync',   default=False, action='store_true')
        self.parser.add_argument('--workers',     default=6,     type=int)
        self.parser.add_argument('--seed',        default=0,     type=int)

        #### Stage 1 Model Arguments ####
        self.parser.add_argument('--pretrain_est',default='./model/est.pth.tar')
        self.parser.add_argument('--resume_ref',  default=None)
        self.parser.add_argument('--num_para',    default=2,     type=int)
        self.parser.add_argument('--fc_height',   default=256,   type=int)
        self.parser.add_argument('--fc_width',    default=256,   type=int)

        #### Stage 2 Model Arguments ####
        self.parser.add_argument('--pretrain_ref', default='./model/ref.pth.tar')
        self.parser.add_argument('--resume_est',   default=None)
        self.parser.add_argument('--fg_gradient',  default=True,  action='store_false')
        self.parser.add_argument('--use_BN',       default=True,  action='store_false')
        self.parser.add_argument('--xi_zeta',      default=True,  action='store_false')
        self.parser.add_argument('--est_r_b',      default=True,  action='store_false')
        self.parser.add_argument('--show_init_sep',default=False, action='store_true')
        self.parser.add_argument('--phi_0',        default=0,     type=float)

        #### Displaying & Saving Arguments ####
        self.parser.add_argument('--ref_model',     default='RefNet')
        self.parser.add_argument('--est_model',     default='EstNet')
        self.parser.add_argument('--save_intv',     default=2,    type=int)
        self.parser.add_argument('--test_intv',     default=1,    type=int)
        self.parser.add_argument('--test_disp',     default=1,    type=int)
        self.parser.add_argument('--test_save',     default=1,    type=int)
        self.parser.add_argument('--max_train_iter',default=-1,   type=int)
        self.parser.add_argument('--max_val_iter',  default=-1,   type=int)
        self.parser.add_argument('--max_test_iter', default=-1,   type=int)
        self.parser.add_argument('--test_save_n',   default=5,    type=int)

        #### Log Arguments ####
        self.parser.add_argument('--save_root',  default='./log/')
        self.parser.add_argument('--item',       default='')
        self.parser.add_argument('--suffix',     default=None)
        self.parser.add_argument('--debug',      default=False, action='store_true')
        self.parser.add_argument('--make_dir',   default=True,  action='store_false')
        self.parser.add_argument('--save_split', default=False, action='store_true')

    def setDefault(self):
        if self.args.debug:
            self.args.max_train_iter = 2
            self.args.max_val_iter = 2
            self.args.max_test_iter = 2
            self.args.test_intv = 1
        if self.args.test_save_n > self.args.val_batch:
            self.args.test_save_n = self.args.val_batch
        self.collectInfo()

    def collectInfo(self):
        self.args.str_keys  = [
                ]
        self.args.val_keys  = [
                ]
        self.args.bool_keys = [
                ] 

    def parse(self):
        self.args = self.parser.parse_args()
        self.setDefault()
        return self.args
