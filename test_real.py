import torch
from options  import test_opts
from utils    import logger, recorders
from datasets import data_loader
from models   import model_builder

from modules import real_tester as tester

args = test_opts.TestOpts().parse()
log  = logger.Logger(args)

def main(args):
    model_est = model_builder.buildEstModel(args)
    model_ref = model_builder.buildRefModel(args)
    models = [model_est, model_ref]
    test_loader = data_loader.realDataloader(args) 
    tester.test(args, 'test', test_loader, models, log)

if __name__ == '__main__':
    main(args)
